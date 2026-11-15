#include "hook.h"

#if defined(CMT_TRACE_ONLY)

namespace cmt {
namespace trace {

struct TraceState
{
	std::atomic<bool> failed{false};
	std::atomic<bool> accepting{false};
	std::atomic<unsigned> in_flight{0};
	std::atomic<uint64_t> sequence{0};
	std::atomic<uint64_t> copies{0};
	std::atomic<uint64_t> copy_errors{0};
	int output_fd = -1;
	int status_fd = -1;
	pid_t owner = 0;
};

TraceState state;
thread_local unsigned depth = 0;

bool BypassTuning()
{
	return true;
}

bool IsActive()
{
	return state.accepting.load() && state.owner == getpid();
}

void WriteStatus(char code)
{
	if (state.status_fd >= 0)
	{
		const int saved = errno;
		ssize_t n;
		do
		{
			n = send(state.status_fd, &code, 1, MSG_NOSIGNAL | MSG_DONTWAIT);
		} while (n < 0 && errno == EINTR);
		errno = saved;
	}
}

void Start()
{
	const auto descriptor = [](const char* name)
	{
		const char* text = std::getenv(name);
		if (text == nullptr || *text == '\0')
		{
			return -1;
		}
		char* end = nullptr;
		const long value = std::strtol(text, &end, 10);
		return end != text && *end == '\0' && value >= 0 && value <= INT_MAX ? static_cast<int>(value) : -1;
	};
	const char* gate = std::getenv("CUDAMEMCPY_TUNER_TRACE_ENABLE");
	if (gate == nullptr || std::strcmp(gate, "1") != 0)
	{
		return;
	}
	state.owner = getpid();
	const int parent = descriptor("CUDAMEMCPY_TUNER_TRACE_PARENT_PID");
	if (parent >= 0 && getppid() != parent)
	{
		return;
	}
	state.status_fd = descriptor("CUDAMEMCPY_TUNER_TRACE_STATUS_FD");
	state.output_fd = descriptor("CUDAMEMCPY_TUNER_TRACE_FD");
	struct stat file{};
	const int flags = state.output_fd >= 0 ? fcntl(state.output_fd, F_GETFL) : -1;
	if (state.output_fd < 0 || fstat(state.output_fd, &file) != 0 || !S_ISREG(file.st_mode) || flags < 0 ||
		(flags & O_APPEND) == 0)
	{
		state.output_fd = -1;
		if (!state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
		return;
	}
	char text[1400];
	const int n = std::snprintf(
		text, sizeof(text),
		"{\"type\":\"trace_begin\",\"schema\":1,\"pid\":%ld,"
		"\"purpose\":\"CUDA API copy-size and memory-type distributions\","
		"\"bytes\":\"requested; failures are not transferred bytes\","
		"\"memory_basis\":\"cuPointerGetAttributes at first and last byte; not whole-range "
		"proof\","
		"\"coverage\":\"supported 1D cudaMemcpy and cuMemcpy APIs and CUDA host allocation "
		"lifecycle\","
		"\"excluded\":\"2D/3D, peer/batch/array APIs, graphics, graph replay transfers, ordinary "
		"malloc/mmap, child processes, calls before initialization or after trace teardown\","
		"\"deduplication\":\"outermost intercepted call on each thread\","
		"\"zero_records\":\"does not prove zero GPU copies\","
		"\"timing\":\"instrumented diagnostic; no GPU synchronization or latency "
		"measurement\"}\n",
		static_cast<long>(state.owner));
	if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(text))
	{
		if (!state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
	}
	else if (!state.failed.load(std::memory_order_relaxed))
	{
		ssize_t written;
		do
		{
			written = write(state.output_fd, text, static_cast<std::size_t>(n));
		} while (written < 0 && errno == EINTR);
		if (written != n && !state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
	}
	if (!state.failed.load(std::memory_order_relaxed))
	{
		WriteStatus('S');
	}
	state.accepting.store(!state.failed.load());
}

void Stop()
{
	if (!IsActive())
	{
		return;
	}
	state.accepting.store(false);
	if (state.in_flight.load() != 0)
	{
		if (!state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
		return;
	}
	char text[240];
	const int n =
		std::snprintf(text, sizeof(text),
					  "{\"type\":\"trace_end\",\"pid\":%ld,\"records\":%llu,\"copy_calls\":%llu,"
					  "\"failed_copy_calls\":%llu}\n",
					  static_cast<long>(state.owner), static_cast<unsigned long long>(state.sequence.load()),
					  static_cast<unsigned long long>(state.copies.load()),
					  static_cast<unsigned long long>(state.copy_errors.load()));
	if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(text))
	{
		if (!state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
	}
	else if (!state.failed.load(std::memory_order_relaxed))
	{
		ssize_t written;
		do
		{
			written = write(state.output_fd, text, static_cast<std::size_t>(n));
		} while (written < 0 && errno == EINTR);
		if (written != n && !state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
	}
	if (!state.failed.load(std::memory_order_relaxed))
	{
		WriteStatus('E');
	}
}

thread_local bool resolving = false;
thread_local const void* forwarding_origin = nullptr;
constexpr std::size_t kBindingsPerApi = 16;

void* Original(const char* symbol, const void* caller, void* self = nullptr, bool module_first = false)
{
	if (resolving)
	{
		if (!state.failed.exchange(true, std::memory_order_relaxed))
		{
			WriteStatus('F');
		}
		return nullptr;
	}

	struct Guard
	{
		Guard()
		{
			resolving = true;
		}

		~Guard()
		{
			resolving = false;
		}
	} guard;

	const auto loader = GetLoaderFunctions();
	const auto lookup = loader.lookup;
	const auto open = loader.open;
	if (lookup == nullptr || open == nullptr)
	{
		return nullptr;
	}
	const auto from_image = [&](const void* address) -> void*
	{
		Dl_info image{};
		if (address == nullptr || dladdr(address, &image) == 0)
		{
			return nullptr;
		}
		void* handle = open(image.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
		if (handle == nullptr)
		{
			handle = open(nullptr, RTLD_LAZY);
		}
		if (handle == nullptr)
		{
			return nullptr;
		}
		void* found = lookup(handle, symbol);
		dlclose(handle);
		return found == self ? nullptr : found;
	};
	const auto from_module = [&]() -> void*
	{
		void* found = from_image(caller);
		if (found == nullptr && forwarding_origin != nullptr && forwarding_origin != caller)
		{
			found = from_image(forwarding_origin);
		}
		return found;
	};
	void* found = module_first ? from_module() : nullptr;
	if (found == nullptr)
	{
		found = lookup(RTLD_NEXT, symbol);
	}
	if (found == self)
	{
		found = nullptr;
	}
	if (found == nullptr && !module_first)
	{
		found = from_module();
	}
	if (found == nullptr && std::strncmp(symbol, "cu", 2) == 0 && std::strncmp(symbol, "cuda", 4) != 0)
	{
		found = CudaSymbol(symbol);
	}
	return found == self ? nullptr : found;
}

struct Scope
{
	int saved_errno = errno;
	bool record = false;
	const void* previous_origin = forwarding_origin;

	explicit Scope(const void* origin)
	{
		forwarding_origin = origin;
		if (depth++ == 0 && IsActive())
		{
			state.in_flight.fetch_add(1);
			if (state.accepting.load())
			{
				record = true;
			}
			else
			{
				state.in_flight.fetch_sub(1);
			}
		}
	}

	~Scope()
	{
		if (record)
		{
			state.in_flight.fetch_sub(1);
		}
		--depth;
		forwarding_origin = previous_origin;
	}
};

using Query = int (*)(unsigned, const unsigned*, void**, cu::CUdeviceptr);

const char* ClassifyPointer(Query fn, const void* pointer)
{
	if (fn == nullptr || pointer == nullptr)
	{
		return "unknown";
	}
	unsigned type = 0;
	unsigned managed = 0;
	const unsigned attributes[] = {cu::kPointerAttributeMemoryType, 8};
	void* values[] = {&type, &managed};
	if (fn(2, attributes, values, static_cast<cu::CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer))) != 0)
	{
		return "unknown";
	}
	if (managed != 0 || type == cu::kMemoryTypeUnified)
	{
		return "managed";
	}
	if (type == cu::kMemoryTypeHost)
	{
		return "pinned";
	}
	if (type == 2)
	{
		return "device";
	}
	return type == 0 ? "pageable" : "unknown";
}

struct Endpoints
{
	const char* first = "not_applicable";
	const char* last = "not_applicable";
};

Endpoints ClassifyEndpoints(Query query, const void* pointer, std::size_t bytes)
{
	if (bytes == 0)
	{
		return {};
	}
	const uintptr_t base = reinterpret_cast<uintptr_t>(pointer);
	if (pointer == nullptr || bytes - 1 > UINTPTR_MAX - base)
	{
		return {"unknown", "unknown"};
	}
	return {ClassifyPointer(query, pointer),
			ClassifyPointer(query, reinterpret_cast<const void*>(base + bytes - 1))};
}

template <class Fn, class... Args>
int Copy(Fn fn, const char* api, const char* direction, const void* src, const void* dst, std::size_t bytes,
		 bool async, Args... args)
{
	Scope scope(reinterpret_cast<void*>(fn));
	Endpoints source;
	Endpoints destination;
	if (scope.record)
	{
		const auto query = reinterpret_cast<Query>(
			Original("cuPointerGetAttributes", reinterpret_cast<void*>(fn), nullptr, true));
		source = ClassifyEndpoints(query, src, bytes);
		destination = ClassifyEndpoints(query, dst, bytes);
	}
	errno = scope.saved_errno;
	const int rc = fn != nullptr ? fn(args...) : cu::kErrorNotFound;
	const int result_errno = errno;
	if (scope.record)
	{
		state.copies.fetch_add(1, std::memory_order_relaxed);
		if (rc != 0)
		{
			state.copy_errors.fetch_add(1, std::memory_order_relaxed);
		}
		char text[800];
		const int n = std::snprintf(
			text, sizeof(text),
			"{\"type\":\"copy\",\"seq\":%llu,\"pid\":%ld,\"tid\":%ld,\"api\":\"%s\","
			"\"direction\":\"%s\",\"bytes_requested\":%zu,\"async_api\":%s,\"result\":%d,"
			"\"src_first\":\"%s\",\"src_last\":\"%s\",\"dst_first\":\"%s\",\"dst_last\":\"%s\"}"
			"\n",
			static_cast<unsigned long long>(state.sequence.fetch_add(1, std::memory_order_relaxed)),
			static_cast<long>(state.owner), static_cast<long>(syscall(SYS_gettid)), api, direction, bytes,
			async ? "true" : "false", rc, source.first, source.last, destination.first, destination.last);
		if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(text))
		{
			if (!state.failed.exchange(true, std::memory_order_relaxed))
			{
				WriteStatus('F');
			}
		}
		else if (!state.failed.load(std::memory_order_relaxed))
		{
			ssize_t written;
			do
			{
				written = write(state.output_fd, text, static_cast<std::size_t>(n));
			} while (written < 0 && errno == EINTR);
			if (written != n && !state.failed.exchange(true, std::memory_order_relaxed))
			{
				WriteStatus('F');
			}
		}
	}
	errno = result_errno;
	return rc;
}

template <class Fn, class... Args>
int Allocation(Fn fn, const char* api, const char* operation, std::size_t bytes, bool size_known,
			   Args... args)
{
	Scope scope(reinterpret_cast<void*>(fn));
	errno = scope.saved_errno;
	const int rc = fn != nullptr ? fn(args...) : cu::kErrorNotFound;
	const int result_errno = errno;
	if (scope.record)
	{
		char size[32];
		char text[400];
		if (size_known)
		{
			std::snprintf(size, sizeof(size), "%zu", bytes);
		}
		else
		{
			std::strcpy(size, "null");
		}
		const int n = std::snprintf(
			text, sizeof(text),
			"{\"type\":\"host_allocation\",\"seq\":%llu,\"pid\":%ld,\"tid\":%ld,\"api\":\"%s\","
			"\"operation\":\"%s\",\"bytes_requested\":%s,\"result\":%d}\n",
			static_cast<unsigned long long>(state.sequence.fetch_add(1, std::memory_order_relaxed)),
			static_cast<long>(state.owner), static_cast<long>(syscall(SYS_gettid)), api, operation, size, rc);
		if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(text))
		{
			if (!state.failed.exchange(true, std::memory_order_relaxed))
			{
				WriteStatus('F');
			}
		}
		else if (!state.failed.load(std::memory_order_relaxed))
		{
			ssize_t written;
			do
			{
				written = write(state.output_fd, text, static_cast<std::size_t>(n));
			} while (written < 0 && errno == EINTR);
			if (written != n && !state.failed.exchange(true, std::memory_order_relaxed))
			{
				WriteStatus('F');
			}
		}
	}
	errno = result_errno;
	return rc;
}

const char* RuntimeDirection(int kind)
{
	switch (kind)
	{
	case cu::cudaMemcpyHostToHost:
		return "htoh";
	case cu::cudaMemcpyHostToDevice:
		return "htod";
	case cu::cudaMemcpyDeviceToHost:
		return "dtoh";
	case cu::cudaMemcpyDeviceToDevice:
		return "dtod";
	case cu::cudaMemcpyDefault:
		return "default";
	default:
		return "unknown";
	}
}

} // namespace trace
} // namespace cmt

#define CMT_TRACE_COPY(name, signature, arguments, direction, source, destination, bytes, asynchronous)      \
	namespace cmt {                                                                                          \
	namespace trace_slots {                                                                                  \
	using name##_fn = int(*) signature;                                                                      \
	std::array<std::atomic<name##_fn>, trace::kBindingsPerApi> name{};                                       \
	}                                                                                                        \
	}                                                                                                        \
	namespace cmt {                                                                                          \
	namespace trace_entries {                                                                                \
	template <std::size_t Index>                                                                             \
	int name signature noexcept                                                                              \
	{                                                                                                        \
		return trace::Copy(trace_slots::name[Index].load(std::memory_order_acquire), #name, direction,       \
						   source, destination, bytes, asynchronous, CMT_ARGS arguments);                    \
	}                                                                                                        \
	template <std::size_t... Indices>                                                                        \
	constexpr auto name##_wrappers(std::index_sequence<Indices...>)                                          \
	{                                                                                                        \
		return std::array<trace_slots::name##_fn, sizeof...(Indices)>{&name<Indices>...};                    \
	}                                                                                                        \
	}                                                                                                        \
	}                                                                                                        \
	extern "C" __attribute__((visibility("default"))) int name signature noexcept                            \
	{                                                                                                        \
		const int saved = errno;                                                                             \
		const auto real = reinterpret_cast<cmt::trace_slots::name##_fn>(                                     \
			cmt::trace::Original(#name, __builtin_extract_return_addr(__builtin_return_address(0)),          \
								 reinterpret_cast<void*>(&::name)));                                         \
		errno = saved;                                                                                       \
		return cmt::trace::Copy(real, #name, direction, source, destination, bytes, asynchronous,            \
								CMT_ARGS arguments);                                                         \
	}
#define CMT_TRACE_ALLOC(name, signature, arguments, operation, bytes, known)                                 \
	namespace cmt {                                                                                          \
	namespace trace_slots {                                                                                  \
	using name##_fn = int(*) signature;                                                                      \
	std::array<std::atomic<name##_fn>, trace::kBindingsPerApi> name{};                                       \
	}                                                                                                        \
	}                                                                                                        \
	namespace cmt {                                                                                          \
	namespace trace_entries {                                                                                \
	template <std::size_t Index>                                                                             \
	int name signature noexcept                                                                              \
	{                                                                                                        \
		return trace::Allocation(trace_slots::name[Index].load(std::memory_order_acquire), #name, operation, \
								 bytes, known, CMT_ARGS arguments);                                          \
	}                                                                                                        \
	template <std::size_t... Indices>                                                                        \
	constexpr auto name##_wrappers(std::index_sequence<Indices...>)                                          \
	{                                                                                                        \
		return std::array<trace_slots::name##_fn, sizeof...(Indices)>{&name<Indices>...};                    \
	}                                                                                                        \
	}                                                                                                        \
	}                                                                                                        \
	extern "C" __attribute__((visibility("default"))) int name signature noexcept                            \
	{                                                                                                        \
		const int saved = errno;                                                                             \
		const auto real = reinterpret_cast<cmt::trace_slots::name##_fn>(                                     \
			cmt::trace::Original(#name, __builtin_extract_return_addr(__builtin_return_address(0)),          \
								 reinterpret_cast<void*>(&::name)));                                         \
		errno = saved;                                                                                       \
		return cmt::trace::Allocation(real, #name, operation, bytes, known, CMT_ARGS arguments);             \
	}
#define CMT_ARGS(...) __VA_ARGS__
#define CMT_PTR(value) reinterpret_cast<const void*>(static_cast<uintptr_t>(value))

CMT_TRACE_COPY(cudaMemcpy, (void* dst, const void* src, size_t n, int kind), (dst, src, n, kind),
			   cmt::trace::RuntimeDirection(kind), src, dst, n, false)
CMT_TRACE_COPY(cudaMemcpyAsync, (void* dst, const void* src, size_t n, int kind, void* stream),
			   (dst, src, n, kind, stream), cmt::trace::RuntimeDirection(kind), src, dst, n, true)
CMT_TRACE_COPY(cudaMemcpy_ptds, (void* dst, const void* src, size_t n, int kind), (dst, src, n, kind),
			   cmt::trace::RuntimeDirection(kind), src, dst, n, false)
CMT_TRACE_COPY(cudaMemcpyAsync_ptsz, (void* dst, const void* src, size_t n, int kind, void* stream),
			   (dst, src, n, kind, stream), cmt::trace::RuntimeDirection(kind), src, dst, n, true)
CMT_TRACE_COPY(cuMemcpyHtoD_v2, (unsigned long long dst, const void* src, size_t n), (dst, src, n), "htod",
			   src, CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpyHtoD_v2_ptds, (unsigned long long dst, const void* src, size_t n), (dst, src, n),
			   "htod", src, CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpyHtoDAsync_v2, (unsigned long long dst, const void* src, size_t n, void* stream),
			   (dst, src, n, stream), "htod", src, CMT_PTR(dst), n, true)
CMT_TRACE_COPY(cuMemcpyHtoDAsync_v2_ptsz, (unsigned long long dst, const void* src, size_t n, void* stream),
			   (dst, src, n, stream), "htod", src, CMT_PTR(dst), n, true)
CMT_TRACE_COPY(cuMemcpyDtoH_v2, (void* dst, unsigned long long src, size_t n), (dst, src, n), "dtoh",
			   CMT_PTR(src), dst, n, false)
CMT_TRACE_COPY(cuMemcpyDtoH_v2_ptds, (void* dst, unsigned long long src, size_t n), (dst, src, n), "dtoh",
			   CMT_PTR(src), dst, n, false)
CMT_TRACE_COPY(cuMemcpyDtoHAsync_v2, (void* dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "dtoh", CMT_PTR(src), dst, n, true)
CMT_TRACE_COPY(cuMemcpyDtoHAsync_v2_ptsz, (void* dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "dtoh", CMT_PTR(src), dst, n, true)
CMT_TRACE_COPY(cuMemcpyDtoD_v2, (unsigned long long dst, unsigned long long src, size_t n), (dst, src, n),
			   "dtod", CMT_PTR(src), CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpyDtoD_v2_ptds, (unsigned long long dst, unsigned long long src, size_t n),
			   (dst, src, n), "dtod", CMT_PTR(src), CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpyDtoDAsync_v2, (unsigned long long dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "dtod", CMT_PTR(src), CMT_PTR(dst), n, true)
CMT_TRACE_COPY(cuMemcpyDtoDAsync_v2_ptsz,
			   (unsigned long long dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "dtod", CMT_PTR(src), CMT_PTR(dst), n, true)
CMT_TRACE_COPY(cuMemcpy, (unsigned long long dst, unsigned long long src, size_t n), (dst, src, n), "default",
			   CMT_PTR(src), CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpy_ptds, (unsigned long long dst, unsigned long long src, size_t n), (dst, src, n),
			   "default", CMT_PTR(src), CMT_PTR(dst), n, false)
CMT_TRACE_COPY(cuMemcpyAsync, (unsigned long long dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "default", CMT_PTR(src), CMT_PTR(dst), n, true)
CMT_TRACE_COPY(cuMemcpyAsync_ptsz, (unsigned long long dst, unsigned long long src, size_t n, void* stream),
			   (dst, src, n, stream), "default", CMT_PTR(src), CMT_PTR(dst), n, true)
CMT_TRACE_ALLOC(cudaMallocHost, (void** p, size_t n), (p, n), "allocate_pinned", n, true)
CMT_TRACE_ALLOC(cudaHostAlloc, (void** p, size_t n, unsigned flags), (p, n, flags), "allocate_pinned", n,
				true)
CMT_TRACE_ALLOC(cudaHostRegister, (void* p, size_t n, unsigned flags), (p, n, flags), "register_pinned", n,
				true)
CMT_TRACE_ALLOC(cudaFreeHost, (void* p), (p), "free_pinned", 0, false)
CMT_TRACE_ALLOC(cudaHostUnregister, (void* p), (p), "unregister", 0, false)
CMT_TRACE_ALLOC(cuMemAllocHost_v2, (void** p, size_t n), (p, n), "allocate_pinned", n, true)
CMT_TRACE_ALLOC(cuMemHostAlloc, (void** p, size_t n, unsigned flags), (p, n, flags), "allocate_pinned", n,
				true)
CMT_TRACE_ALLOC(cuMemHostRegister_v2, (void* p, size_t n, unsigned flags), (p, n, flags), "register_pinned",
				n, true)
CMT_TRACE_ALLOC(cuMemFreeHost, (void* p), (p), "free_pinned", 0, false)
CMT_TRACE_ALLOC(cuMemHostUnregister, (void* p), (p), "unregister", 0, false)

#undef CMT_TRACE_COPY
#undef CMT_TRACE_ALLOC
#undef CMT_ARGS
#undef CMT_PTR

namespace cmt {
namespace trace_slots {
using Proc = int (*)(const char*, void**, int, unsigned long long);
using ProcV2 = int (*)(const char*, void**, int, unsigned long long, void*);
std::array<std::atomic<Proc>, trace::kBindingsPerApi> cuGetProcAddress{};
std::array<std::atomic<ProcV2>, trace::kBindingsPerApi> cuGetProcAddress_v2{};
} // namespace trace_slots
} // namespace cmt

namespace cmt {
namespace trace_entries {
int QueryProc(trace_slots::Proc real, const char* symbol, void** fn, int version, unsigned long long flags)
{
	const int rc = real ? real(symbol, fn, version, flags) : cu::kErrorNotFound;
	const int saved = errno;
	if (rc == 0 && fn != nullptr && *fn != nullptr && trace::BypassTuning())
	{
		const bool per_thread_stream = (flags & cu::kGetProcAddressPerThreadDefaultStream) != 0;
		if (void* wrapped = trace::Redirect(symbol, *fn, per_thread_stream, version))
		{
			*fn = wrapped;
		}
	}
	errno = saved;
	return rc;
}

int QueryProcV2(trace_slots::ProcV2 real, const char* symbol, void** fn, int version,
				unsigned long long flags, void* query_status)
{
	const int rc = real ? real(symbol, fn, version, flags, query_status) : cu::kErrorNotFound;
	const int saved = errno;
	int status = 0;
	if (rc == 0 && query_status != nullptr)
	{
		std::memcpy(&status, query_status, sizeof(status));
	}
	if (rc == 0 && fn != nullptr && *fn != nullptr && status == 0 && trace::BypassTuning())
	{
		const bool per_thread_stream = (flags & cu::kGetProcAddressPerThreadDefaultStream) != 0;
		if (void* wrapped = trace::Redirect(symbol, *fn, per_thread_stream, version))
		{
			*fn = wrapped;
		}
	}
	errno = saved;
	return rc;
}

template <std::size_t Index>
int cuGetProcAddress(const char* symbol, void** fn, int version, unsigned long long flags) noexcept
{
	return QueryProc(trace_slots::cuGetProcAddress[Index].load(std::memory_order_acquire), symbol, fn,
					 version, flags);
}

template <std::size_t Index>
int cuGetProcAddress_v2(const char* symbol, void** fn, int version, unsigned long long flags,
						void* status) noexcept
{
	return QueryProcV2(trace_slots::cuGetProcAddress_v2[Index].load(std::memory_order_acquire), symbol, fn,
					   version, flags, status);
}

template <std::size_t... Indices>
constexpr auto cuGetProcAddress_wrappers(std::index_sequence<Indices...>)
{
	return std::array<trace_slots::Proc, sizeof...(Indices)>{&cuGetProcAddress<Indices>...};
}

template <std::size_t... Indices>
constexpr auto cuGetProcAddress_v2_wrappers(std::index_sequence<Indices...>)
{
	return std::array<trace_slots::ProcV2, sizeof...(Indices)>{&cuGetProcAddress_v2<Indices>...};
}
} // namespace trace_entries
} // namespace cmt

extern "C" __attribute__((visibility("default"))) int
cuGetProcAddress(const char* symbol, void** fn, int version, unsigned long long flags) noexcept
{
	const int saved = errno;
	const auto real = reinterpret_cast<cmt::trace_slots::Proc>(
		cmt::trace::Original("cuGetProcAddress", __builtin_extract_return_addr(__builtin_return_address(0)),
							 reinterpret_cast<void*>(&::cuGetProcAddress)));
	errno = saved;
	return cmt::trace_entries::QueryProc(real, symbol, fn, version, flags);
}

extern "C" __attribute__((visibility("default"))) int cuGetProcAddress_v2(const char* symbol, void** fn,
																		  int version,
																		  unsigned long long flags,
																		  void* query_status) noexcept
{
	const int saved = errno;
	const auto real = reinterpret_cast<cmt::trace_slots::ProcV2>(cmt::trace::Original(
		"cuGetProcAddress_v2", __builtin_extract_return_addr(__builtin_return_address(0)),
		reinterpret_cast<void*>(&::cuGetProcAddress_v2)));
	errno = saved;
	return cmt::trace_entries::QueryProcV2(real, symbol, fn, version, flags, query_status);
}

namespace cmt {
namespace trace {
template <class Fn, std::size_t Count>
void* Bind(std::array<std::atomic<Fn>, Count>& slots, void* real, const std::array<Fn, Count>& wrappers)
{
	const Fn target = reinterpret_cast<Fn>(real);
	for (std::size_t index = 0; index < Count; ++index)
	{
		if (target == wrappers[index])
		{
			return nullptr;
		}
	}
	for (std::size_t index = 0; index < Count; ++index)
	{
		Fn expected = nullptr;
		if (slots[index].compare_exchange_strong(expected, target, std::memory_order_acq_rel) ||
			expected == target)
		{
			return reinterpret_cast<void*>(wrappers[index]);
		}
	}
	Panic("trace exhausted distinct library bindings for one CUDA API");
}

void* Redirect(const char* symbol, void* real, bool per_thread, int version)
{
	if (symbol == nullptr || real == nullptr || !BypassTuning())
	{
		return nullptr;
	}
	char versioned[80];
	const char* name = symbol;
	if (version > 0)
	{
		if (version < 3020)
		{
			return nullptr;
		}
		if (std::strcmp(symbol, "cuGetProcAddress") == 0)
		{
			if (version < 11030)
			{
				return nullptr;
			}
			name = version >= 12000 ? "cuGetProcAddress_v2" : "cuGetProcAddress";
		}
		for (const char* base :
			 {"cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD", "cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync",
			  "cuMemcpyDtoDAsync", "cuMemAllocHost", "cuMemHostRegister"})
		{
			if (std::strcmp(symbol, base) == 0)
			{
				std::snprintf(versioned, sizeof(versioned), "%s_v2", base);
				name = versioned;
				break;
			}
		}
	}
	char threaded[96];
	if (per_thread && std::strncmp(name, "cuMemcpy", 8) == 0 && std::strstr(name, "_pt") == nullptr)
	{
		std::snprintf(threaded, sizeof(threaded), "%s_%s", name,
					  std::strstr(name, "Async") != nullptr ? "ptsz" : "ptds");
		name = threaded;
	}
#define CMT_BIND(api)                                                                                        \
	if (std::strcmp(name, #api) == 0)                                                                        \
	{                                                                                                        \
		if (real == reinterpret_cast<void*>(&::api))                                                         \
		{                                                                                                    \
			return nullptr;                                                                                  \
		}                                                                                                    \
		return Bind(trace_slots::api, real,                                                                  \
					trace_entries::api##_wrappers(std::make_index_sequence<kBindingsPerApi>{}));             \
	}
	CMT_BIND(cudaMemcpy);
	CMT_BIND(cudaMemcpyAsync);
	CMT_BIND(cudaMemcpy_ptds);
	CMT_BIND(cudaMemcpyAsync_ptsz);
	CMT_BIND(cuMemcpyHtoD_v2);
	CMT_BIND(cuMemcpyHtoD_v2_ptds);
	CMT_BIND(cuMemcpyHtoDAsync_v2);
	CMT_BIND(cuMemcpyHtoDAsync_v2_ptsz);
	CMT_BIND(cuMemcpyDtoH_v2);
	CMT_BIND(cuMemcpyDtoH_v2_ptds);
	CMT_BIND(cuMemcpyDtoHAsync_v2);
	CMT_BIND(cuMemcpyDtoHAsync_v2_ptsz);
	CMT_BIND(cuMemcpyDtoD_v2);
	CMT_BIND(cuMemcpyDtoD_v2_ptds);
	CMT_BIND(cuMemcpyDtoDAsync_v2);
	CMT_BIND(cuMemcpyDtoDAsync_v2_ptsz);
	CMT_BIND(cuMemcpy);
	CMT_BIND(cuMemcpy_ptds);
	CMT_BIND(cuMemcpyAsync);
	CMT_BIND(cuMemcpyAsync_ptsz);
	CMT_BIND(cudaMallocHost);
	CMT_BIND(cudaHostAlloc);
	CMT_BIND(cudaHostRegister);
	CMT_BIND(cudaFreeHost);
	CMT_BIND(cudaHostUnregister);
	CMT_BIND(cuMemAllocHost_v2);
	CMT_BIND(cuMemHostAlloc);
	CMT_BIND(cuMemHostRegister_v2);
	CMT_BIND(cuMemFreeHost);
	CMT_BIND(cuMemHostUnregister);
	CMT_BIND(cuGetProcAddress);
	CMT_BIND(cuGetProcAddress_v2);
#undef CMT_BIND
	return nullptr;
}
} // namespace trace
} // namespace cmt

#else

namespace cmt {
namespace trace {

void Start()
{
}

void Stop()
{
}

bool BypassTuning()
{
	return false;
}

void* Redirect(const char*, void*, bool, int)
{
	return nullptr;
}

} // namespace trace
} // namespace cmt

#endif
