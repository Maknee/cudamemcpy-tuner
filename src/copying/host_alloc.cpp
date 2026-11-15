#include "copying/host_alloc.h"

#include <sys/utsname.h>
#include <time.h>

#include <array>
#include <limits>
#include <mutex>
#include <optional>

#include "hook.h"

namespace cmt {
namespace host_alloc {
namespace {

using cu::CUcontext;
using cu::CUdeviceptr;
using cu::CUresult;

using HostAllocFn = CUresult (*)(void**, std::size_t, unsigned);
using AllocHostFn = CUresult (*)(void**, std::size_t);
using FreeHostFn = CUresult (*)(void*);
using GetFlagsFn = CUresult (*)(unsigned*, void*);
using RegisterFn = CUresult (*)(void*, std::size_t, unsigned);
using UnregisterFn = CUresult (*)(void*);
using DevicePointerFn = CUresult (*)(CUdeviceptr*, void*, unsigned);
using CtxGetCurrentFn = CUresult (*)(CUcontext*);
using PointerAttributesFn = CUresult (*)(unsigned, unsigned*, void**, CUdeviceptr);

constexpr unsigned kAllocPortable = 0x01;
constexpr unsigned kAllocDeviceMap = 0x02;
constexpr unsigned kRegisterFlags = 0x01 | 0x02;

// Shared mappings preserve the driver allocation behavior across fork().

constexpr std::size_t kMinBytes = std::size_t{2} << 20;
constexpr std::size_t kMinBytesPerThread = std::size_t{8} << 20;
constexpr std::size_t kMaxAlignment = std::size_t{32} << 20;
constexpr std::size_t kMaxAllocations = 4096;
constexpr unsigned kMaxThreads = 64;

constexpr CUresult kInvalidValue = 1;
constexpr CUresult kNotRegistered = 713;
unsigned ReportedFlags(unsigned requested)
{
	return requested | kAllocDeviceMap;
}

struct Allocation
{
	uintptr_t base = 0;
	std::size_t bytes = 0;
	unsigned flags = 0;
	bool releasing = false;
};

std::mutex g_mutex;
std::array<Allocation, kMaxAllocations> g_table{};
std::size_t g_count = 0;
std::atomic<bool> g_forked_child{false};

bool Insert(const Allocation& allocation)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	if (g_count == kMaxAllocations)
	{
		return false;
	}
	g_table[g_count++] = allocation;
	return true;
}

bool TableFull()
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	return g_count == kMaxAllocations;
}

std::optional<Allocation> Claim(uintptr_t base)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	for (std::size_t i = 0; i < g_count; ++i)
	{
		if (g_table[i].base == base && !g_table[i].releasing)
		{
			g_table[i].releasing = true;
			return g_table[i];
		}
	}
	return std::nullopt;
}

void Finish(uintptr_t base, bool remove)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	for (std::size_t i = 0; i < g_count; ++i)
	{
		if (g_table[i].base == base && g_table[i].releasing)
		{
			if (remove)
			{
				g_table[i] = g_table[--g_count];
			}
			else
			{
				g_table[i].releasing = false;
			}
			return;
		}
	}
}

bool Overlaps(uintptr_t address, std::size_t bytes)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	for (std::size_t i = 0; i < g_count; ++i)
	{
		const uintptr_t base = g_table[i].base;
		if (address < base + g_table[i].bytes && (base <= address || base - address < bytes))
		{
			return true;
		}
	}
	return false;
}

std::optional<Allocation> Find(uintptr_t address)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	for (std::size_t i = 0; i < g_count; ++i)
	{
		if (address - g_table[i].base < g_table[i].bytes)
		{
			return g_table[i];
		}
	}
	return std::nullopt;
}

enum class Symbol : std::uint8_t
{
	HostAlloc,
	AllocHost,
	FreeHost,
	GetFlags,
	Register,
	Unregister,
	DevicePointer,
	CtxGetCurrent,
	PointerAttributes,
	Count
};
constexpr const char* kSymbolNames[] = {"cuMemHostAlloc",
										"cuMemAllocHost_v2",
										"cuMemFreeHost",
										"cuMemHostGetFlags",
										"cuMemHostRegister_v2",
										"cuMemHostUnregister",
										"cuMemHostGetDevicePointer_v2",
										"cuCtxGetCurrent",
										"cuPointerGetAttributes"};
static_assert(std::size(kSymbolNames) == static_cast<std::size_t>(Symbol::Count));
std::array<std::atomic<void*>, static_cast<std::size_t>(Symbol::Count)> g_symbols{};

template <class Fn>
Fn Real(Symbol which)
{
	auto& slot = g_symbols[static_cast<std::size_t>(which)];
	void* p = slot.load(std::memory_order_acquire);
	if (p == nullptr)
	{
		p = CudaSymbol(kSymbolNames[static_cast<std::size_t>(which)]);
		if (p != nullptr)
		{
			slot.store(p, std::memory_order_release);
		}
	}
	return reinterpret_cast<Fn>(p);
}

double Milliseconds(const timespec& from, const timespec& to)
{
	return static_cast<double>(to.tv_sec - from.tv_sec) * 1e3 + static_cast<double>(to.tv_nsec - from.tv_nsec) / 1e6;
}

timespec Now()
{
	timespec t{};
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t;
}

enum class Fallback : std::uint8_t
{
	TableFull,
	Map,
	Register,
	DevicePointer,
	Count
};

void ReportOnce(Fallback kind, std::size_t bytes, const char* what, long code)
{
	static std::array<std::atomic<bool>, static_cast<std::size_t>(Fallback::Count)> said{};
	if (!said[static_cast<std::size_t>(kind)].exchange(true, std::memory_order_relaxed))
	{
		LogLine("note: cuda_host_alloc_hook: {} (code {}) for {} bytes; such allocations use the driver's own "
				"cuMemHostAlloc. Reported once.",
				what, code, bytes);
	}
	else
	{
		LogDebug("cuda_host_alloc_hook: {} (code {}) for {} bytes; driver allocation", what, code, bytes);
	}
}

void Unmap(void* p, std::size_t bytes)
{
	syscall(SYS_munmap, p, bytes);
}

void TouchRange(char* p, std::size_t bytes, std::size_t page)
{
	for (std::size_t offset = 0; offset < bytes; offset += page)
	{
		reinterpret_cast<volatile char*>(p)[offset] = 0;
	}
}

ThreadPlacement CallerNodePlacement()
{
	static const bool multi_node = CpuNodeCount() >= 2;
	ThreadPlacement placement;
	unsigned cpu = 0;
	unsigned node = 0;
	if (multi_node && syscall(SYS_getcpu, &cpu, &node, nullptr) == 0)
	{
		placement.numa_node = static_cast<int>(node);
		placement.automatic = true;
	}
	return placement;
}

unsigned Touch(char* base, std::size_t bytes, std::size_t page, std::size_t alignment)
{
	const std::size_t by_size = std::max<std::size_t>(1, bytes / kMinBytesPerThread);
	const unsigned threads = static_cast<unsigned>(std::min<std::size_t>(
		{std::size_t{GetConfig().cuda_host_alloc_threads}, by_size, std::size_t{kMaxThreads}}));
	std::size_t share = (bytes + threads - 1) / threads;
	share = (share + alignment - 1) / alignment * alignment;
	std::array<Thread, kMaxThreads> helpers;
	std::array<bool, kMaxThreads> started{};
	const ThreadPlacement placement = threads > 1 ? CallerNodePlacement() : ThreadPlacement{};
	for (unsigned i = 1; i < threads && i * share < bytes; ++i)
	{
		char* const from = base + i * share;
		const std::size_t n = std::min(share, bytes - i * share);
		started[i] = static_cast<bool>(helpers[i].Start([=] { TouchRange(from, n, page); }, placement));
	}
	TouchRange(base, std::min(share, bytes), page);
	unsigned ran = 1;
	for (unsigned i = 1; i < threads && i * share < bytes; ++i)
	{
		if (!started[i])
		{
			TouchRange(base + i * share, std::min(share, bytes - i * share), page);
			continue;
		}
		++ran;
		const auto joined = helpers[i].Join();
		if (!joined)
		{
			Panic(joined.error().Message());
		}
	}
	return ran;
}

bool KernelCopiesPinnedPagesAtFork()
{
	static const bool copies = []
	{
		utsname name{};
		unsigned major = 0;
		unsigned minor = 0;
		if (uname(&name) != 0 || std::sscanf(name.release, "%u.%u", &major, &minor) != 2)
		{
			return false;
		}
		return major > 5 || (major == 5 && minor >= 11);
	}();
	return copies;
}

std::size_t HugeAlignment()
{
	const std::size_t thp = ThpBytes();
	return thp <= kMaxAlignment ? thp : BasePageBytes();
}

std::optional<CUresult> Allocate(void** pointer, std::size_t bytes, unsigned flags)
{
	if (g_forked_child.load(std::memory_order_relaxed) || pointer == nullptr || bytes < kMinBytes ||
		(flags & ~(kAllocPortable | kAllocDeviceMap)) != 0 || pinning::detail::t_no_driver != 0)
	{
		LogDebug("cuda_host_alloc_hook: {} bytes, flags {:#x}: driver (not eligible)", bytes, flags);
		return std::nullopt;
	}
	const auto get_current = Real<CtxGetCurrentFn>(Symbol::CtxGetCurrent);
	const auto register_fn = Real<RegisterFn>(Symbol::Register);
	const auto unregister = Real<UnregisterFn>(Symbol::Unregister);
	const auto device_pointer = Real<DevicePointerFn>(Symbol::DevicePointer);
	CUcontext context = nullptr;
	if (get_current == nullptr || register_fn == nullptr || unregister == nullptr || device_pointer == nullptr ||
		get_current(&context) != cu::kSuccess || context == nullptr)
	{
		LogDebug("cuda_host_alloc_hook: {} bytes: driver (no current context or missing entry points)", bytes);
		return std::nullopt;
	}
	if (TableFull())
	{
		ReportOnce(Fallback::TableFull, bytes, "the allocation table is full", static_cast<long>(kMaxAllocations));
		return std::nullopt;
	}
	const std::size_t page = BasePageBytes();
	const std::size_t alignment = HugeAlignment();
	if (bytes > std::numeric_limits<std::size_t>::max() / 2)
	{
		return std::nullopt;
	}
	const std::size_t rounded = (bytes + page - 1) / page * page;
	const std::size_t reserved = rounded + alignment - page;
	const timespec start = Now();
	const bool private_memory = GetConfig().cuda_host_alloc_private;
	void* const raw = mmap(nullptr, reserved, PROT_READ | PROT_WRITE,
						   (private_memory ? MAP_PRIVATE : MAP_SHARED) | MAP_ANONYMOUS, -1, 0);
	if (raw == MAP_FAILED)
	{
		ReportOnce(Fallback::Map, bytes, "mmap failed", errno);
		return std::nullopt;
	}
	const uintptr_t raw_address = reinterpret_cast<uintptr_t>(raw);
	const uintptr_t aligned = (raw_address + alignment - 1) / alignment * alignment;
	char* const base = reinterpret_cast<char*>(aligned);
	if (aligned != raw_address)
	{
		Unmap(raw, aligned - raw_address);
	}
	if (raw_address + reserved != aligned + rounded)
	{
		Unmap(base + rounded, raw_address + reserved - (aligned + rounded));
	}
	if (syscall(SYS_madvise, base, rounded, MADV_HUGEPAGE) != 0)
	{
		LogDebug("cuda_host_alloc_hook: MADV_HUGEPAGE refused (errno {}); base pages", errno);
	}
	if (private_memory && !KernelCopiesPinnedPagesAtFork() && syscall(SYS_madvise, base, rounded, MADV_DONTFORK) != 0)
	{
		ReportOnce(Fallback::Map, bytes, "MADV_DONTFORK failed for private memory on this kernel", errno);
		Unmap(base, rounded);
		return std::nullopt;
	}
	const unsigned threads = Touch(base, rounded, page, alignment);
	const timespec touched = Now();
	const CUresult registered = register_fn(base, rounded, kRegisterFlags);
	const timespec pinned = Now();
	if (registered != cu::kSuccess)
	{
		ReportOnce(Fallback::Register, bytes, "cuMemHostRegister failed", registered);
		Unmap(base, rounded);
		return std::nullopt;
	}
	CUdeviceptr device = 0;
	const CUresult mapped = device_pointer(&device, base, 0);
	if (mapped != cu::kSuccess || device != static_cast<CUdeviceptr>(aligned))
	{
		ReportOnce(Fallback::DevicePointer, bytes,
				   mapped != cu::kSuccess ? "cuMemHostGetDevicePointer failed"
										  : "the device address of the registration is not its host address",
				   mapped);
		const CUresult released = unregister(base);
		if (released != cu::kSuccess)
		{
			LogLine("error: cuda_host_alloc_hook: cuMemHostUnregister failed (code {}) while undoing a {}-byte "
					"allocation; its {} bytes stay mapped",
					released, bytes, rounded);
			return std::nullopt;
		}
		Unmap(base, rounded);
		return std::nullopt;
	}
	if (!Insert(Allocation{aligned, rounded, flags}))
	{
		ReportOnce(Fallback::TableFull, bytes, "the allocation table is full", static_cast<long>(kMaxAllocations));
		if (unregister(base) == cu::kSuccess)
		{
			Unmap(base, rounded);
		}
		return std::nullopt;
	}
	*pointer = base;
	LogDebug("cuda_host_alloc_hook: {} bytes, flags {:#x}: {} {} at {}, touch {:.1f} ms ({} threads), register "
			 "{:.1f} ms",
			 bytes, flags, rounded, private_memory ? "private" : "shared", fmt::ptr(base),
			 Milliseconds(start, touched), threads, Milliseconds(touched, pinned));
	return cu::kSuccess;
}

CUresult HostAllocEntry(void** pointer, std::size_t bytes, unsigned flags) noexcept
{
	const logging::PreserveErrno preserve;
	if (const auto served = Allocate(pointer, bytes, flags))
	{
		return *served;
	}
	const auto real = Real<HostAllocFn>(Symbol::HostAlloc);
	return real != nullptr ? real(pointer, bytes, flags) : cu::kErrorNotFound;
}

CUresult AllocHostEntry(void** pointer, std::size_t bytes) noexcept
{
	const logging::PreserveErrno preserve;
	if (const auto served = Allocate(pointer, bytes, 0))
	{
		return *served;
	}
	const auto real = Real<AllocHostFn>(Symbol::AllocHost);
	return real != nullptr ? real(pointer, bytes) : cu::kErrorNotFound;
}

CUresult FreeHostEntry(void* pointer) noexcept
{
	const logging::PreserveErrno preserve;
	const auto real = Real<FreeHostFn>(Symbol::FreeHost);
	const auto unregister = Real<UnregisterFn>(Symbol::Unregister);
	std::optional<Allocation> own;
	if (!g_forked_child.load(std::memory_order_relaxed) && pointer != nullptr && unregister != nullptr)
	{
		own = Claim(reinterpret_cast<uintptr_t>(pointer));
	}
	if (!own)
	{
		return real != nullptr ? real(pointer) : cu::kErrorNotFound;
	}
	const CUresult released = unregister(pointer);
	if (released == kNotRegistered)
	{
		Finish(own->base, true);
		Unmap(pointer, own->bytes);
		LogDebug("cuda_host_alloc_hook: {} outlived its context; unmapped", pointer);
		return kInvalidValue;
	}
	if (released != cu::kSuccess)
	{
		Finish(own->base, false);
		LogDebug("cuda_host_alloc_hook: cuMemHostUnregister of {} failed (code {}); kept", pointer, released);
		return released;
	}
	Finish(own->base, true);
	Unmap(pointer, own->bytes);
	LogDebug("cuda_host_alloc_hook: freed {} ({} bytes)", pointer, own->bytes);
	return cu::kSuccess;
}

CUresult HostGetFlagsEntry(unsigned* flags, void* pointer) noexcept
{
	if (!g_forked_child.load(std::memory_order_relaxed) && flags != nullptr)
	{
		if (const auto own = Find(reinterpret_cast<uintptr_t>(pointer)))
		{
			*flags = ReportedFlags(own->flags);
			return cu::kSuccess;
		}
	}
	const auto real = Real<GetFlagsFn>(Symbol::GetFlags);
	return real != nullptr ? real(flags, pointer) : cu::kErrorNotFound;
}

CUresult HostRegisterEntry(void* pointer, std::size_t bytes, unsigned flags) noexcept
{
	if (!g_forked_child.load(std::memory_order_relaxed) && Overlaps(reinterpret_cast<uintptr_t>(pointer), bytes))
	{
		return kInvalidValue;
	}
	const auto real = Real<RegisterFn>(Symbol::Register);
	return real != nullptr ? real(pointer, bytes, flags) : cu::kErrorNotFound;
}

CUresult HostUnregisterEntry(void* pointer) noexcept
{
	if (!g_forked_child.load(std::memory_order_relaxed) && Find(reinterpret_cast<uintptr_t>(pointer)))
	{
		return kInvalidValue;
	}
	const auto real = Real<UnregisterFn>(Symbol::Unregister);
	return real != nullptr ? real(pointer) : cu::kErrorNotFound;
}

} // namespace

void AfterContextTeardown() noexcept
{
	if (g_forked_child.load(std::memory_order_relaxed) || !GetConfig().cuda_host_alloc_hook)
	{
		return;
	}
	const auto attributes = Real<PointerAttributesFn>(Symbol::PointerAttributes);
	if (attributes == nullptr)
	{
		return;
	}
	const logging::PreserveErrno preserve;
	const std::lock_guard<std::mutex> lock(g_mutex);
	for (std::size_t i = 0; i < g_count;)
	{
		const Allocation& allocation = g_table[i];
		unsigned attribute = cu::kPointerAttributeMemoryType;
		unsigned memory_type = cu::kMemoryTypeHost;
		void* value = &memory_type;
		if (!allocation.releasing &&
			attributes(1, &attribute, &value, static_cast<CUdeviceptr>(allocation.base)) == cu::kSuccess &&
			memory_type == 0)
		{
			LogDebug("cuda_host_alloc_hook: {:#x} ({} bytes) outlived its context; unmapped",
					 static_cast<unsigned long>(allocation.base), allocation.bytes);
			Unmap(reinterpret_cast<void*>(allocation.base), allocation.bytes);
			g_table[i] = g_table[--g_count];
			continue;
		}
		++i;
	}
}

namespace {

struct Entry
{
	const char* query;
	const char* symbol;
	void* replacement;
};

const std::array<Entry, 6>& Entries()
{
	static const std::array<Entry, 6> entries = {{
		{"cuMemHostAlloc", "cuMemHostAlloc", reinterpret_cast<void*>(&HostAllocEntry)},
		{"cuMemAllocHost", "cuMemAllocHost_v2", reinterpret_cast<void*>(&AllocHostEntry)},
		{"cuMemFreeHost", "cuMemFreeHost", reinterpret_cast<void*>(&FreeHostEntry)},
		{"cuMemHostGetFlags", "cuMemHostGetFlags", reinterpret_cast<void*>(&HostGetFlagsEntry)},
		{"cuMemHostRegister", "cuMemHostRegister_v2", reinterpret_cast<void*>(&HostRegisterEntry)},
		{"cuMemHostUnregister", "cuMemHostUnregister", reinterpret_cast<void*>(&HostUnregisterEntry)},
	}};
	return entries;
}

} // namespace

void* RedirectLookup(const char* symbol)
{
	if (symbol == nullptr || !GetConfig().cuda_host_alloc_hook)
	{
		return nullptr;
	}
	for (const Entry& entry : Entries())
	{
		if (std::strcmp(symbol, entry.symbol) == 0)
		{
			return entry.replacement;
		}
	}
	return nullptr;
}

void* RedirectQuery(const char* symbol, void* real)
{
	if (symbol == nullptr || real == nullptr || !GetConfig().cuda_host_alloc_hook)
	{
		return nullptr;
	}
	for (const Entry& entry : Entries())
	{
		if ((std::strcmp(symbol, entry.query) == 0 || std::strcmp(symbol, entry.symbol) == 0) &&
			real == CudaSymbol(entry.symbol))
		{
			return entry.replacement;
		}
	}
	return nullptr;
}

void AfterForkInChild()
{
	g_forked_child.store(true, std::memory_order_relaxed);
}

} // namespace host_alloc
} // namespace cmt
