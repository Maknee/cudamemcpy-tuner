// Real-GPU checks of what the fake libcuda cannot show: the hook must not change what cudaMemcpy means.
//
// ctest runs this binary without the hook (the stock driver: proof that every check below is a property of
// CUDA, not of the test) and with the hook under a tuning table that sends every size through the chunks
// pipeline. A check that passes stock and fails hooked is a hook bug. Exit: 0 all hold, 1 a check failed, 77
// no GPU.
//
//   order_dtoh   a slow kernel writes D on a non-blocking stream S, then a pageable DtoH copy on S. The copy
//   must
//                see the kernel's bytes: a copy that runs early (legacy stream, a helper stream without a
//                fence) reads stale ones.
//   order_*_g    order_dtoh and order_htod (below) with cudaMemcpyDefault: the generic copy keeps the stream.
//   order_htod   a slow kernel reads D into D2 on S, then a pageable HtoD copy on S overwrites D. The kernel
//   must
//                read the OLD bytes: a copy that runs early overwrites D under it.
//   capture      a pageable HtoD copy inside stream capture. The driver may refuse it (fine); if the capture
//                succeeds, every replay must copy the host bytes as they are AT REPLAY. A copy staged through
//                a reused pinned slot at capture time would replay old bytes forever.
//   capture_dflt a captured pageable cudaMemcpyDefault records the generic call's node (4/4) and replays the
//   host
//                bytes as they are at replay.
//   generic      cudaMemcpy(Async) with cudaMemcpyDefault (the driver's generic cuMemcpy) between pageable
//   memory
//                and cudaMalloc / VMM / pool memory, and between managed, pinned and device memory: the bytes
//                land, and (hooked, through the hook's copy audit, gpu_audit.h) the first kind reaches the
//                hook and the second does not (generic_copies below).
//   ptds         (built with --default-stream per-thread) order_dtoh on stream 0 from two threads at once:
//   each
//                thread's stream 0 is its own per-thread stream, and the copy must follow that thread's
//                kernel.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "gpu_audit.h"

namespace {

int g_failed = 0;

#define CK(call)                                                                                             \
	do                                                                                                       \
	{                                                                                                        \
		const cudaError_t e_ = (call);                                                                       \
		if (e_ != cudaSuccess)                                                                               \
		{                                                                                                    \
			std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));      \
			std::exit(1);                                                                                    \
		}                                                                                                    \
	} while (0)

void expect(bool ok, const char* check, std::size_t bytes, const char* what)
{
	std::printf("%-4s %-11s %10zu  %s\n", ok ? "ok" : "FAIL", check, bytes, what);
	if (!ok)
	{
		g_failed++;
	}
}

// Spin for `cycles` GPU clocks, then write `v` to every 32-bit word: a writer that is still running long
// after the host has queued the copy behind it (~1.5 ms at 1.7 GHz for 2.5M cycles).
__global__ void slow_fill(std::uint32_t* d, std::size_t n, std::uint32_t v, long long cycles)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		const long long t0 = clock64();
		while (clock64() - t0 < cycles)
		{
		}
	}
	__syncthreads();
	for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
	{
		d[i] = v;
	}
}

#ifndef HOOK_TEST_PTDS // the per-thread build runs only order_dtoh
// One block: spin, then copy src -> dst. A single block makes "spin first" hold for every word it copies.
__global__ void slow_copy(const std::uint32_t* src, std::uint32_t* dst, std::size_t n, long long cycles)
{
	if (threadIdx.x == 0)
	{
		const long long t0 = clock64();
		while (clock64() - t0 < cycles)
		{
		}
	}
	__syncthreads();
	for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
	{
		dst[i] = src[i];
	}
}
#endif // HOOK_TEST_PTDS

constexpr long long kSpin = 2500000;

/// Pageable host memory the way applications get it (malloc), not cudaMallocHost.
std::uint32_t* pageable(std::size_t bytes)
{
	auto* p = static_cast<std::uint32_t*>(std::malloc(bytes));
	if (!p)
	{
		std::exit(1);
	}
	std::memset(p, 0, bytes);
	return p;
}

bool all_equal(const std::uint32_t* h, std::size_t words, std::uint32_t v)
{
	for (std::size_t i = 0; i < words; i++)
	{
		if (h[i] != v)
		{
			return false;
		}
	}
	return true;
}

void order_dtoh(std::size_t bytes, cudaStream_t s, const char* name, std::uint32_t v,
				cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
	const std::size_t n = bytes / 4;
	std::uint32_t* d = nullptr;
	CK(cudaMalloc(&d, bytes));
	CK(cudaMemsetAsync(d, 0, bytes, s));
	std::uint32_t* h = pageable(bytes);
	slow_fill<<<64, 256, 0, s>>>(d, n, v, kSpin);
	CK(cudaGetLastError());
	CK(cudaMemcpyAsync(h, d, bytes, kind, s));
	CK(cudaStreamSynchronize(s));
	expect(all_equal(h, n, v), name, bytes, "copy sees the kernel's bytes");
	std::free(h);
	CK(cudaFree(d));
}

#ifndef HOOK_TEST_PTDS
void order_htod(std::size_t bytes, cudaStream_t s, const char* name = "order_htod",
				cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	const std::size_t n = bytes / 4;
	std::uint32_t *d = nullptr, *d2 = nullptr;
	CK(cudaMalloc(&d, bytes));
	CK(cudaMalloc(&d2, bytes));
	std::vector<std::uint32_t> old_bytes(n, 0x11111111u);
	CK(cudaMemcpy(d, std::data(old_bytes), bytes, cudaMemcpyHostToDevice));
	std::uint32_t* h = pageable(bytes);
	for (std::size_t i = 0; i < n; i++)
	{
		h[i] = 0x22222222u;
	}
	slow_copy<<<1, 1024, 0, s>>>(d, d2, n, kSpin);
	CK(cudaGetLastError());
	CK(cudaMemcpyAsync(d, h, bytes, kind, s));
	CK(cudaStreamSynchronize(s));
	std::vector<std::uint32_t> got(n), got2(n);
	CK(cudaMemcpy(std::data(got), d, bytes, cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(std::data(got2), d2, bytes, cudaMemcpyDeviceToHost));
	expect(all_equal(std::data(got2), n, 0x11111111u), name, bytes,
		   "the kernel read the bytes from before the copy");
	expect(all_equal(std::data(got), n, 0x22222222u), name, bytes, "the copy landed");
	std::free(h);
	CK(cudaFree(d));
	CK(cudaFree(d2));
}

void capture(std::size_t bytes, cudaStream_t s)
{
	const std::size_t n = bytes / 4;
	std::uint32_t* d = nullptr;
	CK(cudaMalloc(&d, bytes));
	std::uint32_t* h = pageable(bytes);
	for (std::size_t i = 0; i < n; i++)
	{
		h[i] = 0xAAAAAAAAu;
	}
	CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
	const cudaError_t e_copy = cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
	cudaGraph_t g = nullptr;
	const cudaError_t e_end = cudaStreamEndCapture(s, &g);
	(void)cudaGetLastError(); // a refused capture leaves a non-sticky error behind: clear it
	if (e_copy != cudaSuccess || e_end != cudaSuccess || !g)
	{
		std::printf("ok   %-11s %10zu  the driver refused it (%s / %s)\n", "capture", bytes,
					cudaGetErrorName(e_copy), cudaGetErrorName(e_end));
	}
	else
	{
		cudaGraphExec_t x = nullptr;
		CK(cudaGraphInstantiate(&x, g, 0));
		bool ok = true;
		for (std::uint32_t v : {0xBBBBBBBBu, 0xCCCCCCCCu})
		{ // change the host bytes, replay, check
			for (std::size_t i = 0; i < n; i++)
			{
				h[i] = v;
			}
			CK(cudaGraphLaunch(x, s));
			CK(cudaStreamSynchronize(s));
			std::vector<std::uint32_t> got(n);
			CK(cudaMemcpy(std::data(got), d, bytes, cudaMemcpyDeviceToHost));
			ok = ok && all_equal(std::data(got), n, v);
		}
		expect(ok, "capture", bytes, "each replay copies the host bytes as they are at replay");
		CK(cudaGraphExecDestroy(x));
		CK(cudaGraphDestroy(g));
	}
	std::free(h);
	CK(cudaFree(d));
}
#endif // HOOK_TEST_PTDS

// Generic copies (cudaMemcpyDefault, which ends in cuMemcpy / cuMemcpyAsync). Every case checks the bytes,
// stock and hooked. Hooked, the audit also says which way the copy went: pageable <-> this device's memory
// (cudaMalloc, VMM as vLLM's CuMemAllocator maps it, a cudaMallocAsync pool) must reach the hook, with
// CMT_TEST_STAGED=1 the chunks pipeline; managed, pinned and device-to-device copies must not reach it at
// all. With CMT_TEST_UNSTAGED_BYTES=n (a size the loaded table stages in neither direction), an n-byte copy
// must not reach it either: the size gate comes before any attribute query.
enum class Kind : std::uint8_t
{
	Pageable,
	Pinned,
	Device,
	Vmm,
	Pool,
	Managed
};

const char* KindName(Kind k)
{
	constexpr const char* names[] = {"pageable", "pinned", "device", "vmm", "pool", "managed"};
	return names[static_cast<unsigned>(k)];
}

#define CU(call)                                                                                             \
	do                                                                                                       \
	{                                                                                                        \
		const CUresult r_ = (call);                                                                          \
		if (r_ != CUDA_SUCCESS)                                                                              \
		{                                                                                                    \
			std::fprintf(stderr, "%s:%d: %s: %d\n", __FILE__, __LINE__, #call, static_cast<int>(r_));        \
			std::exit(1);                                                                                    \
		}                                                                                                    \
	} while (0)

struct Buffer
{
	Kind kind;
	std::size_t bytes;
	void* p = nullptr;
	CUmemGenericAllocationHandle handle = 0;
	std::size_t reserved = 0;
};

Buffer make(Kind kind, std::size_t bytes, cudaStream_t s)
{
	Buffer b{kind, bytes};
	switch (kind)
	{
	case Kind::Pageable:
		b.p = pageable(bytes);
		break;
	case Kind::Pinned:
		CK(cudaMallocHost(&b.p, bytes));
		break;
	case Kind::Device:
		CK(cudaMalloc(&b.p, bytes));
		break;
	case Kind::Vmm:
	{
		// One reservation per allocation, mapped whole: vLLM's CuMemAllocator (csrc/cumem_allocator.cpp).
		int device = 0;
		CK(cudaGetDevice(&device));
		CUmemAllocationProp prop = {};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = device;
		std::size_t granularity = 0;
		CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
		b.reserved = (bytes + granularity - 1) / granularity * granularity;
		CUdeviceptr va = 0;
		CU(cuMemAddressReserve(&va, b.reserved, 0, 0, 0));
		CU(cuMemCreate(&b.handle, b.reserved, &prop, 0));
		CU(cuMemMap(va, b.reserved, 0, b.handle, 0));
		CUmemAccessDesc access = {};
		access.location = prop.location;
		access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		CU(cuMemSetAccess(va, b.reserved, &access, 1));
		b.p = reinterpret_cast<void*>(static_cast<std::uintptr_t>(va));
		break;
	}
	case Kind::Pool:
		CK(cudaMallocAsync(&b.p, bytes, s));
		CK(cudaStreamSynchronize(s));
		break;
	case Kind::Managed:
		CK(cudaMallocManaged(&b.p, bytes));
		break;
	}
	return b;
}

void release(const Buffer& b, cudaStream_t s)
{
	switch (b.kind)
	{
	case Kind::Pageable:
		std::free(b.p);
		break;
	case Kind::Pinned:
		CK(cudaFreeHost(b.p));
		break;
	case Kind::Device:
	case Kind::Managed:
		CK(cudaFree(b.p));
		break;
	case Kind::Vmm:
	{
		const auto va = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(b.p));
		CU(cuMemUnmap(va, b.reserved));
		CU(cuMemRelease(b.handle));
		CU(cuMemAddressFree(va, b.reserved));
		break;
	}
	case Kind::Pool:
		CK(cudaFreeAsync(b.p, s));
		CK(cudaStreamSynchronize(s));
		break;
	}
}

bool host_side(Kind k)
{
	return k == Kind::Pageable || k == Kind::Pinned || k == Kind::Managed;
}

// Fill / read a buffer of any kind through typed, synchronous copies from a staging vector (the generic copy
// under test is the only cudaMemcpyDefault).
void fill(const Buffer& b, std::uint32_t seed)
{
	std::vector<std::uint32_t> v(b.bytes / 4);
	for (std::size_t i = 0; i < std::size(v); i++)
	{
		v[i] = seed * 2654435761u + static_cast<std::uint32_t>(i);
	}
	if (host_side(b.kind))
	{
		std::memcpy(b.p, std::data(v), b.bytes);
	}
	else
	{
		CK(cudaMemcpy(b.p, std::data(v), b.bytes, cudaMemcpyHostToDevice));
	}
}

bool holds(const Buffer& b, std::uint32_t seed)
{
	std::vector<std::uint32_t> v(b.bytes / 4);
	if (host_side(b.kind))
	{
		std::memcpy(std::data(v), b.p, b.bytes);
	}
	else
	{
		CK(cudaMemcpy(std::data(v), b.p, b.bytes, cudaMemcpyDeviceToHost));
	}
	for (std::size_t i = 0; i < std::size(v); i++)
	{
		if (v[i] != seed * 2654435761u + static_cast<std::uint32_t>(i))
		{
			return false;
		}
	}
	return true;
}

enum class Route : std::uint8_t
{
	Hook,	// entered the hook's routing; staged when CMT_TEST_STAGED=1
	Driver, // never entered the hook's routing
};

void generic(Kind to, Kind from, std::size_t bytes, Route route, cudaStream_t s, bool expect_staged)
{
	const bool hooked = HookAuditPresent();
	for (const bool async : {false, true})
	{
		const Buffer dst = make(to, bytes, s);
		const Buffer src = make(from, bytes, s);
		const std::uint32_t seed = static_cast<std::uint32_t>(bytes) + (async ? 11u : 5u);
		fill(src, seed);
		fill(dst, seed + 1);
		CK(cudaDeviceSynchronize());
		if (hooked)
		{
			HookAuditBegin();
		}
		CK(async ? cudaMemcpyAsync(dst.p, src.p, bytes, cudaMemcpyDefault, s)
				 : cudaMemcpy(dst.p, src.p, bytes, cudaMemcpyDefault));
		if (async && from == Kind::Pageable)
		{
			// A pageable source has been consumed when cudaMemcpyAsync returns (CUDA's API sync behaviour).
			std::memset(src.p, 0x11, bytes);
		}
		const HookAudit audit = hooked ? HookAuditEnd() : HookAudit{};
		CK(cudaStreamSynchronize(s));
		char what[96];
		std::snprintf(what, sizeof(what), "%s %s <- %s, bytes", async ? "async" : "sync", KindName(to),
					  KindName(from));
		expect(holds(dst, seed), "generic", bytes, what);
		if (hooked)
		{
			bool ok = audit.errors == 0;
			if (route == Route::Driver)
			{
				ok = ok && audit.calls == 0;
			}
			else
			{
				ok = ok && audit.calls == 1 && (!expect_staged || audit.staged == 1);
			}
			std::snprintf(what, sizeof(what), "%s %s <- %s: %s (calls %llu, staged %llu)",
						  async ? "async" : "sync", KindName(to), KindName(from),
						  route == Route::Driver ? "driver"
						  : expect_staged		 ? "staged"
												 : "hook",
						  static_cast<unsigned long long>(audit.calls),
						  static_cast<unsigned long long>(audit.staged));
			expect(ok, "generic", bytes, what);
		}
		release(src, s);
		release(dst, s);
	}
}

// A copy over the unmapped middle granule of a VMM reservation (mapped, hole, mapped). The driver refuses it
// (cudaErrorInvalidValue) before writing a byte; hooked, it must reach the driver the same way and the
// process must live: a staged chunk into the hole dies inside libcuda (SIGSEGV), which both a generic and a
// typed copy would without the hook's mapping check.
void hole(cudaMemcpyKind kind, bool to_device)
{
	int device = 0;
	CK(cudaGetDevice(&device));
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = device;
	std::size_t granule = 0;
	CU(cuMemGetAllocationGranularity(&granule, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
	const std::size_t bytes = 3 * granule;
	CUdeviceptr va = 0;
	CU(cuMemAddressReserve(&va, bytes, 0, 0, 0));
	CUmemGenericAllocationHandle handles[2] = {};
	CUmemAccessDesc access = {};
	access.location = prop.location;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	for (int i = 0; i < 2; i++)
	{
		CU(cuMemCreate(&handles[i], granule, &prop, 0));
		CU(cuMemMap(va + 2 * i * granule, granule, 0, handles[i], 0));
		CU(cuMemSetAccess(va + 2 * i * granule, granule, &access, 1));
	}
	void* d = reinterpret_cast<void*>(static_cast<std::uintptr_t>(va));
	CK(cudaMemset(d, 0, granule));
	std::vector<std::uint8_t> host(bytes, 0x5A);
	const bool hooked = HookAuditPresent();
	if (hooked)
	{
		HookAuditBegin();
	}
	const cudaError_t e = to_device ? cudaMemcpy(d, std::data(host), bytes, kind)
									: cudaMemcpy(std::data(host), d, bytes, kind);
	const HookAudit audit = hooked ? HookAuditEnd() : HookAudit{};
	(void)cudaGetLastError(); // the refusal is not sticky
	std::vector<std::uint8_t> first(granule, 0xFF);
	CK(cudaMemcpy(std::data(first), d, granule, cudaMemcpyDeviceToHost));
	bool untouched = true;
	for (std::size_t i = 0; i < granule && untouched; i++)
	{
		untouched = first[i] == 0 && (to_device || host[i] == 0x5A);
	}
	char what[96];
	std::snprintf(what, sizeof(what), "%s %s over a hole: refused (%s), nothing written%s",
				  kind == cudaMemcpyDefault ? "default" : "typed", to_device ? "h2d" : "d2h",
				  cudaGetErrorName(e),
				  hooked ? (audit.staged == 0 ? ", not staged" : ", STAGED") : "");
	expect(e != cudaSuccess && untouched && audit.staged == 0, "hole", bytes, what);
	for (int i = 0; i < 2; i++)
	{
		CU(cuMemUnmap(va + 2 * i * granule, granule));
		CU(cuMemRelease(handles[i]));
	}
	CU(cuMemAddressFree(va, bytes));
}

// redirected false: the per-thread-stream build, whose cudaMemcpyDefault ends in cuMemcpy_ptds /
// cuMemcpyAsync_ptsz, which the hook does not redirect: no copy may reach it.
void generic_copies(cudaStream_t s, bool redirected)
{
	const char* staged_env = std::getenv("CMT_TEST_STAGED");
	const bool staged = staged_env != nullptr && std::strcmp(staged_env, "1") == 0;
	const std::size_t bytes = std::size_t{64} << 20;
	const Route pageable_route = redirected ? Route::Hook : Route::Driver;
	for (const Kind device : {Kind::Device, Kind::Vmm, Kind::Pool})
	{
		generic(device, Kind::Pageable, bytes, pageable_route, s, staged);
		generic(Kind::Pageable, device, bytes, pageable_route, s, staged);
	}
	generic(Kind::Managed, Kind::Pageable, bytes, Route::Driver, s, staged);
	generic(Kind::Pageable, Kind::Managed, bytes, Route::Driver, s, staged);
	generic(Kind::Device, Kind::Pinned, bytes, Route::Driver, s, staged);
	generic(Kind::Pinned, Kind::Device, bytes, Route::Driver, s, staged);
	generic(Kind::Device, Kind::Device, bytes, Route::Driver, s, staged);
	if (const char* unstaged = std::getenv("CMT_TEST_UNSTAGED_BYTES"))
	{
		const std::size_t small = std::strtoull(unstaged, nullptr, 10) / 4 * 4;
		generic(Kind::Device, Kind::Pageable, small, Route::Driver, s, staged);
		generic(Kind::Pageable, Kind::Device, small, Route::Driver, s, staged);
	}
	hole(cudaMemcpyDefault, true);
	hole(cudaMemcpyDefault, false);
	hole(cudaMemcpyHostToDevice, true);
	hole(cudaMemcpyDeviceToHost, false);
}

#ifndef HOOK_TEST_PTDS
// A captured cudaMemcpyAsync with cudaMemcpyDefault (pageable -> device). CUDA records the generic call as a
// memcpy node of memory types 4/4 (unified); the typed call would record 1/2 (host/device), a different graph
// (measured on RTX 3090, driver 615.71.09). Hooked, the copy on a capturing stream goes to the driver and must be the call the
// application made. Each replay must copy the host bytes as they are at replay, as for the typed capture.
void capture_generic(std::size_t bytes, cudaStream_t s)
{
	const std::size_t n = bytes / 4;
	std::uint32_t* d = nullptr;
	CK(cudaMalloc(&d, bytes));
	std::uint32_t* h = pageable(bytes);
	CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
	const cudaError_t e_copy = cudaMemcpyAsync(d, h, bytes, cudaMemcpyDefault, s);
	cudaGraph_t g = nullptr;
	const cudaError_t e_end = cudaStreamEndCapture(s, &g);
	(void)cudaGetLastError();
	if (e_copy != cudaSuccess || e_end != cudaSuccess || !g)
	{
		std::printf("ok   %-11s %10zu  the driver refused it (%s / %s)\n", "capture_dflt", bytes,
					cudaGetErrorName(e_copy), cudaGetErrorName(e_end));
	}
	else
	{
		std::size_t count = 0;
		CK(cudaGraphGetNodes(g, nullptr, &count));
		std::vector<cudaGraphNode_t> nodes(count);
		CK(cudaGraphGetNodes(g, std::data(nodes), &count));
		bool unified = count == 1;
		for (cudaGraphNode_t node : nodes)
		{
			CUDA_MEMCPY3D p = {};
			unified = unified &&
					  cuGraphMemcpyNodeGetParams(reinterpret_cast<CUgraphNode>(node), &p) == CUDA_SUCCESS &&
					  p.srcMemoryType == CU_MEMORYTYPE_UNIFIED && p.dstMemoryType == CU_MEMORYTYPE_UNIFIED;
		}
		expect(unified, "capture_dflt", bytes,
			   "one memcpy node of memory types 4/4, as the generic call records");
		cudaGraphExec_t x = nullptr;
		CK(cudaGraphInstantiate(&x, g, 0));
		bool ok = true;
		for (std::uint32_t v : {0xBBBBBBBBu, 0xCCCCCCCCu})
		{
			for (std::size_t i = 0; i < n; i++)
			{
				h[i] = v;
			}
			CK(cudaGraphLaunch(x, s));
			CK(cudaStreamSynchronize(s));
			std::vector<std::uint32_t> got(n);
			CK(cudaMemcpy(std::data(got), d, bytes, cudaMemcpyDeviceToHost));
			ok = ok && all_equal(std::data(got), n, v);
		}
		expect(ok, "capture_dflt", bytes, "each replay copies the host bytes as they are at replay");
		CK(cudaGraphExecDestroy(x));
		CK(cudaGraphDestroy(g));
	}
	std::free(h);
	CK(cudaFree(d));
}

#endif // HOOK_TEST_PTDS

} // namespace

int main()
{
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
	{
		std::fprintf(stderr, "no CUDA device\n");
		return 77;
	}
	const std::size_t sizes[] = {std::size_t{64} << 10, std::size_t{4} << 20, std::size_t{64} << 20};
#ifdef HOOK_TEST_PTDS
	// two threads, each on its own per-thread default stream (stream 0 under --default-stream per-thread)
	for (std::size_t b : sizes)
	{
		std::thread t1([b] { order_dtoh(b, 0, "ptds", 0x5A5A0001u); });
		std::thread t2([b] { order_dtoh(b, 0, "ptds", 0x5A5A0002u); });
		t1.join();
		t2.join();
	}
	cudaStream_t s = nullptr;
	CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
	generic_copies(s, false);
	CK(cudaStreamDestroy(s));
#else
	cudaStream_t s = nullptr;
	CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
	for (std::size_t b : sizes)
	{
		order_dtoh(b, s, "order_dtoh", 0x5A5A5A5Au);
		order_htod(b, s);
		// The same with cudaMemcpyDefault: a generic async copy must stay on the application's stream.
		order_dtoh(b, s, "order_dtoh_g", 0x5A5A5A5Bu, cudaMemcpyDefault);
		order_htod(b, s, "order_htod_g", cudaMemcpyDefault);
		capture(b, s);
	}
	generic_copies(s, true);
	capture_generic(std::size_t{64} << 20, s);
	CK(cudaStreamDestroy(s));
#endif
	std::printf("%s\n", g_failed ? "FAILED" : "all checks hold");
	return g_failed ? 1 : 0;
}
