#include "hook.h"

#include "copying/host_alloc.h"

namespace cmt {
namespace {

using DlsymFn = void* (*)(void*, const char*);
using DlopenFn = void* (*)(const char*, int);

std::atomic<DlsymFn> g_real_dlsym{nullptr};
std::atomic<DlopenFn> g_real_dlopen{nullptr};
std::atomic<void*> g_pending_handle{nullptr};
std::atomic<void*> g_libcuda_handle{nullptr};
std::atomic<bool> g_install_failed{false};
std::atomic<bool> g_hook_disabled{false};
uintptr_t g_libcuda_base = 0;
char g_libcuda_path[4096] = {0};
std::atomic<uintptr_t> g_libcuda_lo{0};
std::atomic<uintptr_t> g_libcuda_hi{0};

void EnsureOriginals()
{
	if (g_real_dlsym.load(std::memory_order_acquire) != nullptr)
	{
		return;
	}
	void* p = dlvsym(RTLD_NEXT, "dlsym", kRequiredGlibcSymbolVersion);
	const auto dlsym_fn = reinterpret_cast<DlsymFn>(p);
	p = dlvsym(RTLD_NEXT, "dlopen", kRequiredGlibcSymbolVersion);
	g_real_dlopen.store(reinterpret_cast<DlopenFn>(p), std::memory_order_relaxed);
	g_real_dlsym.store(dlsym_fn, std::memory_order_release);
}

} // namespace

LoaderFunctions GetLoaderFunctions()
{
	EnsureOriginals();
	return {g_real_dlsym.load(std::memory_order_acquire), g_real_dlopen.load(std::memory_order_acquire)};
}

namespace {

void VersionFromPath(const char* path, char* out, std::size_t out_len)
{
	out[0] = '\0';
	const char* base_name = std::strrchr(path, '/');
	base_name = base_name ? base_name + 1 : path;
	if (std::strncmp(base_name, "libcuda.so.", 11) == 0)
	{
		std::snprintf(out, out_len, "%s", base_name + 11);
	}
}

bool LooksLikeLibcuda(const char* path)
{
	if (path == nullptr)
	{
		return false;
	}
	const char* base = std::strrchr(path, '/');
	base = base ? base + 1 : path;
	return std::strncmp(base, "libcuda.so", 10) == 0;
}

bool RetainDriverImage(void* handle)
{
	link_map* image = nullptr;
	if (dlinfo(handle, RTLD_DI_LINKMAP, &image) != 0 || image == nullptr || image->l_name == nullptr ||
		image->l_name[0] == '\0')
	{
		return false;
	}
	const auto open = g_real_dlopen.load(std::memory_order_acquire);
	void* retained =
		open != nullptr ? open(image->l_name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD | RTLD_NODELETE) : nullptr;
	if (retained == handle)
	{
		return true;
	}
	if (retained != nullptr)
	{
		dlclose(retained);
	}
	return false;
}

void InstallForHandle(void* handle, bool eager) noexcept
{
	if (!RetainDriverImage(handle))
	{
		LogLine("note: cannot retain the exact libcuda image; leaving its entry points unchanged");
		return;
	}
	void* installed = nullptr;
	if (!g_pending_handle.compare_exchange_strong(installed, handle, std::memory_order_acq_rel))
	{
		dlclose(handle);
		if (handle != installed)
		{
			LogDebug("NOTE: second libcuda handle {} (first {}) - patches only on the first image",
					 fmt::ptr(handle), fmt::ptr(installed));
		}
		return;
	}

	if (trace::BypassTuning())
	{
		g_libcuda_handle.store(handle, std::memory_order_release);
		return;
	}
	void* cu_init = g_real_dlsym.load(std::memory_order_acquire)(handle, "cuInit");
	Dl_info info;
	if (cu_init == nullptr || dladdr(cu_init, &info) == 0)
	{
		LogDebug("cannot resolve cuInit/dladdr; the hook passes every call to the driver");
		cu::ResolveDriverApi(CudaSymbol, GetConfig().debug);
		g_install_failed.store(true, std::memory_order_release);
		return;
	}
	g_libcuda_base = reinterpret_cast<uintptr_t>(info.dli_fbase);
	const auto extent = cu::FindImageExtent(reinterpret_cast<uintptr_t>(cu_init));
	if (extent.lo != extent.hi)
	{
		g_libcuda_lo.store(extent.lo, std::memory_order_relaxed);
		g_libcuda_hi.store(extent.hi, std::memory_order_release);
	}
	std::snprintf(g_libcuda_path, sizeof(g_libcuda_path), "%s", info.dli_fname ? info.dli_fname : "?");

	char real_path[PATH_MAX] = {0};
	const char* path_for_version = g_libcuda_path;
	if (realpath(g_libcuda_path, real_path) != nullptr)
	{
		path_for_version = real_path;
	}
	char version[64];
	VersionFromPath(path_for_version, version, sizeof(version));
	const char* row_note = "";
	const patch::DriverVersion* vi =
		patch::SelectRow(patch::NativeVersionTable(), version, GetConfig().allow_unverified_rows, &row_note);
	LogDebug("libcuda {} at base {:x} (version '{}', {} db row: {}{}{}, pid {})", g_libcuda_path,
			 static_cast<unsigned long>(g_libcuda_base), version, ArchName(kNativeArch),
			 vi ? vi->version : "none", *row_note != '\0' ? "; " : "", row_note, static_cast<int>(getpid()));
	if (vi == nullptr && patch::FindVersion(patch::NativeVersionTable(), version) != nullptr &&
		(GetConfig().WantH2HPatch() || GetConfig().WantStagingSizeWrap()))
	{
		LogLine("note: libcuda {} has an {} table row that is unverified on hardware; "
				"in-driver patches skipped (set MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS=1 to use it)",
				version, ArchName(kNativeArch));
	}

	cu::ResolveDriverApi(CudaSymbol, GetConfig().debug);
	const HookConfig& cfg = GetConfig();
	if (eager)
	{
		driver::InstallPatches(vi, g_libcuda_base);
	}
	else if (cfg.WantH2HPatch() || cfg.WantInternalMemcpyPatch() || cfg.WantStagingSizeWrap())
	{
		LogLine("note: libcuda {} was loaded after the hook's constructor; requested in-driver patches are "
				"skipped (patching code that running threads may execute is unsafe). API copy interposition is "
				"unaffected.",
				g_libcuda_path);
	}
	g_libcuda_handle.store(handle,
						   std::memory_order_release);
}

void* RedirectLookup(const char* symbol, bool per_thread_stream);
void* RedirectForAddress(void* real);

using CuGetProcAddressFn = int (*)(const char*, void**, int, unsigned long long);
using CuGetProcAddressV2Fn = int (*)(const char*, void**, int, unsigned long long, int*);
constexpr int kSizeTDriverAbiVersion = 3020;
constexpr int kProcAddressFound = 0;

bool IsPrimaryTeardown(const char* symbol)
{
	return std::strcmp(symbol, "cuDevicePrimaryCtxReset") == 0 ||
		   std::strcmp(symbol, "cuDevicePrimaryCtxReset_v2") == 0 ||
		   std::strcmp(symbol, "cuDevicePrimaryCtxRelease") == 0 ||
		   std::strcmp(symbol, "cuDevicePrimaryCtxRelease_v2") == 0;
}

void* WrapUnmatchedPrimary(const char* symbol, void* real, int cuda_version)
{
	if (!IsPrimaryTeardown(symbol))
	{
		return nullptr;
	}
	void* wrapped = ring::WrapPrimaryByName(real);
	if (wrapped == nullptr)
	{
		if (!ring::DisableCopiesForUnwrappedTeardown("primary-context"))
		{
			Panic("primary-context wrapper capacity exhausted; cannot preserve context teardown");
		}
		return nullptr;
	}
	static std::atomic<bool> said{false};
	if (!said.exchange(true, std::memory_order_relaxed))
	{
		LogLine("error: cuGetProcAddress(\"{}\", cudaVersion {}) returned an unexported address {}; "
				"the hook binds that address and frees its context resources first. Reported once.",
				symbol, cuda_version, fmt::ptr(real));
	}
	return wrapped;
}

void* RedirectQuery(const char* symbol, void* real, int cuda_version, unsigned long long flags)
{
	if (symbol == nullptr || real == nullptr || trace::BypassTuning() ||
		g_hook_disabled.load(std::memory_order_relaxed))
	{
		return nullptr;
	}
	if (void* wrapped = RedirectForAddress(real))
	{
		return wrapped;
	}
	if (void* wrapped = WrapUnmatchedPrimary(symbol, real, cuda_version))
	{
		return wrapped;
	}
	if (std::strcmp(symbol, "cuGetProcAddress") == 0 || std::strcmp(symbol, "cuGetProcAddress_v2") == 0)
	{
		for (const char* name : {"cuGetProcAddress", "cuGetProcAddress_v2"})
		{
			if (real == CudaSymbol(name))
			{
				return RedirectLookup(name, false);
			}
		}
		return nullptr;
	}
	if (std::strcmp(symbol, "cuCtxDestroy") == 0 || std::strcmp(symbol, "cuCtxDestroy_v2") == 0)
	{
		for (const char* name : {"cuCtxDestroy", "cuCtxDestroy_v2"})
		{
			if (real == CudaSymbol(name))
			{
				return RedirectLookup(name, false);
			}
		}
		return ring::WrapContextByName(real);
	}
	if (void* wrapped = host_alloc::RedirectQuery(symbol, real))
	{
		return wrapped;
	}
	if (cuda_version < kSizeTDriverAbiVersion || (flags & cu::kGetProcAddressPerThreadDefaultStream) != 0)
	{
		return nullptr;
	}

	struct CopyEntry
	{
		const char* query_name;
		const char* export_name;
	};

	constexpr CopyEntry copies[] = {
		{"cuMemcpyHtoD", "cuMemcpyHtoD_v2"},
		{"cuMemcpyDtoH", "cuMemcpyDtoH_v2"},
		{"cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"},
		{"cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"},
		{"cuMemcpy", "cuMemcpy"},
		{"cuMemcpyAsync", "cuMemcpyAsync"},
	};
	for (const CopyEntry& entry : copies)
	{
		if ((std::strcmp(symbol, entry.query_name) == 0 || std::strcmp(symbol, entry.export_name) == 0) &&
			real == CudaSymbol(entry.export_name))
		{
			return RedirectLookup(entry.export_name, false);
		}
	}
	return nullptr;
}

int GetProcAddressEntry(const char* symbol, void** pfn, int cuda_version, unsigned long long flags)
{
	static std::atomic<CuGetProcAddressFn> real_fn{nullptr};
	if (real_fn.load(std::memory_order_acquire) == nullptr)
	{
		real_fn.store(reinterpret_cast<CuGetProcAddressFn>(CudaSymbol("cuGetProcAddress")),
					  std::memory_order_release);
	}
	const auto real = real_fn.load(std::memory_order_acquire);
	if (real == nullptr)
	{
		return cu::kErrorNotFound;
	}
	const int rc = real(symbol, pfn, cuda_version, flags);
	if (rc == cu::kSuccess && pfn != nullptr && *pfn != nullptr)
	{
		if (void* wrapped = RedirectQuery(symbol, *pfn, cuda_version, flags))
		{
			*pfn = wrapped;
		}
	}
	return rc;
}

int GetProcAddressV2Entry(const char* symbol, void** pfn, int cuda_version, unsigned long long flags,
						  int* symbol_status)
{
	static std::atomic<CuGetProcAddressV2Fn> real_fn{nullptr};
	if (real_fn.load(std::memory_order_acquire) == nullptr)
	{
		real_fn.store(reinterpret_cast<CuGetProcAddressV2Fn>(CudaSymbol("cuGetProcAddress_v2")),
					  std::memory_order_release);
	}
	const auto real = real_fn.load(std::memory_order_acquire);
	if (real == nullptr)
	{
		return cu::kErrorNotFound;
	}
	const int rc = real(symbol, pfn, cuda_version, flags, symbol_status);
	if (rc == cu::kSuccess && pfn != nullptr && *pfn != nullptr &&
		(symbol_status == nullptr || *symbol_status == kProcAddressFound))
	{
		if (void* wrapped = RedirectQuery(symbol, *pfn, cuda_version, flags))
		{
			*pfn = wrapped;
		}
	}
	return rc;
}

void* RedirectForAddress(void* real)
{
	if (trace::BypassTuning() || real == nullptr)
	{
		return nullptr;
	}
	return ring::RedirectForAddress(real);
}

void OnForkChild()
{
	if (!ring::UsedInProcess() && !pinning::Active())
	{
		return;
	}
	ring::AfterForkInChild();
	pinning::AfterForkInChild();
}

void OnForkChildAllocations()
{
	host_alloc::AfterForkInChild();
}

void* RedirectLookup(const char* symbol, bool per_thread_stream)
{
	if (trace::BypassTuning() || symbol == nullptr || g_hook_disabled.load(std::memory_order_relaxed))
	{
		return nullptr;
	}
	if (std::strcmp(symbol, "cuGetProcAddress") == 0)
	{
		return reinterpret_cast<void*>(&GetProcAddressEntry);
	}
	if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0)
	{
		return reinterpret_cast<void*>(&GetProcAddressV2Entry);
	}
	if (void* wrapped = host_alloc::RedirectLookup(symbol))
	{
		return wrapped;
	}
	return ring::RedirectLookup(symbol, per_thread_stream);
}

__attribute__((constructor)) void HookConstructor() noexcept
{
	cmt::InstallFatalAllocationHandler();
	trace::Start();
	if (trace::BypassTuning())
	{
		EnsureOriginals();
		pinning::PrepareInterposition();
		return;
	}
	const HookConfig& cfg = GetConfig();
	if (pthread_atfork(nullptr, nullptr, &OnForkChild) != 0 ||
		(cfg.cuda_host_alloc_hook && pthread_atfork(nullptr, nullptr, &OnForkChildAllocations) != 0))
	{
		g_hook_disabled.store(true, std::memory_order_relaxed);
		LogLine("error: cannot install the hook's fork handler; the hook is disabled and CUDA calls reach the "
				"driver unchanged");
		return;
	}
	if (cfg.debug)
	{
		PrintConfig(cfg, STDERR_FILENO);
		if (cfg.tune_table != nullptr)
		{
			tuning::Print(*cfg.tune_table, cfg.tune_table_path, STDERR_FILENO);
		}
	}
	EnsureOriginals();
	if (g_real_dlsym.load(std::memory_order_acquire) == nullptr ||
		g_real_dlopen.load(std::memory_order_acquire) == nullptr)
	{
		LogDebug("cannot resolve real dlsym/dlopen; hook disabled");
		return;
	}
	pinning::PrepareInterposition();
	if (!cfg.no_eager)
	{
		const auto open = g_real_dlopen.load(std::memory_order_acquire);
		void* h = open("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE | RTLD_NOLOAD);
		if (h == nullptr && (cfg.WantH2HPatch() || cfg.WantInternalMemcpyPatch() || cfg.WantStagingSizeWrap()))
		{
			h = open("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
		}
		if (h != nullptr)
		{
			InstallForHandle(h, true);
		}
		else
		{
			LogDebug("libcuda.so.1 not loaded by the constructor; waiting for the app");
		}
	}
}

__attribute__((destructor)) void HookDestructor()
{
	if (trace::BypassTuning())
	{
		trace::Stop();
		return;
	}
	pinning::Shutdown();
}

} // namespace

void* CudaSymbol(const char* name)
{
	void* handle = g_pending_handle.load(std::memory_order_acquire);
	const auto lookup = g_real_dlsym.load(std::memory_order_acquire);
	if (handle == nullptr || lookup == nullptr)
	{
		return nullptr;
	}
	return lookup(handle, name);
}

bool DriverApiReady()
{
	return g_libcuda_handle.load(std::memory_order_acquire) != nullptr;
}

bool DriverApiResolved()
{
	return DriverApiReady() || g_install_failed.load(std::memory_order_acquire);
}

bool IsCudaDriverAddress(uintptr_t address) noexcept
{
	return address >= g_libcuda_lo.load(std::memory_order_relaxed) &&
		   address < g_libcuda_hi.load(std::memory_order_acquire);
}

} // namespace cmt

namespace {

void* LookupExplicitHandle(void* handle, const char* symbol) noexcept
{
	cmt::EnsureOriginals();
	if (cmt::g_pending_handle.load(std::memory_order_acquire) == nullptr)
	{
		link_map* image = nullptr;
		if (!cmt::g_hook_disabled.load(std::memory_order_relaxed) && dlinfo(handle, RTLD_DI_LINKMAP, &image) == 0 &&
			image != nullptr && cmt::LooksLikeLibcuda(image->l_name))
		{
			cmt::InstallForHandle(handle, false);
		}
	}
	void* fn = cmt::g_real_dlsym.load(std::memory_order_acquire)(handle, symbol);
	if (cmt::trace::BypassTuning())
	{
		if (void* wrapped = cmt::trace::Redirect(symbol, fn))
		{
			return wrapped;
		}
		return fn;
	}
	void* cuda = cmt::g_pending_handle.load(std::memory_order_acquire);
	if (fn != nullptr && cuda != nullptr && handle == cuda)
	{
		void* redir = cmt::RedirectLookup(symbol, false);
		if (redir == nullptr)
		{
			redir = cmt::RedirectForAddress(fn);
		}
		if (redir != nullptr)
		{
			return redir;
		}
	}
	return fn;
}

} // namespace

extern "C" __attribute__((visibility("hidden"))) cmt::DlsymFn cmt_dlsym_target(void* handle) noexcept
{
	cmt::EnsureOriginals();
	return handle == RTLD_NEXT || handle == RTLD_DEFAULT ? cmt::g_real_dlsym.load(std::memory_order_acquire)
														 : &LookupExplicitHandle;
}

// RTLD_NEXT/DEFAULT depend on the caller address, so tail-forward with the original return address.
#if defined(__x86_64__)
asm(".text\n"
	".globl dlsym\n"
	".type dlsym,@function\n"
	".hidden cmt_dlsym_target\n"
	"dlsym:\n"
	".cfi_startproc\n"
	"endbr64\n"
	"subq $24, %rsp\n"
	".cfi_def_cfa_offset 32\n"
	"movq %rdi, 8(%rsp)\n"
	"movq %rsi, 16(%rsp)\n"
	"call cmt_dlsym_target\n"
	"movq 8(%rsp), %rdi\n"
	"movq 16(%rsp), %rsi\n"
	"addq $24, %rsp\n"
	".cfi_def_cfa_offset 8\n"
	"jmp *%rax\n"
	".cfi_endproc\n"
	".size dlsym, .-dlsym\n");
#elif defined(__aarch64__)
asm(".text\n"
	".globl dlsym\n"
	".type dlsym,%function\n"
	".hidden cmt_dlsym_target\n"
	"dlsym:\n"
	".cfi_startproc\n"
	".inst 0xd503245f\n"
	"stp x29, x30, [sp, #-32]!\n"
	".cfi_def_cfa_offset 32\n"
	".cfi_offset 29, -32\n"
	".cfi_offset 30, -24\n"
	"stp x0, x1, [sp, #16]\n"
	"bl cmt_dlsym_target\n"
	"mov x16, x0\n"
	"ldp x0, x1, [sp, #16]\n"
	"ldp x29, x30, [sp], #32\n"
	".cfi_restore 29\n"
	".cfi_restore 30\n"
	".cfi_def_cfa_offset 0\n"
	"br x16\n"
	".cfi_endproc\n"
	".size dlsym, .-dlsym\n");
#else
#error "The loader entry stub needs an audited ABI implementation for this architecture"
#endif
