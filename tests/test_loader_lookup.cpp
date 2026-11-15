#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <initializer_list>
#include <sys/wait.h>
#include <unistd.h>

#ifdef MH_LOADER_LINKED
extern "C" int mh_loader_run(const char*);
#endif

namespace {
using Query = int (*)(const char*, void**, int, unsigned long long);
using QueryV2 = int (*)(const char*, void**, int, unsigned long long, int*);

bool Check(bool condition, const char* description)
{
	if (!condition)
	{
		std::fprintf(stderr, "CUDA loader: %s\n", description);
	}
	return condition;
}

// Run this mode with the optimizing hook and fake libcuda. No GPU is used.
[[maybe_unused]] int CudaLookups(void* handle)
{
#if defined(__x86_64__)
	constexpr const char* kDlsymVersion = "GLIBC_2.2.5";
#elif defined(__aarch64__)
	constexpr const char* kDlsymVersion = "GLIBC_2.17";
#endif
	const auto original_lookup =
		reinterpret_cast<void* (*)(void*, const char*)>(dlvsym(RTLD_NEXT, "dlsym", kDlsymVersion));
	if (original_lookup == nullptr)
	{
		return 1;
	}
	const auto query = reinterpret_cast<Query>(dlsym(handle, "cuGetProcAddress"));
	const auto query_v2 = reinterpret_cast<QueryV2>(dlsym(handle, "cuGetProcAddress_v2"));
	const auto original_query = reinterpret_cast<Query>(original_lookup(handle, "cuGetProcAddress"));
	const auto original_query_v2 = reinterpret_cast<QueryV2>(original_lookup(handle, "cuGetProcAddress_v2"));
	if (!Check(query != nullptr && query_v2 != nullptr && original_query != nullptr &&
				   original_query_v2 != nullptr && query != original_query && query_v2 != original_query_v2,
			   "both query entry points must be intercepted"))
	{
		return 1;
	}
	bool ok = true;
	// Exhaust each bank (8 slots) in its own child. Without progressive-pinning registrations the hook then
	// stops staging for good and hands out the driver's own function for the ninth address, instead of
	// terminating the application (hook.h, DisableCopiesForUnwrappedTeardown).
	for (const char* name : {"cuDevicePrimaryCtxReset", "cuCtxDestroy"})
	{
		const pid_t child = fork();
		if (child == 0)
		{
			for (int version = 100; version <= 108; ++version)
			{
				void* entry = nullptr;
				void* raw = nullptr;
				if (query(name, &entry, version, 0) != 0 || original_query(name, &raw, version, 0) != 0)
				{
					_exit(2);
				}
				if ((version < 108) == (entry == raw)) // wrapped while slots last, the original after
				{
					_exit(3);
				}
			}
			_exit(0);
		}
		int status = 0;
		ok &= Check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
						WEXITSTATUS(status) == 0,
					"wrapper exhaustion must hand out the original and keep the application running");
	}

	struct Rejected
	{
		const char* symbol;
		int version;
		unsigned long long flags;
	};

	const Rejected rejected[] = {
		{"cuMemcpyHtoD", 99999, 0},
		{"cuMemcpyHtoD", 12000, 8},
		{"cuCtxDestroy_v2", 12000, 0}, // driver queries use the base name, not the export name
		{nullptr, 12000, 0},
		{"cuMemcpyHtoD_v2", 3010, 0}, // v2: rc succeeds, query status rejects the ABI
	};
	for (const Rejected& request : rejected)
	{
		void* expected = reinterpret_cast<void*>(1);
		void* actual = expected;
		const int expected_rc = original_query(request.symbol, &expected, request.version, request.flags);
		const int actual_rc = query(request.symbol, &actual, request.version, request.flags);
		ok &= Check(actual_rc == expected_rc && actual == expected, "v1 must preserve a rejected query");
		expected = reinterpret_cast<void*>(1);
		actual = expected;
		int expected_status = -7;
		int actual_status = expected_status;
		const int expected_v2 =
			original_query_v2(request.symbol, &expected, request.version, request.flags, &expected_status);
		const int actual_v2 =
			query_v2(request.symbol, &actual, request.version, request.flags, &actual_status);
		ok &= Check(actual_v2 == expected_v2 && actual == expected && actual_status == expected_status,
					"v2 must preserve return code, pointer, and query status");
	}
	ok &= Check(query("cuMemcpyHtoD", nullptr, 12000, 0) == original_query("cuMemcpyHtoD", nullptr, 12000, 0),
				"null output must reach the driver");
	void* entry = nullptr;
	void* original = nullptr;
	query("cuMemcpyHtoD", &entry, 2000, 0);
	original_query("cuMemcpyHtoD", &original, 2000, 0);
	ok &= Check(entry == original && dlsym(handle, "cuMemcpyHtoD") == original,
				"old copy ABI and old direct export must remain unchanged");
	query("cuMemcpyHtoD", &entry, 12000, 0);
	original_query("cuMemcpyHtoD", &original, 12000, 0);
	ok &= Check(entry != original && entry == dlsym(handle, "cuMemcpyHtoD_v2"),
				"modern copy query must use the compatible wrapper");
	query("cuMemcpyHtoD", &entry, 12000, 2);
	original_query("cuMemcpyHtoD", &original, 12000, 2);
	ok &= Check(entry == original, "per-thread default stream must remain unchanged");
	int status = -1;
	query_v2("cuGetProcAddress", &entry, 11030, 0, &status);
	ok &= Check(status == 0 && entry == reinterpret_cast<void*>(query), "v2 query can request the v1 ABI");
	query("cuGetProcAddress", &entry, 12000, 0);
	ok &= Check(entry == reinterpret_cast<void*>(query_v2), "v1 query can request the v2 ABI");
	query("cuCtxDestroy", &entry, 100, 0);
	original_query("cuCtxDestroy", &original, 100, 0);
	ok &= Check(entry != original &&
					reinterpret_cast<int (*)(void*)>(entry)(reinterpret_cast<void*>(0x1000)) == 77,
				"unexported context teardown must call its exact original");
	return ok ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		return 2;
	}
#ifdef MH_LOADER_LINKED
	return mh_loader_run(argv[1]);
#else
	if (argc != 3)
	{
		return 2;
	}
	// The caller and its exports stay local. RTLD_DEFAULT inside it must include
	// its local lookup scope; resolving from the hook instead loses that scope.
	void* handle = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
	if (handle == nullptr)
	{
		std::fprintf(stderr, "caller dlopen: %s\n", dlerror());
		return 3;
	}
	if (std::strcmp(argv[1], "cuda") == 0)
	{
		return CudaLookups(handle);
	}
	void* symbol = dlsym(handle, "mh_loader_run");
	if (symbol == nullptr)
	{
		return 4;
	}
	int (*run)(const char*) = nullptr;
	// SAFETY: loader_caller.cpp exports exactly this signature.
	static_assert(sizeof(run) == sizeof(symbol));
	std::memcpy(&run, &symbol, sizeof(run));
	const int result = run(argv[1]);
	(void)dlclose(handle);
	return result;
#endif
}
