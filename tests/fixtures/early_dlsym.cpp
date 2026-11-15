// early_dlsym.cpp - a library preloaded AFTER the hook (LD_PRELOAD=<hook>:<this>), whose constructor
// therefore runs BEFORE the hook's constructors and static initializers (the dynamic loader initializes
// preloaded objects in reverse order). It looks up a redirected libcuda entry point the way an early library
// would, through the hook's native dlopen and intercepted explicit-handle dlsym, and keeps the answer for
// test_ring_lifecycle early_dlsym.
#include <dlfcn.h>
#include <cstdlib>

extern "C"
{
	__attribute__((visibility("default"))) void* early_dlsym_htod = nullptr;
	__attribute__((visibility("default"))) int early_dlsym_ran = 0;
	// 1 when this constructor ran before the hook's (libcuda was not loaded yet; see early()).
	__attribute__((visibility("default"))) int early_dlsym_before_hook = 0;
}

namespace {

__attribute__((constructor)) void early()
{
	// The hook's constructor sets nothing we can read, but it prints the knob table with MEMCPY_HOOK_DEBUG=1
	// and dlopens libcuda eagerly: before it, libcuda.so.1 is not loaded yet (RTLD_NOLOAD finds nothing).
	early_dlsym_before_hook = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD) == nullptr ? 1 : 0;
	void* h = dlopen("libcuda.so.1", RTLD_NOW);
	if (h != nullptr)
	{
		early_dlsym_htod = dlsym(h, "cuMemcpyHtoD_v2");
	}
	early_dlsym_ran = 1;
}

} // namespace
