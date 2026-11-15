// Caller-sensitive lookup fixtures. This DSO owns RUNPATH=$ORIGIN and exports
// its own probe, distinct from the provider that should follow it for RTLD_NEXT.
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

extern "C" __attribute__((visibility("default"))) int mh_loader_probe() { return 42; }

extern "C" __attribute__((visibility("default"))) int mh_loader_run(const char* mode)
{
	void* handle = RTLD_DEFAULT;
	int expected = 42;
	if (std::strcmp(mode, "next") == 0)
	{
		handle = RTLD_NEXT;
		expected = 7;
	}
	else if (std::strcmp(mode, "sibling") == 0)
	{
		handle = dlopen("libmh_loader_provider.so", RTLD_NOW | RTLD_LOCAL);
		if (handle == nullptr)
		{
			std::fprintf(stderr, "sibling lookup failed: %s\n", dlerror());
			return 1;
		}
		expected = 7;
	}
	(void)dlerror();
	void* symbol = dlsym(handle, "mh_loader_probe");
	const char* error = dlerror();
	if (symbol == nullptr || error != nullptr)
	{
		std::fprintf(stderr, "%s lookup failed: %s\n", mode, error ? error : "null symbol");
		return 2;
	}
	int (*probe)() = nullptr;
	// SAFETY: this fixture controls both providers; each exports this exact ABI.
	static_assert(sizeof(probe) == sizeof(symbol));
	std::memcpy(&probe, &symbol, sizeof(probe));
	const int actual = probe();
	if (std::strcmp(mode, "sibling") == 0)
	{
		(void)dlclose(handle);
	}
	if (actual != expected)
	{
		std::fprintf(stderr, "%s: expected provider %d, got %d\n", mode, expected, actual);
		return 3;
	}
	// A failed lookup must retain normal dlerror reporting, including after a
	// successful caller-sensitive lookup through our entry stub.
	(void)dlerror();
	symbol = dlsym(RTLD_DEFAULT, "mh_loader_symbol_that_does_not_exist");
	if (symbol != nullptr || dlerror() == nullptr)
	{
		return 4;
	}
	return 0;
}
