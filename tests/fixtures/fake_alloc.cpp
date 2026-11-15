// fake_alloc.cpp - a malloc replacement preloaded AFTER the hook (LD_PRELOAD=<hook>:<this>), standing in for
// tcmalloc / jemalloc / mimalloc in that position. malloc() then resolves here, while free() and
// realloc() resolve to the hook, which comes first in the search order. The hook must forward them to this
// library.
//
// Every block carries a header just below the pointer it returns (a magic word and the real base), so the
// pointers are foreign to glibc: glibc's free() of one aborts ("free(): invalid pointer"), and this library's
// free() of a pointer it did not make aborts too, with a message. Counters and a per-thread failure switch
// are exported for test_ring_lifecycle:
//   fake_alloc_frees()            frees that reached this library
//   fake_alloc_fail_next(n)       the next n allocations made by the calling thread fail (ENOMEM), so
//   operator new
//                                 throws std::bad_alloc there (out of memory inside the hook's teardown)
// A block of 64 KiB or more is reported to the fake libcuda as replaced pages (fake_pages_changed) when it is
// freed, before it goes back to glibc: from then on its pages may be other physical pages, and a DMA through
// a registration made before that reads the old ones (fake_libcuda.cpp, "stale registrations").
#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" void* __libc_memalign(std::size_t, std::size_t);
extern "C" void __libc_free(void*);

#define FAKE_ALLOC_API extern "C" __attribute__((visibility("default")))

namespace {

constexpr uint64_t kMagic = 0xFA4EA110CA7EDull;

struct Header
{
	uint64_t magic;
	void* base;
	std::size_t size;
	uint64_t pad;
};

static_assert(sizeof(Header) == 32, "header size");

std::atomic<long> g_frees{0};
thread_local int t_fail_next = 0;

// fake_pages_changed of the fake libcuda, once it is loaded (it is dlopen'ed RTLD_LOCAL: look it up by
// handle).
thread_local bool t_resolving = false;
std::atomic<void (*)(const void*, std::size_t)> g_report{nullptr};

void report_replaced(void* p, std::size_t n)
{
	void (*fn)(const void*, std::size_t) = g_report.load(std::memory_order_acquire);
	if (fn == nullptr && !t_resolving)
	{
		t_resolving = true;
		if (void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD))
		{
			if (void* sym = dlsym(lib, "fake_pages_changed"))
			{
				std::memcpy(&fn, &sym, sizeof(fn));
				g_report.store(fn, std::memory_order_release);
			}
			dlclose(lib);
		}
		t_resolving = false;
	}
	if (fn != nullptr)
	{
		const uintptr_t pg = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
		const uintptr_t a = (reinterpret_cast<uintptr_t>(p) + pg - 1) & ~(pg - 1);
		const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + n) & ~(pg - 1);
		if (b > a)
		{
			fn(reinterpret_cast<void*>(a), b - a);
		}
	}
}

[[noreturn]] void die(const char* what)
{
	(void)!write(2, what, std::strlen(what));
	_exit(86);
}

Header* header_of(void* p) { return reinterpret_cast<Header*>(static_cast<uint8_t*>(p) - sizeof(Header)); }

void* alloc(std::size_t align, std::size_t n)
{
	if (t_fail_next > 0)
	{
		--t_fail_next;
		errno = ENOMEM;
		return nullptr;
	}
	if (align < sizeof(Header))
	{
		align = sizeof(Header);
	}
	// The header must fit below the returned pointer, which keeps `align`: reserve one whole alignment unit.
	void* base = __libc_memalign(align, align + (n == 0 ? 1 : n));
	if (base == nullptr)
	{
		errno = ENOMEM;
		return nullptr;
	}
	void* user = static_cast<uint8_t*>(base) + align;
	*header_of(user) = Header{kMagic, base, n, 0};
	return user;
}

} // namespace

FAKE_ALLOC_API void* malloc(std::size_t n) { return alloc(16, n); }

FAKE_ALLOC_API void free(void* p)
{
	if (p == nullptr)
	{
		return;
	}
	Header* h = header_of(p);
	if (h->magic != kMagic)
	{
		die("fake_alloc: free() of a pointer this allocator did not make\n");
	}
	h->magic = 0;
	g_frees.fetch_add(1, std::memory_order_relaxed);
	if (h->size >= (64u << 10))
	{
		report_replaced(p, h->size);
	}
	__libc_free(h->base);
}

FAKE_ALLOC_API void* calloc(std::size_t n, std::size_t m)
{
	if (m != 0 && n > SIZE_MAX / m)
	{
		errno = ENOMEM;
		return nullptr;
	}
	void* p = alloc(16, n * m);
	if (p != nullptr)
	{
		std::memset(p, 0, n * m);
	}
	return p;
}

FAKE_ALLOC_API std::size_t malloc_usable_size(void* p) { return p == nullptr ? 0 : header_of(p)->size; }

FAKE_ALLOC_API void* realloc(void* p, std::size_t n)
{
	if (p == nullptr)
	{
		return malloc(n);
	}
	if (header_of(p)->magic != kMagic)
	{
		die("fake_alloc: realloc() of a pointer this allocator did not make\n");
	}
	if (n == 0)
	{
		free(p); // as glibc: realloc(p, 0) frees p and returns NULL
		return nullptr;
	}
	void* q = malloc(n);
	if (q == nullptr)
	{
		return nullptr;
	}
	const std::size_t old = header_of(p)->size;
	std::memcpy(q, p, old < n ? old : n);
	free(p);
	return q;
}

FAKE_ALLOC_API int posix_memalign(void** out, std::size_t align, std::size_t n)
{
	void* p = alloc(align, n);
	if (p == nullptr)
	{
		return ENOMEM;
	}
	*out = p;
	return 0;
}

FAKE_ALLOC_API void* aligned_alloc(std::size_t align, std::size_t n) { return alloc(align, n); }

FAKE_ALLOC_API void* memalign(std::size_t align, std::size_t n) { return alloc(align, n); }

FAKE_ALLOC_API void* valloc(std::size_t n)
{
	return alloc(static_cast<std::size_t>(sysconf(_SC_PAGESIZE)), n);
}

FAKE_ALLOC_API void* pvalloc(std::size_t n)
{
	const std::size_t pg = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
	return alloc(pg, (n + pg - 1) / pg * pg);
}

FAKE_ALLOC_API long fake_alloc_frees() { return g_frees.load(); }

FAKE_ALLOC_API void fake_alloc_fail_next(int n) { t_fail_next = n; }
