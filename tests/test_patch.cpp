// test_patch.cpp - live patches on this machine's architecture: a wrap and a full replace of synthetic
// functions in an executable page, then calls through them. Pure planning (both architectures, both tables)
// is in test_patch_plan.cpp; this test is the part that writes code and runs it.
//
// x86-64: the wrapped function has a RIP-relative load in the moved bytes, so the relocation is exercised for
// real. aarch64 (native, or under qemu in the cross build): the moved prologue is stp/mov/nop/nop and the
// body loads its constant with adr (PC-relative), which stays in place after the moved bytes; a function with
// adr inside the first 16 bytes must be refused and left untouched. Under qemu this proves the bytes and the
// control flow; it cannot prove that __builtin___clear_cache is sufficient on real hardware (qemu's
// instruction cache is coherent).
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <initializer_list>

#include "arch/copy.h"
#include "check.h"
#include "hook.h"
#include "cuda/versions.h"

// The patcher's mprotect calls resolve to this definition (the patcher is linked into this executable). When
// armed, requests for exactly PROT_READ | PROT_EXEC first pass `g_rx_pass` times, then fail `g_rx_fail` times
// with EACCES.
int g_rx_pass = 0;
int g_rx_fail = 0;
extern "C" int mprotect(void* addr, size_t len, int prot) noexcept
{
	if (prot == (PROT_READ | PROT_EXEC) && g_rx_fail > 0)
	{
		if (g_rx_pass > 0)
		{
			--g_rx_pass;
		}
		else
		{
			--g_rx_fail;
			errno = EACCES;
			return -1;
		}
	}
	return static_cast<int>(syscall(SYS_mprotect, addr, len, prot));
}

namespace {

void* rwx_page()
{
	void* p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	return p == MAP_FAILED ? nullptr : p;
}

using IntFn = int (*)(int);

template <class Fn>
Fn as_fn(void* p)
{
	Fn f;
	std::memcpy(&f, &p, sizeof(f));
	return f;
}

template <class Fn>
void* as_ptr(Fn f)
{
	void* p;
	std::memcpy(&p, &f, sizeof(p));
	return p;
}

void place(uint8_t* code, const void* body, std::size_t len, int64_t constant)
{
	std::memcpy(code, body, len);
	std::memcpy(code + 256, &constant, 8);
	__builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code) + 4096);
}

#if defined(__x86_64__)
// int f(int a) { return *(int64_t*)(rip-relative data) + a; } with a 15-byte prologue: push rbp; mov rbp,rsp;
// mov rax,[rip+d]; add eax,edi; push r15 | pop r15; pop rbp; ret.
void* make_function(int64_t constant)
{
	auto* code = static_cast<uint8_t*>(rwx_page());
	if (code == nullptr)
	{
		return nullptr;
	}
	uint8_t body[] = {0x55, 0x48, 0x89, 0xe5, 0x48, 0x8b, 0x05, 0,	  0,   0,
					  0,	0x01, 0xf8, 0x41, 0x57, 0x41, 0x5f, 0x5d, 0xc3};
	const int32_t disp = 256 - 11; // data at +256, rip = end of the mov (+11)
	std::memcpy(body + 7, &disp, 4);
	place(code, body, sizeof(body), constant);
	return code;
}

// A prologue that starts with a call: its displacement cannot move.
void* make_unmovable_function()
{
	auto* code = static_cast<uint8_t*>(rwx_page());
	if (code != nullptr)
	{
		const uint8_t body[] = {0xe8, 0,	0,	  0,	0,	  0x90, 0x90, 0x90,
								0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
		place(code, body, sizeof(body), 0);
	}
	return code;
}
#elif defined(__aarch64__)
void put_words(uint8_t* out, std::initializer_list<uint32_t> insns)
{
	std::size_t i = 0;
	for (uint32_t w : insns)
	{
		std::memcpy(out + 4 * i++, &w, 4);
	}
}

// int f(int a) { return *(int64_t*)data + a; } (encodings checked with the cross assembler):
//   stp x29,x30,[sp,#-16]!; mov x29,sp; nop; nop | adr x1,data; ldr x1,[x1]; add w0,w0,w1; ldp
//   x29,x30,[sp],#16; ret
void* make_function(int64_t constant)
{
	auto* code = static_cast<uint8_t*>(rwx_page());
	if (code == nullptr)
	{
		return nullptr;
	}
	uint8_t body[36];
	put_words(body, {0xa9bf7bfd, 0x910003fd, 0xd503201f, 0xd503201f, 0x10000781 /* adr x1, .+240 = +256 */,
					 0xf9400021, 0x0b010000, 0xa8c17bfd, 0xd65f03c0});
	place(code, body, sizeof(body), constant);
	return code;
}

// adr inside the first 16 bytes: PC-relative, must not move.
void* make_unmovable_function()
{
	auto* code = static_cast<uint8_t*>(rwx_page());
	if (code != nullptr)
	{
		uint8_t body[20];
		put_words(body, {0xa9bf7bfd, 0x10000781, 0x910003fd, 0xd503201f, 0xd65f03c0});
		place(code, body, sizeof(body), 0);
	}
	return code;
}
#endif

IntFn g_orig = nullptr;

int wrapper(int a)
{
	return g_orig(a) + 1;
}

int replacement(int a)
{
	return a * 3;
}

void test_wrap()
{
	void* fn = make_function(1000);
	if (fn == nullptr)
	{
		std::printf("skip wrap: no RWX mapping\n");
		return;
	}
	CHECK(as_fn<IntFn>(fn)(5) == 1005);
	void* orig = nullptr;
	const auto installed =
		cmt::patch::InstallWrapper(reinterpret_cast<uintptr_t>(fn), as_ptr(&wrapper), &orig, "test");
	CHECK(installed);
	if (!installed)
	{
		return;
	}
	g_orig = as_fn<IntFn>(orig);
#if defined(__x86_64__)
	CHECK(static_cast<uint8_t*>(fn)[14] == 0xcc); // stolen 15: one int3 pad byte
#endif
	CHECK(g_orig(5) == 1005);			// the moved prologue, then the rest of the original
	CHECK(as_fn<IntFn>(fn)(5) == 1006); // the entry now goes through the wrapper
}

void test_full_replace()
{
	void* fn = make_function(7);
	if (fn == nullptr)
	{
		return;
	}
	CHECK(cmt::patch::VerifyBytes(reinterpret_cast<uintptr_t>(fn), static_cast<const uint8_t*>(fn), 16));
	const uint8_t wrong[16] = {0x90};
	CHECK(!cmt::patch::VerifyBytes(reinterpret_cast<uintptr_t>(fn), wrong, 16));
	CHECK(cmt::patch::InstallFullReplacement(reinterpret_cast<uintptr_t>(fn), as_ptr(&replacement), "test"));
	CHECK(as_fn<IntFn>(fn)(5) == 15);
}

void test_refuse()
{
	void* fn = make_unmovable_function();
	if (fn == nullptr)
	{
		return;
	}
	uint8_t before[16];
	std::memcpy(before, fn, sizeof(before));
	void* orig = nullptr;
	const auto refused =
		cmt::patch::InstallWrapper(reinterpret_cast<uintptr_t>(fn), as_ptr(&wrapper), &orig, "test");
	CHECK(!refused && refused.error().IsUnsupported());
	CHECK(orig == nullptr && std::memcmp(before, fn, sizeof(before)) == 0); // nothing written
}

// This process's own objects carry no BTI request (no -mbranch-protection), so the library preflight lets the
// patcher work on them; on x86-64 it is always true.
void test_protection_error()
{
	void* site = rwx_page();
	if (site == nullptr)
	{
		return;
	}
	munmap(site, 4096);
	const auto failed = cmt::patch::InstallFullReplacement(reinterpret_cast<uintptr_t>(site),
														   as_ptr(&replacement), "unmapped test page");
	CHECK(!failed && failed.error().GetKind() == cmt::Error::Kind::Os && failed.error().GetCode() != 0);
}

// Restoring R-X after writing a patch fails: the original bytes go back, R-X is restored on the second try, the
// install fails and the function behaves as before (never left writable with the patch in it). Mutation: return
// the error without reverting -> the function is the replacement / the wrapper.
void test_rx_restore_failure()
{
	void* fn = make_function(40);
	if (fn == nullptr)
	{
		return;
	}
	uint8_t before[16];
	std::memcpy(before, fn, sizeof(before));
	g_rx_pass = 0;
	g_rx_fail = 1;
	const auto failed =
		cmt::patch::InstallFullReplacement(reinterpret_cast<uintptr_t>(fn), as_ptr(&replacement), "rx test");
	CHECK(g_rx_fail == 0);
	CHECK(!failed && failed.error().GetKind() == cmt::Error::Kind::Os);
	CHECK(std::memcmp(before, fn, sizeof(before)) == 0);
	CHECK(as_fn<IntFn>(fn)(5) == 45); // the original code, executable again
	// The wrapper: its trampoline page becomes R-X first (passes), then the site's restore fails once.
	void* wrapped = make_function(50);
	if (wrapped == nullptr)
	{
		return;
	}
	std::memcpy(before, wrapped, sizeof(before));
	void* orig = nullptr;
	g_rx_pass = 1;
	g_rx_fail = 1;
	const auto refused =
		cmt::patch::InstallWrapper(reinterpret_cast<uintptr_t>(wrapped), as_ptr(&wrapper), &orig, "rx test");
	CHECK(g_rx_fail == 0);
	CHECK(!refused && refused.error().GetKind() == cmt::Error::Kind::Os);
	CHECK(std::memcmp(before, wrapped, sizeof(before)) == 0);
	CHECK(as_fn<IntFn>(wrapped)(5) == 55);
}

void test_preflight()
{
	const char* why = nullptr;
	CHECK(cmt::patch::IsLibraryPatchable(reinterpret_cast<uintptr_t>(&test_preflight), &why) && *why == '\0');
}

} // namespace

int main()
{
	std::printf("native arch: %s\n", cmt::ArchName(cmt::kNativeArch));
	test_wrap();
	test_full_replace();
	test_refuse();
	test_protection_error();
	test_rx_restore_failure();
	test_preflight();
	return check::exit_code();
}
