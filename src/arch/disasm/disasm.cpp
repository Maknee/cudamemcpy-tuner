#include "arch/disasm/disasm.h"

namespace cmt {
namespace patch {
namespace disasm {
namespace {

constexpr std::size_t kAlign = 16;
constexpr std::size_t kHeader = 16;

alignas(kAlign) unsigned char g_arena[kArenaBytes];
std::size_t g_used = 0;
std::size_t g_peak = 0;
std::atomic<bool> g_busy{false};

std::size_t RoundUp(std::size_t n)
{
	return (n + kAlign - 1) & ~(kAlign - 1);
}

std::size_t BlockSize(void* p)
{
	std::size_t n;
	std::memcpy(&n, static_cast<unsigned char*>(p) - kHeader, sizeof(n));
	return n;
}

void* ArenaMalloc(std::size_t size)
{
	const std::size_t need = kHeader + RoundUp(size == 0 ? 1 : size);
	if (need > kArenaBytes - g_used)
	{
		return nullptr;
	}
	unsigned char* block = g_arena + g_used;
	std::memcpy(block, &size, sizeof(size));
	g_used += need;
	if (g_used > g_peak)
	{
		g_peak = g_used;
	}
	return block + kHeader;
}

void* ArenaCalloc(std::size_t nmemb, std::size_t size)
{
	if (size != 0 && nmemb > (kArenaBytes / size))
	{
		return nullptr;
	}
	void* p = ArenaMalloc(nmemb * size);
	if (p != nullptr)
	{
		std::memset(p, 0, nmemb * size);
	}
	return p;
}

void ArenaFree(void* /*ptr*/)
{
}

void* ArenaRealloc(void* ptr, std::size_t size)
{
	if (ptr == nullptr)
	{
		return ArenaMalloc(size);
	}
	const std::size_t old = BlockSize(ptr);
	if (size <= old)
	{
		return ptr;
	}
	void* p = ArenaMalloc(size);
	if (p != nullptr)
	{
		std::memcpy(p, ptr, old);
	}
	return p;
}

int ArenaVsnprintf(char* str, std::size_t size, const char* format, va_list ap)
{
	return std::vsnprintf(str, size, format, ap);
}

cs_arch ToCsArch(Arch a)
{
	return a == Arch::X86_64 ? CS_ARCH_X86 : CS_ARCH_ARM64;
}

cs_mode ToCsMode(Arch a)
{
	return a == Arch::X86_64 ? CS_MODE_64 : CS_MODE_LITTLE_ENDIAN;
}

} // namespace

Session::Session(Arch arch)
{
	bool expected = false;
	if (!g_busy.compare_exchange_strong(expected, true))
	{
		return;
	}
	owns_arena = true;
	g_used = 0;
	cs_opt_mem mem{};
	mem.malloc = ArenaMalloc;
	mem.calloc = ArenaCalloc;
	mem.realloc = ArenaRealloc;
	mem.free = ArenaFree;
	mem.vsnprintf = ArenaVsnprintf;
	if (cs_option(0, CS_OPT_MEM, reinterpret_cast<size_t>(&mem)) != CS_ERR_OK)
	{
		return;
	}
	if (cs_open(ToCsArch(arch), ToCsMode(arch), &handle) != CS_ERR_OK)
	{
		handle = 0;
		return;
	}
	if (cs_option(handle, CS_OPT_DETAIL, CS_OPT_ON) != CS_ERR_OK)
	{
		return;
	}
	instruction = cs_malloc(handle);
}

Session::~Session()
{
	if (instruction != nullptr)
	{
		cs_free(instruction, 1);
	}
	if (handle != 0)
	{
		cs_close(&handle);
	}
	if (owns_arena)
	{
		cs_opt_mem libc{};
		libc.malloc = std::malloc;
		libc.calloc = std::calloc;
		libc.realloc = std::realloc;
		libc.free = std::free;
		libc.vsnprintf = std::vsnprintf;
		cs_option(0, CS_OPT_MEM, reinterpret_cast<size_t>(&libc));
		g_used = 0;
		g_busy.store(false);
	}
}

const cs_insn* Session::Next(const uint8_t** code, std::size_t* avail, uint64_t* addr)
{
	if (instruction == nullptr)
	{
		return nullptr;
	}
	if (!cs_disasm_iter(handle, code, avail, addr, instruction))
	{
		return nullptr;
	}
	return instruction->detail != nullptr ? instruction : nullptr;
}

bool Session::InGroup(const cs_insn* insn, cs_group_type group) const
{
	return cs_insn_group(handle, insn, group);
}

std::size_t ArenaUsed()
{
	return g_used;
}

std::size_t ArenaPeak()
{
	return g_peak;
}

} // namespace disasm
} // namespace patch
} // namespace cmt
