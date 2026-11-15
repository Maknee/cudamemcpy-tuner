#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <sys/auxv.h>
#endif

#include "common/error.h"

#include "arch/aarch64/variants.h"
#include "arch/x86_64/variants.h"

#if defined(__x86_64__)
#define CMT_NATIVE_COPY_VARIANTS(X) CMT_X86_64_COPY_VARIANTS(X)
#elif defined(__aarch64__)
#define CMT_NATIVE_COPY_VARIANTS(X) CMT_AARCH64_COPY_VARIANTS(X)
#endif

namespace cmt {

enum class Arch : std::uint8_t
{
	X86_64,
	Aarch64
};

enum class Isa : std::uint8_t
{
	Base,
	Sse2,
	Avx2,
	Avx512,
	Neon,
	Sve
};

enum class Method : std::uint8_t
{
	Memcpy,
	Movsb,
	Cached,
	NtStore,
	NtLoad,
	NtStorePrefetch
};

enum class WriteBack : std::uint8_t
{
	None,
	Flush,
	Clean
};

struct LoopName
{
	Method method = Method::Memcpy;
	std::optional<Isa> isa;
	WriteBack write_back = WriteBack::None;
};

constexpr LoopName PinnedName(Method m, Isa i, WriteBack w = WriteBack::None)
{
	return LoopName{m, i == Isa::Base ? std::optional<Isa>() : std::optional<Isa>(i), w};
}

struct Variant
{
	Method method = Method::Memcpy;
	Isa isa = Isa::Base;
};

struct CpuIsas
{
	std::uint16_t mask = 0;
	std::size_t sve_bytes = 0;

	bool Has(Isa i) const
	{
		return (mask & (1u << static_cast<unsigned>(i))) != 0;
	}

	void Add(Isa i)
	{
		mask = static_cast<std::uint16_t>(mask | (1u << static_cast<unsigned>(i)));
	}
};

Arch NativeArch();
const char* ArchName(Arch a);

CpuIsas DetectCpuIsas();

const char* IsaName(Isa i);
const char* MethodName(Method m);
const char* WriteBackName(WriteBack w);
std::size_t VectorBytes(Isa i, const CpuIsas& cpu);

std::optional<Isa> ParseIsa(Arch arch, std::string_view name);

template <class T>
struct ConstList
{
	const T* first = nullptr;
	std::size_t count = 0;

	const T* begin() const
	{
		return first;
	}

	const T* end() const
	{
		return first + count;
	}

	std::size_t size() const
	{
		return count;
	}

	const T& operator[](std::size_t i) const
	{
		return first[i];
	}
};

ConstList<Variant> Variants(Arch arch);
std::vector<Method> Methods(Arch arch);

std::optional<LoopName> ParseLoopName(Arch arch, std::string_view name);

std::string LoopNameText(const LoopName& n);
std::string VariantText(const Variant& v);

std::optional<Variant> Resolve(Arch arch, const LoopName& name, const CpuIsas& cpu, std::optional<Isa> pin);

LoopName DefaultLoopName(Arch arch);

#if defined(__x86_64__)
constexpr Arch kNativeArch = Arch::X86_64;
#elif defined(__aarch64__)
constexpr Arch kNativeArch = Arch::Aarch64;
#else
#error                                                                                                       \
	"libcuda_memcpy_hook supports x86-64 and aarch64 only (its patch tables and copy loops are per architecture)"
#endif

inline void CpuRelax()
{
#if defined(__x86_64__)
	__builtin_ia32_pause();
#else
	__asm__ __volatile__("yield" ::: "memory");
#endif
}

namespace hwcopy {

constexpr std::size_t kDefaultFlushBatch = 4096;

} // namespace hwcopy

struct CopyCtx
{
	std::size_t flush_batch =
		hwcopy::kDefaultFlushBatch;
	WriteBack write_back = WriteBack::None;
};

enum class VariantId : std::uint8_t
{
#define CMT_VARIANT_ENUM(id, method, isa, fn) id,
	CMT_NATIVE_COPY_VARIANTS(CMT_VARIANT_ENUM)
#undef CMT_VARIANT_ENUM
};

struct CopyLoop
{
	VariantId id = VariantId::Memcpy;
	CopyCtx ctx;
};

static_assert(std::is_trivially_copyable_v<CopyLoop>, "worker jobs carry CopyLoop by value");

namespace hwcopy {

#define CMT_VARIANT_DECL(id, method, isa, fn) void fn(void* dst, const void* src, std::size_t bytes);
CMT_NATIVE_COPY_VARIANTS(CMT_VARIANT_DECL)
#undef CMT_VARIANT_DECL

std::size_t WriteBackLineBytes();
void WriteBackLines(WriteBack wb, const void* p, std::size_t bytes);
void WriteBackFence();

const cmt::CpuIsas& GetCpu();

cmt::Variant VariantOf(VariantId id);
std::optional<VariantId> GetVariantId(const cmt::Variant& v);

struct Selection
{
	CopyLoop loop;
	bool fell_back = false;
};

std::optional<Selection> Select(std::string_view name, CopyCtx ctx, std::optional<Isa> pin,
								const cmt::CpuIsas& cpu);

CopyLoop DefaultLoop(CopyCtx ctx, std::optional<Isa> pin, const cmt::CpuIsas& cpu);

std::string Describe(const CopyLoop& loop);

template <class Copy, class WriteBack, class Fence>
void CopyThenWriteBack(std::size_t batch, std::size_t line, uint8_t* dst, const uint8_t* src,
					   std::size_t bytes, Copy&& copy, WriteBack&& write_back, Fence&& fence)
{
	if (batch == 0 || bytes <= batch)
	{
		copy(dst, src, bytes);
		write_back(dst, bytes);
		fence();
		return;
	}
	const uintptr_t line_mask = ~uintptr_t(line - 1);
	const uintptr_t end = reinterpret_cast<uintptr_t>(dst) + bytes;
	uintptr_t cur = reinterpret_cast<uintptr_t>(dst);
	while (cur < end)
	{
		uintptr_t next = (cur + batch) & line_mask;
		if (next <= cur || next > end)
		{
			next = end;
		}
		const std::size_t n = next - cur;
		const std::size_t off = cur - reinterpret_cast<uintptr_t>(dst);
		copy(dst + off, src + off, n);
		write_back(dst + off, n);
		cur = next;
	}
	fence();
}

template <void (*Body)(void*, const void*, std::size_t)>
void CopyWithWriteBack(CopyCtx ctx, void* dst, const void* src, std::size_t bytes)
{
	const WriteBack wb = ctx.write_back;
	CopyThenWriteBack(
		ctx.flush_batch, WriteBackLineBytes(), static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src),
		bytes, [](uint8_t* d, const uint8_t* s, std::size_t n) { Body(d, s, n); },
		[wb](const uint8_t* p, std::size_t n) { WriteBackLines(wb, p, n); }, [] { WriteBackFence(); });
}

template <VariantId Id>
struct Kernel;
#define CMT_VARIANT_SPEC(id, m, i, fn)                                                                       \
	template <>                                                                                              \
	struct Kernel<VariantId::id>                                                                             \
	{                                                                                                        \
		static constexpr VariantId kId = VariantId::id;                                                      \
		static constexpr Method kMethod = Method::m;                                                         \
		static constexpr Isa kIsa = Isa::i;                                                                  \
		static void Copy(CopyCtx ctx, void* dst, const void* src, std::size_t bytes)                         \
		{                                                                                                    \
			if (ctx.write_back == WriteBack::None)                                                           \
			{                                                                                                \
				fn(dst, src, bytes);                                                                         \
				return;                                                                                      \
			}                                                                                                \
			CopyWithWriteBack<&fn>(ctx, dst, src, bytes);                                                    \
		}                                                                                                    \
	};
CMT_NATIVE_COPY_VARIANTS(CMT_VARIANT_SPEC)
#undef CMT_VARIANT_SPEC

template <class F>
decltype(auto) WithLoop(VariantId id, F&& f)
{
	switch (id)
	{
#define CMT_VARIANT_CASE(id, method, isa, fn)                                                                \
	case VariantId::id:                                                                                      \
		return f(Kernel<VariantId::id>{});
		CMT_NATIVE_COPY_VARIANTS(CMT_VARIANT_CASE)
#undef CMT_VARIANT_CASE
	}
	return f(Kernel<VariantId::Memcpy>{});
}

inline void Run(const CopyLoop& loop, void* dst, const void* src, std::size_t bytes)
{
	WithLoop(loop.id, [&](auto k) { decltype(k)::Copy(loop.ctx, dst, src, bytes); });
}

} // namespace hwcopy

void StoreFence();

} // namespace cmt

namespace cmt {

void FlushRange(const void* p, std::size_t bytes);
bool FlushSupported();

} // namespace cmt
