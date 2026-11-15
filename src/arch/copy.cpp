#include "arch/copy.h"

namespace cmt {

namespace {

#define CMT_VARIANT_CATALOG(id, method, isa, fn) {Method::method, Isa::isa},
constexpr Variant kX86Variants[] = {CMT_X86_64_COPY_VARIANTS(CMT_VARIANT_CATALOG)};
constexpr Variant kA64Variants[] = {CMT_AARCH64_COPY_VARIANTS(CMT_VARIANT_CATALOG)};
#undef CMT_VARIANT_CATALOG

struct MethodNameEntry
{
	Method method;
	const char* name;
};

constexpr MethodNameEntry kMethodNames[] = {
	{Method::Memcpy, "memcpy"},	 {Method::Movsb, "movsb"},
	{Method::Cached, "cached"},	 {Method::NtStore, "nt_store"},
	{Method::NtLoad, "nt_load"}, {Method::NtStorePrefetch, "nt_store_prefetch"},
};

struct IsaNameEntry
{
	Isa isa;
	const char* name;
};

constexpr IsaNameEntry kIsaNames[] = {
	{Isa::Base, "base"},	 {Isa::Sse2, "sse2"}, {Isa::Avx2, "avx2"},
	{Isa::Avx512, "avx512"}, {Isa::Neon, "neon"}, {Isa::Sve, "sve"},
};

bool HasVariant(Arch arch, Method m, Isa i)
{
	const ConstList<Variant> vs = Variants(arch);
	return std::any_of(std::begin(vs), std::end(vs),
					   [&](const Variant& v) { return v.method == m && v.isa == i; });
}

bool HasAnyVariant(Arch arch, Method m)
{
	const ConstList<Variant> vs = Variants(arch);
	return std::any_of(std::begin(vs), std::end(vs), [&](const Variant& v) { return v.method == m; });
}

bool StoresToCache(Method m)
{
	return m != Method::NtStore && m != Method::NtStorePrefetch;
}

#if defined(__aarch64__)
__attribute__((target("+sve"))) std::size_t SveVectorBytes()
{
	std::uint64_t v = 0;
	__asm__ __volatile__("cntb %0" : "=r"(v));
	return static_cast<std::size_t>(v);
}
#endif

} // namespace

Arch NativeArch()
{
#if defined(__aarch64__)
	return Arch::Aarch64;
#else
	return Arch::X86_64;
#endif
}

const char* ArchName(Arch a)
{
	return a == Arch::X86_64 ? "x86_64" : "aarch64";
}

CpuIsas DetectCpuIsas()
{
	CpuIsas c;
	c.Add(Isa::Base);
#if defined(__x86_64__)
	c.Add(Isa::Sse2);
	unsigned a = 0;
	unsigned b = 0;
	unsigned cx = 0;
	unsigned d = 0;
	if (!__get_cpuid(1, &a, &b, &cx, &d) || (cx & (1u << 27)) == 0)
	{
		return c;
	}
	unsigned xcr0 = 0;
	unsigned xcr0_hi = 0;
	__asm__ __volatile__("xgetbv" : "=a"(xcr0), "=d"(xcr0_hi) : "c"(0));
	if (!__get_cpuid_count(7, 0, &a, &b, &cx, &d))
	{
		return c;
	}
	if ((b & (1u << 5)) != 0 && (xcr0 & 0x6u) == 0x6u)
	{
		c.Add(Isa::Avx2);
	}
	if ((b & (1u << 16)) != 0 && (xcr0 & 0xE6u) == 0xE6u)
	{
		c.Add(Isa::Avx512);
	}
#elif defined(__aarch64__)
	constexpr unsigned long kHwcapAsimd =
		1ul << 1;
	constexpr unsigned long kHwcapSve = 1ul << 22;
	const unsigned long hw = getauxval(AT_HWCAP);
	if ((hw & kHwcapAsimd) != 0)
	{
		c.Add(Isa::Neon);
	}
	if ((hw & kHwcapSve) != 0)
	{
		c.Add(Isa::Sve);
		c.sve_bytes = SveVectorBytes();
	}
#endif
	return c;
}

const char* IsaName(Isa i)
{
	for (const IsaNameEntry& n : kIsaNames)
	{
		if (n.isa == i)
		{
			return n.name;
		}
	}
	return "?";
}

const char* MethodName(Method m)
{
	for (const MethodNameEntry& n : kMethodNames)
	{
		if (n.method == m)
		{
			return n.name;
		}
	}
	return "?";
}

const char* WriteBackName(WriteBack w)
{
	return w == WriteBack::Flush ? "flush" : w == WriteBack::Clean ? "clean" : "";
}

std::size_t VectorBytes(Isa i, const CpuIsas& cpu)
{
	switch (i)
	{
	case Isa::Base:
		return 0;
	case Isa::Sse2:
		return 16;
	case Isa::Avx2:
		return 32;
	case Isa::Avx512:
		return 64;
	case Isa::Neon:
		return 16;
	case Isa::Sve:
		return cpu.sve_bytes;
	}
	return 0;
}

std::optional<Isa> ParseIsa(Arch arch, std::string_view name)
{
	for (const IsaNameEntry& n : kIsaNames)
	{
		if (n.isa != Isa::Base && name == n.name)
		{
			const bool mine = arch == Arch::X86_64
								  ? (n.isa == Isa::Sse2 || n.isa == Isa::Avx2 || n.isa == Isa::Avx512)
								  : (n.isa == Isa::Neon || n.isa == Isa::Sve);
			return mine ? std::optional<Isa>(n.isa) : std::nullopt;
		}
	}
	return std::nullopt;
}

template <class T, std::size_t N>
constexpr ConstList<T> TableOf(const T (&a)[N])
{
	return ConstList<T>{a, N};
}

ConstList<Variant> Variants(Arch arch)
{
	return arch == Arch::X86_64 ? TableOf(kX86Variants) : TableOf(kA64Variants);
}

std::vector<Method> Methods(Arch arch)
{
	std::vector<Method> out;
	for (const MethodNameEntry& n : kMethodNames)
	{
		if (HasAnyVariant(arch, n.method))
		{
			out.push_back(n.method);
		}
	}
	return out;
}

std::optional<LoopName> ParseLoopName(Arch arch, std::string_view name)
{
	LoopName n;
	if (const std::size_t plus = name.find('+'); plus != std::string_view::npos)
	{
		const std::string_view w = name.substr(plus + 1);
		if (w == "flush")
		{
			n.write_back = WriteBack::Flush;
		}
		else if (w == "clean")
		{
			n.write_back = WriteBack::Clean;
		}
		else
		{
			return std::nullopt;
		}
		name = name.substr(0, plus);
	}
	if (const std::size_t at = name.find('@'); at != std::string_view::npos)
	{
		n.isa = ParseIsa(arch, name.substr(at + 1));
		if (!n.isa)
		{
			return std::nullopt;
		}
		name = name.substr(0, at);
	}
	bool found = false;
	for (const MethodNameEntry& m :
		 kMethodNames)
	{
		if (name == m.name && HasAnyVariant(arch, m.method))
		{
			n.method = m.method;
			found = true;
		}
	}
	if (!found)
	{
		return std::nullopt;
	}
	if (n.isa && !HasVariant(arch, n.method, *n.isa))
	{
		return std::nullopt;
	}
	if (n.write_back != WriteBack::None && !StoresToCache(n.method))
	{
		return std::nullopt;
	}
	return n;
}

std::string VariantText(const Variant& v)
{
	std::string s = MethodName(v.method);
	if (v.isa != Isa::Base)
	{
		s += std::string("@") + IsaName(v.isa);
	}
	return s;
}

std::string LoopNameText(const LoopName& n)
{
	std::string s = MethodName(n.method);
	if (n.isa && *n.isa != Isa::Base)
	{
		s += std::string("@") + IsaName(*n.isa);
	}
	if (n.write_back != WriteBack::None)
	{
		s += std::string("+") + WriteBackName(n.write_back);
	}
	return s;
}

std::optional<Variant> Resolve(Arch arch, const LoopName& name, const CpuIsas& cpu, std::optional<Isa> pin)
{
	const ConstList<Variant> vs = Variants(arch);
	auto runs = [&](Isa i) { return i == Isa::Base || cpu.Has(i); };
	auto widest = [&](auto fits)
	{
		std::optional<Variant> best;
		for (const Variant& v : vs)
		{
			if (v.method == name.method && runs(v.isa) && fits(v.isa) &&
				(!best || VectorBytes(v.isa, cpu) > VectorBytes(best->isa, cpu)))
			{
				best = v;
			}
		}
		return best;
	};
	if (pin)
	{
		if (!runs(*pin))
		{
			return std::nullopt;
		}
		if (HasVariant(arch, name.method, *pin))
		{
			return Variant{name.method, *pin};
		}
		const std::size_t cap = VectorBytes(*pin, cpu);
		return widest([&](Isa i) { return VectorBytes(i, cpu) < cap; });
	}
	if (name.isa)
	{
		if (HasVariant(arch, name.method, *name.isa) && runs(*name.isa))
		{
			return Variant{name.method, *name.isa};
		}
		return std::nullopt;
	}
	return widest([](Isa) { return true; });
}

LoopName DefaultLoopName(Arch arch)
{
	return arch == Arch::X86_64 ? PinnedName(Method::NtStore, Isa::Sse2)
								: PinnedName(Method::Memcpy, Isa::Base);
}

} // namespace cmt

namespace cmt {
namespace hwcopy {

namespace {

struct Entry
{
	VariantId id;
	cmt::Variant variant;
};

const Entry kVariants[] = {
#define CMT_VARIANT_ENTRY(id, m, i, fn) {VariantId::id, cmt::Variant{Method::m, Isa::i}},
	CMT_NATIVE_COPY_VARIANTS(CMT_VARIANT_ENTRY)
#undef CMT_VARIANT_ENTRY
};

CopyLoop LoopOf(const cmt::Variant& v, CopyCtx ctx)
{
	const std::optional<VariantId> id = GetVariantId(v);
	return CopyLoop{id ? *id : VariantId::Memcpy,
					ctx};
}

} // namespace

void CopyMemcpy(void* dst, const void* src, std::size_t bytes)
{
	std::memcpy(dst, src, bytes);
}

const cmt::CpuIsas& GetCpu()
{
	static const cmt::CpuIsas c = cmt::DetectCpuIsas();
	return c;
}

cmt::Variant VariantOf(VariantId id)
{
	for (const Entry& e : kVariants)
	{
		if (e.id == id)
		{
			return e.variant;
		}
	}
	return cmt::Variant{};
}

std::optional<VariantId> GetVariantId(const cmt::Variant& v)
{
	for (const Entry& e : kVariants)
	{
		if (e.variant.method == v.method && e.variant.isa == v.isa)
		{
			return e.id;
		}
	}
	return std::nullopt;
}

CopyLoop DefaultLoop(CopyCtx ctx, std::optional<Isa> pin, const cmt::CpuIsas& c)
{
	ctx.write_back = WriteBack::None;
	const cmt::LoopName name = cmt::DefaultLoopName(kNativeArch);
	std::optional<cmt::Variant> v = cmt::Resolve(kNativeArch, name, c, pin);
	if (!v)
	{
		v = cmt::Resolve(kNativeArch, name, c,
						 std::nullopt);
	}
	return LoopOf(v ? *v : cmt::Variant{}, ctx);
}

std::optional<Selection> Select(std::string_view name, CopyCtx ctx, std::optional<Isa> pin,
								const cmt::CpuIsas& c)
{
	const std::optional<cmt::LoopName> parsed = cmt::ParseLoopName(kNativeArch, name);
	if (!parsed)
	{
		return std::nullopt;
	}
	const std::optional<cmt::Variant> v = cmt::Resolve(kNativeArch, *parsed, c, pin);
	if (!v)
	{
		return Selection{DefaultLoop(ctx, pin, c), true};
	}
	ctx.write_back = parsed->write_back;
	return Selection{LoopOf(*v, ctx), false};
}

std::string Describe(const CopyLoop& loop)
{
	std::string s = cmt::VariantText(VariantOf(loop.id));
	if (loop.ctx.write_back != WriteBack::None)
	{
		s += std::string("+") + cmt::WriteBackName(loop.ctx.write_back) + " (batch " +
			 std::to_string(loop.ctx.flush_batch) + ")";
	}
	return s;
}

} // namespace hwcopy
} // namespace cmt

namespace cmt {

bool FlushSupported()
{
	static const bool supported = []
	{
#if defined(__x86_64__)
		constexpr unsigned kClflushopt = 1u << 23;
		unsigned a = 0;
		unsigned b = 0;
		unsigned cx = 0;
		unsigned d = 0;
		return __get_cpuid_count(7, 0, &a, &b, &cx, &d) && (b & kClflushopt) != 0;
#else
		return false;
#endif
	}();
	return supported;
}

#if defined(__x86_64__)
__attribute__((target("clflushopt"))) static void FlushClflushopt(const void* p, std::size_t bytes)
{
	const auto a = reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{63};
	const auto e = reinterpret_cast<std::uintptr_t>(p) + bytes;
	for (std::uintptr_t c = a; c < e; c += 64)
	{
		_mm_clflushopt(reinterpret_cast<void*>(c));
	}
	_mm_mfence();
}
#endif

void FlushRange(const void* p, std::size_t bytes)
{
#if defined(__x86_64__)
	if (FlushSupported())
	{
		FlushClflushopt(p, bytes);
	}
#else
	(void)p, (void)bytes;
#endif
}

} // namespace cmt
