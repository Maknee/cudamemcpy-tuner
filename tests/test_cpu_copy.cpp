#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include "arch/copy.h"
#include "check.h"
#include "tests/check.h"

// What a loop name means on each architecture, and which variant a CPU runs for it (arch/copy.h). The
// CPU here is a parameter, so a test can be a CPU without AVX2 or with 128-bit SVE whatever machine runs it.

using namespace cmt;

namespace {

CpuIsas x86(std::initializer_list<Isa> isas)
{
	CpuIsas c;
	c.Add(Isa::Base);
	c.Add(Isa::Sse2);
	for (Isa i : isas)
	{
		c.Add(i);
	}
	return c;
}

CpuIsas a64(std::size_t sve_bytes)
{
	CpuIsas c;
	c.Add(Isa::Base);
	c.Add(Isa::Neon);
	if (sve_bytes != 0)
	{
		c.Add(Isa::Sve);
		c.sve_bytes = sve_bytes;
	}
	return c;
}

std::string runs(Arch a, const std::string& name, const CpuIsas& cpu, std::optional<Isa> pin = std::nullopt)
{
	const auto n = ParseLoopName(a, name);
	if (!n)
	{
		return "unknown";
	}
	const auto v = Resolve(a, *n, cpu, pin);
	return v ? VariantText(*v) : "none";
}

bool same(const LoopName& a, const LoopName& b)
{
	return a.method == b.method && a.isa == b.isa && a.write_back == b.write_back;
}

} // namespace

// Every old name means exactly the loop it named before: the method, PINNED to the ISA in its name.
TEST(legacy_aliases_are_rejected)
{
	for (const char* name : {"default", "quick", "sse_nt", "avx2_nt", "memcpy_flush", "neon_nt"})
	{
		CHECK(!ParseLoopName(Arch::X86_64, name));
		CHECK(!ParseLoopName(Arch::Aarch64, name));
	}
}

// An explicitly selected ISA stays fixed on wider CPUs.
TEST(explicit_names_run_their_own_variant)
{
	const CpuIsas all = x86({Isa::Avx2, Isa::Avx512});
	CHECK_EQ(runs(Arch::X86_64, "nt_store@avx2", all), std::string("nt_store@avx2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store@sse2", all), std::string("nt_store@sse2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store@sse2", all),
			 std::string("nt_store@sse2")); // libcuda's movntps loop, not widened
	CHECK_EQ(runs(Arch::X86_64, "cached@avx2+flush", all), std::string("cached@avx2"));
	// A pinned ISA the CPU lacks: no variant (the hook then runs its default loop, as for avx512 on an AVX2
	// CPU before)
	CHECK_EQ(runs(Arch::X86_64, "nt_store@avx512", x86({Isa::Avx2})), std::string("none"));
	CHECK_EQ(runs(Arch::X86_64, "nt_load@avx2", x86({})), std::string("none"));
	CHECK_EQ(runs(Arch::Aarch64, "nt_store@neon", a64(64)),
			 std::string("nt_store@neon")); // pinned: SVE does not take it
}

// A bare method runs its widest variant this CPU has; a CPU without AVX2 falls back to SSE2.
TEST(a_method_runs_its_widest_variant)
{
	CHECK_EQ(runs(Arch::X86_64, "nt_store", x86({Isa::Avx2, Isa::Avx512})), std::string("nt_store@avx512"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store", x86({Isa::Avx2})), std::string("nt_store@avx2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store", x86({})), std::string("nt_store@sse2")); // no AVX2: SSE2
	CHECK_EQ(runs(Arch::X86_64, "cached", x86({})), std::string("cached@sse2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store_prefetch", x86({Isa::Avx2})),
			 std::string("nt_store_prefetch@avx2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_load", x86({})), std::string("none")); // no SSE2 variant (SSE4.1)
	CHECK_EQ(runs(Arch::X86_64, "movsb", x86({})), std::string("movsb"));
	CHECK_EQ(runs(Arch::X86_64, "memcpy+flush", x86({Isa::Avx512})), std::string("memcpy"));
	// aarch64: SVE only when its vectors are wider than NEON's; 128-bit SVE (Grace) keeps NEON
	CHECK_EQ(runs(Arch::Aarch64, "nt_store", a64(16)), std::string("nt_store@neon"));
	CHECK_EQ(runs(Arch::Aarch64, "nt_store", a64(32)), std::string("nt_store@sve"));
	CHECK_EQ(runs(Arch::Aarch64, "cached", a64(0)), std::string("cached@neon"));
	CHECK_EQ(runs(Arch::Aarch64, "nt_load", a64(256)), std::string("nt_load@sve"));
	CHECK_EQ(runs(Arch::Aarch64, "nt_store@sve", a64(0)), std::string("none"));
}

// STAGING_COPY_ISA overrides the requested ISA and caps method width.
TEST(a_pin_wins_and_caps)
{
	const CpuIsas all = x86({Isa::Avx2, Isa::Avx512});
	CHECK_EQ(runs(Arch::X86_64, "nt_store", all, Isa::Avx2), std::string("nt_store@avx2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store@avx512", all, Isa::Avx2), std::string("nt_store@avx2"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store@sse2", all, Isa::Avx512), std::string("nt_store@avx512"));
	CHECK_EQ(runs(Arch::X86_64, "nt_load", all, Isa::Sse2), std::string("none")); // never wider than the pin
	CHECK_EQ(runs(Arch::X86_64, "movsb", all, Isa::Avx512), std::string("movsb"));
	CHECK_EQ(runs(Arch::X86_64, "memcpy", all, Isa::Sse2), std::string("memcpy"));
	CHECK_EQ(runs(Arch::X86_64, "nt_store", x86({}), Isa::Avx2),
			 std::string("none")); // never run what the CPU lacks
	CHECK_EQ(runs(Arch::Aarch64, "nt_store", a64(16), Isa::Sve), std::string("nt_store@sve"));
}

TEST(the_grammar_and_what_it_refuses)
{
	CHECK_EQ(runs(Arch::X86_64, "cached@avx2+clean", x86({Isa::Avx2})), std::string("cached@avx2"));
	CHECK(ParseLoopName(Arch::X86_64, "cached@avx2+clean")->write_back == WriteBack::Clean);
	CHECK(ParseLoopName(Arch::Aarch64, "nt_load@sve+flush")->write_back == WriteBack::Flush);
	for (const char* bad :
		 {"", "nt", "warp_drive", "memcpy@avx2", "nt_load@sse2", "nt_store+flush", "nt_store_prefetch+clean",
		  "movsb@sse2", "cached@neon", "nt_store@sve", "cached+clwb", "cached@", "@avx2", "cached@avx2+",
		  "nt_store@neon", "neon", "memcpy_clean"})
	{
		CHECK(!ParseLoopName(Arch::X86_64, bad).has_value());
	}
	for (const char* bad : {"movsb", "nt_store_prefetch", "nt_store@avx2", "nt_store@sse2", "avx2",
							"memcpy_clwb", "movsb+flush", "cached@sse2"})
	{
		CHECK(!ParseLoopName(Arch::Aarch64, bad).has_value());
	}
	CHECK(!ParseIsa(Arch::X86_64, "neon") && !ParseIsa(Arch::Aarch64, "avx2") &&
		  !ParseIsa(Arch::X86_64, "base"));
	CHECK(ParseIsa(Arch::X86_64, "avx512") == Isa::Avx512);
}

// Every name the grammar writes reads back the same, and every variant of every architecture has one.
TEST(names_round_trip)
{
	for (Arch a : {Arch::X86_64, Arch::Aarch64})
	{
		for (const Variant& v : Variants(a))
		{
			const auto n = ParseLoopName(a, VariantText(v));
			CHECK(n.has_value());
			if (n)
			{
				CHECK(n->method == v.method);
			}
		}
		for (Method m : Methods(a))
		{
			for (WriteBack w : {WriteBack::None, WriteBack::Flush, WriteBack::Clean})
			{
				LoopName n;
				n.method = m;
				n.write_back = w;
				const auto back = ParseLoopName(a, LoopNameText(n));
				if (back)
				{
					CHECK(same(*back, n));
				}
			}
		}
	}
}

TEST(this_cpu)
{
	const CpuIsas c = DetectCpuIsas();
	CHECK(c.Has(Isa::Base));
#if defined(__x86_64__)
	CHECK(c.Has(Isa::Sse2));
	CHECK(!c.Has(Isa::Avx512) || c.Has(Isa::Avx2));
#endif
	std::printf("  this cpu:");
	for (Isa i : {Isa::Sse2, Isa::Avx2, Isa::Avx512, Isa::Neon, Isa::Sve})
	{
		if (c.Has(i))
		{
			std::printf(" %s", IsaName(i));
		}
	}
	std::printf("\n");
	CHECK(DefaultLoopName(Arch::X86_64).isa == Isa::Sse2); // libcuda's movntps loop, never widened
	CHECK(DefaultLoopName(Arch::Aarch64).method == Method::Memcpy);
}

// Every copy variant of this architecture through the CopyLoop dispatch (odd sizes,
// misaligned heads and tails, guard bytes around the destination), every write-back, and which variant a name
// selects: on this CPU, and on pretend CPUs (one without AVX2, one with 128-bit SVE) through the same
// selection code.
//
// Argument (aarch64 under qemu, tests/CMakeLists.txt): expect-sve=<bytes>, the SVE vector length the
// emulated CPU was given (QEMU_CPU), 0 for a CPU without SVE. The test then checks that detection saw exactly
// that and that a bare method name selects SVE only where its vectors are wider than NEON's.

namespace {

using cmt::CopyCtx;
using cmt::CopyLoop;
using cmt::Isa;
using cmt::VariantId;
using cmt::WriteBack;

constexpr std::size_t kGuard = 64;
constexpr uint8_t kGuardByte = 0xA5;

// Copies n bytes from src+src_off to dst+dst_off through `loop` and checks the result and that no byte
// outside the destination range changed.
bool copy_ok(const CopyLoop& loop, std::size_t n, std::size_t src_off, std::size_t dst_off)
{
	// Base pointers are page aligned, so the offsets set the alignment.
	check::Buffer src(n + src_off + 2 * kGuard);
	check::Buffer dst(n + dst_off + 2 * kGuard, kGuardByte);
	for (std::size_t i = 0; i < std::size(src); i++)
	{
		src[i] = check::pattern(i);
	}
	uint8_t* s = std::data(src) + kGuard + src_off;
	uint8_t* d = std::data(dst) + kGuard + dst_off;
	cmt::hwcopy::Run(loop, d, s, n);
	cmt::StoreFence();
	if (std::memcmp(d, s, n) != 0)
	{
		return false;
	}
	for (std::size_t i = 0; i < std::size(dst); i++)
	{
		const bool inside = i >= kGuard + dst_off && i < kGuard + dst_off + n;
		if (!inside && dst[i] != kGuardByte)
		{
			return false;
		}
	}
	return true;
}

bool cpu_runs(Isa i)
{
	return i == Isa::Base || cmt::hwcopy::GetCpu().Has(i);
}

std::string text(VariantId id)
{
	return cmt::VariantText(cmt::hwcopy::VariantOf(id));
}

// This build's variants: the list (arch/{x86_64,aarch64}/variants.h) in enum order.
std::vector<VariantId> all_ids()
{
	std::vector<VariantId> ids;
#define MH_ID(id, m, i, fn) ids.push_back(VariantId::id);
	CMT_NATIVE_COPY_VARIANTS(MH_ID)
#undef MH_ID
	return ids;
}

void test_variant(VariantId id)
{
	const cmt::Variant v = cmt::hwcopy::VariantOf(id);
	if (!cpu_runs(v.isa))
	{
		std::printf("skip %s (this CPU does not run %s)\n", text(id).c_str(), cmt::IsaName(v.isa));
		return;
	}
	const CopyLoop loop{id, CopyCtx{}};
	// Every size 0..300 with a few head offsets on both sides.
	for (std::size_t n = 0; n <= 300; n++)
	{
		for (std::size_t so : {0u, 1u, 15u, 33u})
		{
			for (std::size_t dof : {0u, 3u, 31u, 63u})
			{
				CHECKF(copy_ok(loop, n, so, dof), "%s n=%zu src_off=%zu dst_off=%zu", text(id).c_str(), n, so,
					   dof);
			}
		}
	}
	// All 64 x 64 alignment pairs on a size with head, body and tail.
	for (std::size_t so = 0; so < 64; so++)
	{
		for (std::size_t dof = 0; dof < 64; dof++)
		{
			CHECKF(copy_ok(loop, 4096 + 77, so, dof), "%s align src=%zu dst=%zu", text(id).c_str(), so, dof);
		}
	}
	// Large odd sizes (prefetch distance, unrolled body).
	for (std::size_t n : {std::size_t{447}, std::size_t{448}, std::size_t{449}, std::size_t{4095},
						  std::size_t{4096}, std::size_t{65537}, std::size_t{1} << 20,
						  (std::size_t{1} << 20) + 13, std::size_t{3} * 1000 * 1000 + 1})
	{
		CHECKF(copy_ok(loop, n, 5, 9), "%s large n=%zu", text(id).c_str(), n);
	}
	std::printf("ok   %s\n", text(id).c_str());
}

// The hook's variant list and cmt::variants() (what the tuner names) are the same list, and the one dispatch
// (with_loop) selects the instance of the id it is given, for every variant.
void test_list_and_dispatch()
{
	const std::vector<VariantId> ids = all_ids();
	const cmt::ConstList<cmt::Variant> want = cmt::Variants(cmt::kNativeArch);
	CHECK(std::size(ids) == std::size(want));
	for (std::size_t i = 0; i < std::size(ids) && i < std::size(want); i++)
	{
		const cmt::Variant v = cmt::hwcopy::VariantOf(ids[i]);
		CHECKF(v.method == want[i].method && v.isa == want[i].isa, "variant %zu: %s vs %s", i,
			   text(ids[i]).c_str(), cmt::VariantText(want[i]).c_str());
		CHECK(static_cast<std::size_t>(ids[i]) == i); // the enum follows the list
		CHECK(cmt::hwcopy::GetVariantId(v) == ids[i]);
		const VariantId got = cmt::hwcopy::WithLoop(ids[i], [](auto k) { return decltype(k)::kId; });
		CHECKF(got == ids[i], "with_loop(%s) dispatched to %s", text(ids[i]).c_str(), text(got).c_str());
	}
}

// select(): what a name runs, on this CPU and on pretend ones. Old names keep their variant; a bare method
// takes the widest the CPU runs; a CPU without the pinned ISA gets the default loop; the write-back comes
// from the name and the batch from the context.
void test_select()
{
	using cmt::hwcopy::Select;
	const cmt::CpuIsas here = cmt::hwcopy::GetCpu();
	CopyCtx ctx;
	ctx.flush_batch = 192;
	CHECK(!Select("warp_drive", ctx, std::nullopt, here));
	CHECK(!Select("default", ctx, std::nullopt, here));
#if defined(__x86_64__)
	cmt::CpuIsas sse2_only;
	sse2_only.Add(Isa::Base);
	sse2_only.Add(Isa::Sse2);
	cmt::CpuIsas avx512 = sse2_only;
	avx512.Add(Isa::Avx2);
	avx512.Add(Isa::Avx512);
	CHECK(cmt::hwcopy::DefaultLoop(ctx, std::nullopt, here).id == VariantId::NtStoreSse2);
	// The old names, exactly their old loops.
	const std::pair<const char*, VariantId> old[] = {
		{"memcpy", VariantId::Memcpy},
		{"movsb", VariantId::Movsb},
		{"nt_store@sse2", VariantId::NtStoreSse2},
		{"nt_store@avx2", VariantId::NtStoreAvx2},
		{"nt_store@avx2", VariantId::NtStoreAvx2},
		{"nt_store@avx512", VariantId::NtStoreAvx512},
		{"nt_store@avx512", VariantId::NtStoreAvx512},
		{"nt_load@avx2", VariantId::NtLoadAvx2},
		{"nt_load@avx512", VariantId::NtLoadAvx512},
		{"cached@avx2", VariantId::CachedAvx2},
		{"nt_store_prefetch@avx2", VariantId::NtStorePrefetchAvx2},
		{"nt_store@sse2", VariantId::NtStoreSse2},
		{"nt_store@sse2", VariantId::NtStoreSse2},
	};
	for (const auto& [name, id] : old)
	{
		const auto s = Select(name, ctx, std::nullopt, avx512);
		CHECKF(s && !s->fell_back && s->loop.id == id && s->loop.ctx.write_back == WriteBack::None, "%s",
			   name);
	}
	const auto mf = Select("movsb+flush", ctx, std::nullopt, avx512);
	CHECK(mf && mf->loop.id == VariantId::Movsb && mf->loop.ctx.write_back == WriteBack::Flush &&
		  mf->loop.ctx.flush_batch == 192);
	const auto cc = Select("cached@avx2+clean", ctx, std::nullopt, avx512);
	CHECK(cc && cc->loop.id == VariantId::CachedAvx2 && cc->loop.ctx.write_back == WriteBack::Clean);
	// A bare method: its widest variant; without AVX2, SSE2.
	CHECK(Select("nt_store", ctx, std::nullopt, avx512)->loop.id == VariantId::NtStoreAvx512);
	CHECK(Select("nt_store", ctx, std::nullopt, sse2_only)->loop.id == VariantId::NtStoreSse2);
	CHECK(Select("cached", ctx, std::nullopt, sse2_only)->loop.id == VariantId::CachedSse2);
	// A pinned ISA the CPU lacks, and a method with no variant it runs: the default loop, without the
	// write-back.
	const auto fb = Select("cached@avx2+flush", ctx, std::nullopt, sse2_only);
	CHECK(fb && fb->fell_back && fb->loop.id == VariantId::NtStoreSse2 &&
		  fb->loop.ctx.write_back == WriteBack::None);
	CHECK(Select("nt_load", ctx, std::nullopt, sse2_only)->fell_back);
	// STAGING_COPY_ISA pins every name, the old ones and the default included.
	CHECK(Select("nt_store@avx512", ctx, Isa::Avx2, avx512)->loop.id == VariantId::NtStoreAvx2);
	CHECK(Select("nt_store_prefetch", ctx, Isa::Sse2, avx512)->loop.id == VariantId::NtStorePrefetchSse2);
	CHECK(cmt::hwcopy::DefaultLoop(ctx, Isa::Avx512, avx512).id == VariantId::NtStoreAvx512);
#else
	cmt::CpuIsas neon;
	neon.Add(Isa::Base);
	neon.Add(Isa::Neon);
	cmt::CpuIsas sve128 = neon, sve256 = neon;
	sve128.Add(Isa::Sve);
	sve128.sve_bytes = 16;
	sve256.Add(Isa::Sve);
	sve256.sve_bytes = 32;
	CHECK(cmt::hwcopy::DefaultLoop(ctx, std::nullopt, here).id == VariantId::Memcpy);
	const std::pair<const char*, VariantId> old[] = {
		{"memcpy", VariantId::Memcpy},
		{"cached@neon", VariantId::CachedNeon},
		{"nt_store@neon", VariantId::NtStoreNeon},
		{"nt_store@neon", VariantId::NtStoreNeon},
		{"nt_load@neon", VariantId::NtLoadNeon},
		{"memcpy", VariantId::Memcpy},
	};
	for (const auto& [name, id] : old)
	{
		const auto s = Select(name, ctx, std::nullopt, sve256);
		CHECKF(s && !s->fell_back && s->loop.id == id, "%s", name);
	}
	const auto cf = Select("cached@neon+clean", ctx, std::nullopt, neon);
	CHECK(cf && cf->loop.id == VariantId::CachedNeon && cf->loop.ctx.write_back == WriteBack::Clean);
	CHECK(Select("nt_store", ctx, std::nullopt, neon)->loop.id == VariantId::NtStoreNeon);
	CHECK(Select("nt_store", ctx, std::nullopt, sve128)->loop.id ==
		  VariantId::NtStoreNeon); // Grace: NEON kept
	CHECK(Select("nt_store", ctx, std::nullopt, sve256)->loop.id == VariantId::NtStoreSve);
	CHECK(Select("cached@sve", ctx, std::nullopt, neon)->fell_back);
	CHECK(Select("nt_load", ctx, Isa::Sve, sve128)->loop.id == VariantId::NtLoadSve);
#endif
}

// The write-back batch is the copy's own (CopyCtx), not a process-wide setting. The batching template is the
// one every write-back runs (arch/copy.h); here it gets recording steps. Two contexts in flight at once,
// on two threads, each see their own batch; a process-wide value would let the second thread's apply to both.
void test_flush_batch_per_context()
{
	struct Recorder
	{
		std::vector<std::size_t> copies;
		std::size_t write_backs = 0;
		unsigned fences = 0;
	};

	auto run = [](std::size_t batch, std::size_t bytes, std::size_t dst_off, Recorder* r)
	{
		check::Buffer src(bytes + 64);
		check::Buffer dst(bytes + 64);
		cmt::hwcopy::CopyThenWriteBack(
			batch, 64, std::data(dst) + dst_off, std::data(src), bytes,
			[r](uint8_t* d, const uint8_t* s, std::size_t n)
			{
				r->copies.push_back(n);
				std::memcpy(d, s, n);
			},
			[r](const uint8_t*, std::size_t n) { r->write_backs += n; }, [r] { r->fences++; });
	};
	for (unsigned round = 0; round < 50; round++)
	{
		Recorder a, b;
		CopyCtx ca, cb;
		ca.flush_batch = 192;
		cb.flush_batch = 4096;
		std::thread ta([&] { run(ca.flush_batch, 20000, 0, &a); });
		std::thread tb([&] { run(cb.flush_batch, 20000, 0, &b); });
		ta.join();
		tb.join();
		// 20000 / 192 -> 104 batches of 192 and a tail of 32; 20000 / 4096 -> 4 batches and a tail of 3616.
		CHECK(std::size(a.copies) == 105 && a.copies.front() == 192 && a.copies.back() == 32);
		CHECK(std::size(b.copies) == 5 && b.copies.front() == 4096 && b.copies.back() == 3616);
		CHECK(a.write_backs == 20000 && b.write_backs == 20000 && a.fences == 1 && b.fences == 1);
	}
	// Batch ends fall on destination lines: a destination 13 bytes into a line ends its first batch 13 early.
	Recorder c;
	run(192, 1000, 13, &c);
	CHECK(!std::empty(c.copies) && c.copies.front() == 192 - 13);
	// 0 = copy everything, then one write-back.
	Recorder d;
	run(0, 50000, 0, &d);
	CHECK(std::size(d.copies) == 1 && d.copies[0] == 50000 && d.write_backs == 50000);
}

// The real write-backs after every cached-store variant this CPU runs, at every batch size: odd sizes, odd
// alignments, sizes on and around batch boundaries.
void test_write_backs()
{
	const std::size_t sizes[] = {
		0, 1, 63, 64, 65, 191, 4095, 4096, 4097, 8191, 12289, 65536 + 3, (1u << 20) + 77};
	const std::size_t batches[] = {0, 64, 192, 4096, 65536};
	unsigned ran = 0;
	for (VariantId id : all_ids())
	{
		const cmt::Variant v = cmt::hwcopy::VariantOf(id);
		if (!cpu_runs(v.isa) || v.method == cmt::Method::NtStore || v.method == cmt::Method::NtStorePrefetch)
		{
			continue;
		}
		for (WriteBack wb : {WriteBack::Flush, WriteBack::Clean})
		{
			for (std::size_t batch : batches)
			{
				CopyCtx ctx;
				ctx.flush_batch = batch;
				ctx.write_back = wb;
				for (std::size_t n : sizes)
				{
					for (std::size_t dof : {0u, 1u, 37u, 63u})
					{
						CHECKF(copy_ok(CopyLoop{id, ctx}, n, 7, dof), "%s+%s batch=%zu n=%zu dst_off=%zu",
							   text(id).c_str(), cmt::WriteBackName(wb), batch, n, dof);
					}
				}
			}
			ran++;
		}
	}
	CHECK(ran >= 2); // memcpy at least, both write-backs
	std::printf("ok   write-backs (%u variant x write-back pairs)\n", ran);
}

// aarch64 under qemu: detection saw the emulated CPU, and a bare method selects SVE only where it is wider.
void test_expected_cpu(const char* arg)
{
	const char* const key = "expect-sve=";
	if (arg == nullptr || std::strncmp(arg, key, std::strlen(key)) != 0)
	{
		return;
	}
	const std::size_t want = std::strtoul(arg + std::strlen(key), nullptr, 10);
	const cmt::CpuIsas& c = cmt::hwcopy::GetCpu();
	CHECKF(c.Has(Isa::Sve) == (want != 0) && c.sve_bytes == want, "sve detected %d, %zu bytes; expected %zu",
		   c.Has(Isa::Sve), c.sve_bytes, want);
	const auto s = cmt::hwcopy::Select("nt_store", CopyCtx{}, std::nullopt, c);
	const bool sve = s && s->loop.id == cmt::hwcopy::GetVariantId({cmt::Method::NtStore, Isa::Sve});
	CHECKF(sve == (want > 16), "nt_store runs %s with %zu-byte SVE", s ? text(s->loop.id).c_str() : "?",
		   want);
	std::printf("ok   expected cpu: SVE %zu bytes, nt_store runs %s\n", want,
				s ? text(s->loop.id).c_str() : "?");
}

} // namespace

int main(int argc, char** argv)
{
	const cmt::CpuIsas& c = cmt::hwcopy::GetCpu();
	std::printf("cpu:");
	for (Isa i : {Isa::Sse2, Isa::Avx2, Isa::Avx512, Isa::Neon, Isa::Sve})
	{
		if (c.Has(i))
		{
			std::printf(" %s", cmt::IsaName(i));
		}
	}
	std::printf(" (sve %zu bytes); write-back stride %zu bytes\n", c.sve_bytes,
				cmt::hwcopy::WriteBackLineBytes());
	test_expected_cpu(argc > 1 ? argv[1] : nullptr);
	test_list_and_dispatch();
	test_select();
	test_flush_batch_per_context();
	test_write_backs();
	for (VariantId id : all_ids())
	{
		test_variant(id);
	}
	check::run_all();
	return check::exit_code();
}
