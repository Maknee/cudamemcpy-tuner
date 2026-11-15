// test_config.cpp - environment parsing rules, install gates, impl
// selection, CPU lists.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <optional>
#include <map>
#include <memory>
#include <string>

#include "check.h"
#include "numa_fixture.h"
#include "copying/chunk_pipeline.h"
#include "config.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "arch/copy.h"
#include "copying/cpu_copy.h"
#include "cuda/driver.h"

namespace {

using Env = std::map<std::string, std::string>;

// Preserve parser errors. Positive fixtures require success before inspecting their settings.
cmt::Result<cmt::HookConfig> Parse(const Env& env)
{
	return cmt::ParseConfig(
		[&env](const char* name) -> const char*
		{
			auto it = env.find(name);
			return it == std::end(env) ? nullptr : it->second.c_str();
		});
}

cmt::HookConfig parse(const Env& env)
{
	return check::Require(Parse(env));
}

// The same on a pretend CPU (STAGING_COPY_ISA is checked against it).
cmt::Result<cmt::HookConfig> parse_on(const Env& env, const cmt::CpuIsas& cpu)
{
	return cmt::ParseConfig(
		[&env](const char* name) -> const char*
		{
			auto it = env.find(name);
			return it == std::end(env) ? nullptr : it->second.c_str();
		},
		cpu);
}

// The id a name selects on this CPU (the default loop when it falls back), for comparisons.
cmt::VariantId id_of(const char* name)
{
	const auto s = cmt::hwcopy::Select(name, {}, std::nullopt, cmt::hwcopy::GetCpu());
	return s ? s->loop.id : cmt::hwcopy::DefaultLoop({}, std::nullopt, cmt::hwcopy::GetCpu()).id;
}

// Loop names of this architecture for the table and selection checks (arch/copy.h).
#if defined(__x86_64__)
constexpr const char* kLoopA = "movsb";			  // a Base loop
constexpr const char* kForeign = "nt_store@neon"; // the other architecture's loop
#else
constexpr const char* kLoopA = "cached@neon";
constexpr const char* kForeign = "movsb";
#endif

void test_defaults()
{
	const cmt::HookConfig c = parse({});
	CHECK(c.num_buffers == 0);
	CHECK(c.staging_buffer_kib == 0);
	CHECK(c.pipeline_chunk_bytes == 4096ull * 1024);
	CHECK(!c.impl && !c.impl_h2d && !c.impl_d2h);
	CHECK(c.pool_cpus.empty());
	CHECK(!c.bench_dma && !c.debug && !c.no_eager);
	CHECK(!c.pinning_read_only);
	CHECK(!c.WantH2HPatch() && !c.WantInternalMemcpyPatch() && !c.WantStagingSizeWrap());
}

void test_numbers()
{
	CHECK(parse({{"STAGING_THREADS_COPY_NUM_BUFFERS", "0"}}).num_buffers == 0);
	CHECK(parse({{"STAGING_THREADS_COPY_NUM_BUFFERS", "-3"}}).num_buffers == 0);
	const auto invalid_buffers = Parse({{"STAGING_THREADS_COPY_NUM_BUFFERS", "abc"}});
	CHECK(!invalid_buffers && invalid_buffers.error().IsUsage());
	CHECK(parse({{"STAGING_THREADS_COPY_NUM_BUFFERS", "4"}}).num_buffers == 4);
	CHECK(parse({{"STAGING_THREADS_COPY_NUM_BUFFERS", "99"}}).num_buffers == 32);

	const cmt::HookConfig s = parse({{"STAGING_BUFFER_SIZE", "16384"}});
	CHECK(s.staging_buffer_kib == 16384 && s.pipeline_chunk_bytes == 16384ull * 1024);
	CHECK(s.WantStagingSizeWrap());
	const cmt::HookConfig neg = parse({{"STAGING_BUFFER_SIZE", "-5"}});
	CHECK(!neg.WantStagingSizeWrap() && neg.pipeline_chunk_bytes == 4096ull * 1024);


}

void test_flags()
{
	CHECK(parse({{"CMT_PINNING_READ_ONLY", "1"}}).pinning_read_only);
	CHECK(!parse({{"CMT_PINNING_READ_ONLY", "0"}}).pinning_read_only);
	CHECK(!Parse({{"CMT_PINNING_READ_ONLY", "garbage"}}));
	CHECK(!Parse({{"CMT_PINNING_READ_ONLY", "-1"}}));
	CHECK(parse({{"STAGING_BENCH_DMA", "1"}}).bench_dma);
	CHECK(parse({{"MEMCPY_HOOK_DEBUG", "1"}}).debug);
	CHECK(parse({{"MEMCPY_HOOK_NO_EAGER", "1"}}).no_eager);
	// Unverified table rows (aarch64): exactly "1" opts in.
	CHECK(!parse({}).allow_unverified_rows);
	CHECK(parse({{"MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS", "1"}}).allow_unverified_rows);
	CHECK(!parse({{"MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS", "0"}}).allow_unverified_rows);
	const auto invalid_rows = Parse({{"MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS", "yes"}});
	CHECK(!invalid_rows && invalid_rows.error().IsUsage());
}

void test_gates()
{
	for (const char* name : {"default", "quick", "bogus"})
	{
		const auto invalid = Parse({{"STAGING_COPY_NT_IMPL", name}});
		CHECK(!invalid && invalid.error().IsUsage());
	}
	CHECK(parse({{"STAGING_COPY_NT_IMPL", kLoopA}}).WantH2HPatch());
	CHECK(parse({{"STAGING_COPY_NT_IMPL", kLoopA}}).WantInternalMemcpyPatch());
	CHECK(parse({{"STAGING_COPY_NT_IMPL_H2D", "memcpy"}}).WantH2HPatch());
	CHECK(!parse({{"STAGING_COPY_NT_IMPL_H2D", "memcpy"}}).WantInternalMemcpyPatch());
	CHECK(parse({{"STAGING_COPY_NT_IMPL_D2H", "memcpy"}}).WantH2HPatch());
	for (const char* value : {"2", "1"})
	{
		const auto removed = Parse({{"STAGING_THREADS_COPY_NUM_THREADS", value}});
		CHECK(!removed && removed.error().IsUsage());
	}
	CHECK(parse({{"STAGING_BENCH_DMA", "1"}}).WantH2HPatch());
}

void test_impls()
{
	using cmt::VariantId;
	const VariantId def = cmt::hwcopy::DefaultLoop({}, std::nullopt, cmt::hwcopy::GetCpu()).id;
	cmt::HostCopyLoops s = cmt::SelectCopyLoops(parse({}));
	CHECK(s.global.id == def && !s.h2d && !s.d2h && s.global.ctx.flush_batch == 4096);
	s = cmt::SelectCopyLoops(parse({{"STAGING_COPY_NT_IMPL", kLoopA},
									{"STAGING_COPY_NT_IMPL_H2D", "memcpy"},
									{"STAGING_COPY_FLUSH_BYTES", "1000"}}));
	CHECK(s.global.id == id_of(kLoopA) && s.h2d && s.h2d->id == VariantId::Memcpy && !s.d2h);
	CHECK(s.global.ctx.flush_batch == 960 && s.h2d->ctx.flush_batch == 960); // the context comes from the env
	const auto invalid =
		Parse({{"STAGING_COPY_NT_IMPL", "nonsense"}, {"STAGING_COPY_NT_IMPL_H2D", "default"}});
	CHECK(!invalid && invalid.error().IsUsage());
	// A loop from another architecture is rejected before selection.
	const auto foreign = Parse({{"STAGING_COPY_NT_IMPL", kForeign}, {"STAGING_COPY_NT_IMPL_D2H", kForeign}});
	CHECK(!foreign && foreign.error().IsUsage());
	// A write-back name carries its write-back into the context.
	s = cmt::SelectCopyLoops(
		parse({{"STAGING_COPY_NT_IMPL_H2D", "memcpy+flush"}, {"STAGING_COPY_FLUSH_BYTES", "512"}}));
	CHECK(s.h2d && s.h2d->id == VariantId::Memcpy && s.h2d->ctx.write_back == cmt::WriteBack::Flush &&
		  s.h2d->ctx.flush_batch == 512);
#if defined(__x86_64__)
	const auto avx512 =
		Parse({{"STAGING_COPY_NT_IMPL", "nt_store@avx512"}, {"STAGING_COPY_NT_IMPL_D2H", "nt_load@avx512"}});
	if (cmt::hwcopy::GetCpu().Has(cmt::Isa::Avx512))
	{
		s = cmt::SelectCopyLoops(check::Require(avx512));
		CHECK(s.global.id == VariantId::NtStoreAvx512 && s.d2h && s.d2h->id == VariantId::NtLoadAvx512);
	}
	else
	{
		CHECK(!avx512 && avx512.error().IsUnsupported());
	}
	// STAGING_COPY_ISA=sse2 overrides an explicit @ISA as well as an unpinned name.
	s = cmt::SelectCopyLoops(parse({{"STAGING_COPY_ISA", "sse2"},
									{"STAGING_COPY_NT_IMPL", "nt_store@avx2"},
									{"STAGING_COPY_NT_IMPL_H2D", "cached"}}));
	CHECK(s.global.id == VariantId::NtStoreSse2 && s.h2d && s.h2d->id == VariantId::CachedSse2);
#endif
}

// Unknown and unsupported explicit ISAs return errors instead of removing the override.
void test_copy_isa()
{
#if defined(__x86_64__)
	cmt::CpuIsas no_avx2;
	no_avx2.Add(cmt::Isa::Base);
	no_avx2.Add(cmt::Isa::Sse2);
	cmt::CpuIsas avx2 = no_avx2;
	avx2.Add(cmt::Isa::Avx2);
	const auto selected = parse_on({{"STAGING_COPY_ISA", "avx2"}}, avx2);
	CHECK(selected && selected->copy_isa == cmt::Isa::Avx2);
	CHECK(!parse_on({{"STAGING_COPY_ISA", "avx2"}}, no_avx2)); // the CPU lacks it
	CHECK(!parse_on({{"STAGING_COPY_ISA", "avx512"}}, avx2));
	CHECK(!parse_on({{"STAGING_COPY_ISA", "neon"}}, avx2)); // another architecture's
#else
	cmt::CpuIsas neon;
	neon.Add(cmt::Isa::Base);
	neon.Add(cmt::Isa::Neon);
	cmt::CpuIsas sve = neon;
	sve.Add(cmt::Isa::Sve);
	sve.sve_bytes = 32;
	const auto selected = parse_on({{"STAGING_COPY_ISA", "sve"}}, sve);
	CHECK(selected && selected->copy_isa == cmt::Isa::Sve);
	CHECK(!parse_on({{"STAGING_COPY_ISA", "sve"}}, neon));
	CHECK(!parse_on({{"STAGING_COPY_ISA", "avx2"}}, sve));
#endif
	const auto invalid = Parse({{"STAGING_COPY_ISA", "warp9"}});
	CHECK(!invalid && invalid.error().IsUsage());
	CHECK(!parse({}).copy_isa);
}

void test_new_knobs()
{
	CHECK(parse({}).copy_flush_bytes == 4096);
	CHECK(parse({{"STAGING_COPY_FLUSH_BYTES", "0"}}).copy_flush_bytes == 0);
	CHECK(parse({{"STAGING_COPY_FLUSH_BYTES", "-5"}}).copy_flush_bytes == 0);
	CHECK(parse({{"STAGING_COPY_FLUSH_BYTES", "32"}}).copy_flush_bytes == 64);
	CHECK(parse({{"STAGING_COPY_FLUSH_BYTES", "8191"}}).copy_flush_bytes == 8128);
	CHECK(parse({{"STAGING_COPY_FLUSH_BYTES", "65536"}}).copy_flush_bytes == 65536);
}

void TestProgressivePinningPolicy()
{
	using cmt::ProgressivePinning;
	CHECK(parse({}).progressive_pinning == ProgressivePinning::Auto);
	CHECK(parse({}).valid);
	CHECK(!cmt::HookConfig{}.valid);
	FILE* request = std::tmpfile();
	if (!request)
	{
		std::abort();
	}
	CHECK(std::fputs(R"({"schema":"cmt.benchmark-run","version":2,"direction":"htod",)"
					 R"("sizes_bytes":[4096,4194304],"process_round":1,"settings":{"method":"driver"}})",
					 request) >= 0);
	CHECK(std::fflush(request) == 0);
	const std::string request_path = "/proc/self/fd/" + std::to_string(fileno(request));
	const auto fixed = parse({{"CMT_EXPECTED_RUN", request_path}});
	CHECK(fixed.valid && fixed.fixed_benchmark && fixed.progressive_pinning == ProgressivePinning::Off);
	CHECK(!Parse({{"CMT_EXPECTED_RUN", request_path}, {"CMT_OVERRIDE", "progressive_pinning=on"}}));
	CHECK(std::fclose(request) == 0);
	for (const auto& [name, expected] :
		 {std::pair{"auto", ProgressivePinning::Auto}, std::pair{"on", ProgressivePinning::On},
		  std::pair{"off", ProgressivePinning::Off}})
	{
		const std::string setting = std::string("progressive_pinning=") + name;
		const auto config = parse({{"CMT_OVERRIDE", setting}});
		CHECK(config.progressive_pinning == expected && config.valid);
		const auto overrides = cmt::ParseRunOverrides(setting);
		CHECK(overrides && overrides->progressive_pinning == expected && !overrides->HasStaging());
	}
	for (const char* invalid : {"", "0", "1", "2", "On", "retain", "typo"})
	{
		const auto config = Parse({{"CMT_OVERRIDE", std::string("progressive_pinning=") + invalid}});
		CHECK(!config && config.error().IsUsage());
	}
}

// The run-time registration limit (HookConfig::pinning_max_bytes): unset by default, so each copy may pin up to
// its GPU's memory (progressive_pinning.cpp, RegistrationBudget); CMT_PINNING_MAX_MIB sets it, and
// `--override pinning_max_mib=N` (CMT_OVERRIDE) replaces it. A fixed measured benchmark refuses the override.
void TestPinningLimit()
{
	constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
	CHECK(parse({}).pinning_max_bytes == 0);
	CHECK(parse({{"CMT_PINNING_MAX_MIB", "1024"}}).pinning_max_bytes == 1024 * kMiB);
	CHECK(parse({{"CMT_PINNING_MAX_MIB", "0"}}).pinning_max_bytes == 0);
	CHECK(parse({{"CMT_OVERRIDE", "pinning_max_mib=48"}}).pinning_max_bytes == 48 * kMiB);
	CHECK(parse({{"CMT_PINNING_MAX_MIB", "1024"}, {"CMT_OVERRIDE", "pinning_max_mib=48"}}).pinning_max_bytes ==
		  48 * kMiB);
	const auto overrides = check::Require(cmt::ParseRunOverrides("pinning_max_mib=16384"));
	CHECK(overrides.pinning_max_bytes == 16384 * kMiB && !overrides.HasStaging() && !overrides.Empty());
	CHECK(cmt::FormatRunOverrides(overrides) == "pinning_max_mib=16384");
	CHECK(check::Require(cmt::ParseRunOverrides(cmt::FormatRunOverrides(overrides))).pinning_max_bytes ==
		  overrides.pinning_max_bytes);
	CHECK(check::Require(cmt::ParseRunOverrides("pinning_max_mib=1048576")).pinning_max_bytes == kMiB * kMiB);
	for (const char* invalid : {"0", "-1", "1G", "1.5", " 1", "1048577", "18446744073709551616"})
	{
		const auto config = Parse({{"CMT_OVERRIDE", std::string("pinning_max_mib=") + invalid}});
		CHECK(!config && config.error().IsUsage());
	}
	CHECK(!cmt::ParseRunOverrides("pinning_max_mib=1,pinning_max_mib=2"));
}

void TestRemovedControls()
{
	for (const char* key :
		 {"STAGING_THREADS_COPY_NUM_THREADS", "STAGING_THREADS_COPY_MPMC", "STAGING_THREADS_COPY_POOL_H2D",
		  "STAGING_THREADS_COPY_POOL_D2H", "STAGING_THREADS_COPY_SPIN_US", "STAGING_AUTO_CPUS",
		  "STAGING_AUTO_CPUS_LOG", "STAGING_PIPELINE_STREAMS"})
	{
		for (const char* value : {"", "0", "1", "2", "100", "off"})
		{
			const auto config = Parse({{key, value},
									   {"STAGING_THREADS_COPY_NUM_BUFFERS", "4"},
									   {"STAGING_COPY_NT_IMPL", "memcpy"},
									   {"STAGING_BUFFER_SIZE", "64"}});
			CHECK(!config && config.error().IsUsage());
		}
	}
}

void test_pipeline_modes()
{
	cmt::HookConfig c = parse({});
	CHECK(c.pipeline_workers == 2);
	CHECK(parse({{"STAGING_PIPELINE_WORKERS", "0"}}).pipeline_workers == 1);
	CHECK(parse({{"STAGING_PIPELINE_WORKERS", "99"}}).pipeline_workers == 16);

}

// An empty variable is unset, as CMT_CONFIG= always was: `CMT_CUDA_HOST_ALLOC_HOOK=` must not disable the whole
// configuration. Variables naming files are ignored in an AT_SECURE process.
void test_empty_and_secure_variables()
{
	const auto empty = parse({{"CMT_CUDA_HOST_ALLOC_HOOK", ""},
							  {"STAGING_THREADS_COPY_NUM_BUFFERS", ""},
							  {"MEMCPY_HOOK_DEBUG", ""},
							  {"CMT_EXPECTED_RUN", ""},
							  {"CMT_CONFIG", ""}});
	const auto defaults = parse({});
	CHECK(empty.valid && empty.cuda_host_alloc_hook == defaults.cuda_host_alloc_hook &&
		  empty.cuda_host_alloc_threads == defaults.cuda_host_alloc_threads &&
		  empty.num_buffers == defaults.num_buffers && !empty.fixed_benchmark && empty.tune_table_path.empty());
	CHECK(parse({{"CMT_CUDA_HOST_ALLOC_HOOK", "1"}}).cuda_host_alloc_hook);
	const auto invalid = Parse({{"CMT_CUDA_HOST_ALLOC_HOOK", "yes"}});
	CHECK(!invalid && invalid.error().IsUsage());

	const Env env{{"CMT_CONFIG", "/etc/shadow"}, {"CMT_EXPECTED_RUN", ""}};
	const cmt::EnvLookup lookup = [&env](const char* name) -> const char*
	{
		auto it = env.find(name);
		return it == std::end(env) ? nullptr : it->second.c_str();
	};
	CHECK(std::string(cmt::PathVariable(lookup, "CMT_CONFIG", false)) == "/etc/shadow");
	CHECK(cmt::PathVariable(lookup, "CMT_CONFIG", true) == nullptr);
	CHECK(cmt::PathVariable(lookup, "CMT_EXPECTED_RUN", false) == nullptr);
	CHECK(cmt::PathVariable(lookup, "CMT_UNSET", false) == nullptr);
	CHECK(!cmt::SecureExecution()); // the test runs unprivileged: ParseConfig reads them
}

void test_cpu_list()
{
	const std::pair<const char*, std::vector<unsigned>> valid[] = {
		{nullptr, {}},		{"", {}},
		{"3", {3}},			{"0,2,4-6", {0, 2, 4, 5, 6}},
		{" 8 , 9", {8, 9}}, {"3,1-2,3", {3, 1, 2, 3}},
		{"4096", {4096}}};
	for (const auto& [text, expected] : valid)
	{
		const auto cpus = cmt::ParseCpuList(text);
		CHECK(cpus && *cpus == expected);
	}
	for (const char* text : {" 8 , x, 9", "3,", ",3", "1,,2", "3-1", "-3", "1 2", "3x", "4294967296",
							 "0-4294967295", "-9223372036854775808-0"})
	{
		const auto cpus = cmt::ParseCpuList(text);
		CHECK(!cpus && cpus.error().IsUsage());
	}
	CHECK((parse({{"STAGING_THREADS_COPY_CPUS", "1,3"}}).pool_cpus == std::vector<unsigned>{1, 3}));
	const auto invalid =
		Parse({{"STAGING_THREADS_COPY_CPUS", "1,x"}, {"STAGING_THREADS_COPY_NUM_BUFFERS", "4"}});
	CHECK(!invalid && invalid.error().IsUsage());
}

// Writes `text` to a file in the working directory (ctest runs in the build tree) and returns its name.
std::string write_file(const char* name, const std::string& text)
{
	std::FILE* f = std::fopen(name, "w");
	CHECK(f != nullptr);
	if (f != nullptr)
	{
		std::fputs(text.c_str(), f);
		std::fclose(f);
	}
	return name;
}

// File-admission policy selection is covered by the canonical JSON policy test.
// Malformed input must reject the whole configuration, including any explicit
// environment staging or binary patch request.
void test_tune_table()
{
	auto config = parse({});
	CHECK(!config.tune_table && config.tune_table_path.empty());
	CHECK(!config.tune_table || !config.tune_table->HasStagedCopies());
	config = parse({{"CMT_CONFIG", ""}});
	CHECK(!config.tune_table && config.tune_table_path.empty());
	for (const std::string& text : {std::string{}, std::string{"{}"}, std::string{"not json"},
									std::string{"cmt-tune-table 1\nrow htod 0 inf driver\n"}})
	{
		const std::string path = write_file("test_config_invalid.json", text);
		CHECK(!cmt::tuning::Load(path.c_str()));
		const auto invalid = Parse({{"CMT_CONFIG", path},
									{"STAGING_THREADS_COPY_NUM_BUFFERS", "4"},
									{"STAGING_COPY_NT_IMPL", kLoopA}});
		CHECK(!invalid && invalid.error().IsUsage());
		std::remove(path.c_str());
	}
	CHECK(!cmt::tuning::Load("test_config_missing.json"));
	CHECK(!cmt::tuning::Load("."));

	// Direct loop selection retains explicit flush configuration.
	cmt::CopyCtx row;
	row.flush_batch = 8192;
	const auto selection = cmt::hwcopy::Select("memcpy+flush", row, std::nullopt, cmt::hwcopy::GetCpu());
	CHECK(selection && selection->loop.id == cmt::VariantId::Memcpy &&
		  selection->loop.ctx.write_back == cmt::WriteBack::Flush && selection->loop.ctx.flush_batch == 8192);
}

} // namespace

// The chunks pipeline's fit rule (copying/chunk_pipeline.h): a copy's layout, workers x slots x the chunk
// rounded up to 4 KiB, must fit the one pinned pool, and its workers and slots the threads and events.
// A copy that does not fit goes to the driver; this is the predicate (SettingsFit) that decides it.
void test_knobs_fit()
{
	// Rows htod w2 n4 2M (16 MiB) and dtoh w3 n2 16M (96 MiB): the pool is 96 MiB, not 3 x 4 x 16M.
	cmt::ring::PipelineCapacity shape;
	shape.workers = 3;
	shape.slots = 4;
	shape.pool_bytes = std::size_t{96} << 20;
	auto knobs = [](unsigned w, uint32_t n, uint64_t chunk)
	{
		cmt::CopySettings k;
		k.workers = w;
		k.slots = n;
		k.chunk_bytes = chunk;
		k.loop = cmt::hwcopy::DefaultLoop({}, std::nullopt, cmt::hwcopy::GetCpu());
		return k;
	};
	CHECK(cmt::ring::SettingsFit(shape, knobs(2, 4, std::size_t{2} << 20)));
	CHECK(cmt::ring::SettingsFit(shape, knobs(3, 2, std::size_t{16} << 20)));
	CHECK(!cmt::ring::SettingsFit(shape, knobs(3, 4, std::size_t{16} << 20))); // the product of the maxima
	CHECK(!cmt::ring::SettingsFit(shape, knobs(4, 1, 4096)));				   // more workers than threads
	CHECK(!cmt::ring::SettingsFit(shape, knobs(1, 5, 4096)));				   // more slots than events
	CHECK(!cmt::ring::SettingsFit(shape, knobs(0, 1, 4096)));
	CHECK(!cmt::ring::SettingsFit(shape, knobs(1, 1, 0)));
	// The stride: a chunk that is not a multiple of 4 KiB takes the next multiple in the pool.
	CHECK(cmt::ring::SlotStride(100 * 1024) == 100 * 1024);
	CHECK(cmt::ring::SlotStride(1) == 4096 && cmt::ring::SlotStride(4097) == 8192);
	shape.pool_bytes = 3 * 8192;
	CHECK(cmt::ring::SettingsFit(shape, knobs(3, 1, 8192)));
	CHECK(cmt::ring::SettingsFit(shape, knobs(3, 1, 4097)));  // 3 x 8192 fits exactly
	CHECK(!cmt::ring::SettingsFit(shape, knobs(3, 1, 8193))); // 3 x 12288 does not
}

// STAGING_SLOT_ALLOC accepts WC and keeps cached as the default. An unknown value
// returns a parsing error. host_alloc must pass the flag through; with the two-argument
// allocator it must refuse a flag instead of handing out cached memory under a WC label.
namespace legacy {
unsigned new_calls = 0, old_calls = 0, last_flags = 0xFFFF;

cmt::cu::CUresult new_alloc(void** p, std::size_t, unsigned flags)
{
	++new_calls;
	last_flags = flags;
	*p = reinterpret_cast<void*>(0x1000);
	return cmt::cu::kSuccess;
}

cmt::cu::CUresult old_alloc(void** p, std::size_t)
{
	++old_calls;
	*p = reinterpret_cast<void*>(0x2000);
	return cmt::cu::kSuccess;
}
} // namespace legacy

void test_slot_alloc()
{
	CHECK(parse({}).slot_alloc == cmt::StagingAllocation::CachedPinned);
	CHECK(parse({{"STAGING_SLOT_ALLOC", "hostalloc"}}).slot_alloc == cmt::StagingAllocation::CachedPinned);
	CHECK(parse({{"STAGING_SLOT_ALLOC", "mmap_4k"}}).slot_alloc == cmt::StagingAllocation::Registered4K);
	CHECK(!Parse({{"STAGING_SLOT_ALLOC", "mmap_thp"}}));
	CHECK(parse({{"STAGING_SLOT_ALLOC", "hostalloc_wc"}}).slot_alloc ==
		  cmt::StagingAllocation::WriteCombinedPinned);
	const auto invalid = Parse({{"STAGING_SLOT_ALLOC", "wc"}});
	CHECK(!invalid && invalid.error().IsUsage());

	using namespace cmt::cu;
	void* p = nullptr;
	DriverApi both;
	both.cuMemHostAlloc = legacy::new_alloc;
	both.cuMemAllocHost = legacy::old_alloc;
	CHECK(HostAlloc(both, &p, 4096, kMemHostAllocWriteCombined) == kSuccess);
	CHECK(legacy::new_calls == 1 && legacy::old_calls == 0 &&
		  legacy::last_flags == kMemHostAllocWriteCombined);

	DriverApi old_only;
	old_only.cuMemAllocHost = legacy::old_alloc;
	p = nullptr;
	CHECK(HostAlloc(old_only, &p, 4096, kMemHostAllocWriteCombined) == kErrorNotSupported);
	CHECK(legacy::old_calls == 0 && p == nullptr); // refused before any allocation
	CHECK(HostAlloc(old_only, &p, 4096, 0) == kSuccess);
	CHECK(legacy::old_calls == 1 && p == reinterpret_cast<void*>(0x2000));

	CHECK(HostAlloc(DriverApi{}, &p, 4096, 0) == kErrorNotFound);
}

namespace {

struct WorkerEnvironment
{
	const char* names[3] = {"CMT_WORKER_NUMA", "CMT_WORKER_CPUS", "CMT_WORKER_NUMA_NODE"};
	std::optional<std::string> saved[3];

	WorkerEnvironment()
	{
		for (unsigned i = 0; i < 3; ++i)
		{
			if (const char* value = std::getenv(names[i]))
			{
				saved[i] = value;
			}
			CHECK(unsetenv(names[i]) == 0);
		}
	}

	~WorkerEnvironment()
	{
		for (unsigned i = 0; i < 3; ++i)
		{
			CHECK((saved[i] ? setenv(names[i], saved[i]->c_str(), 1) : unsetenv(names[i])) == 0);
		}
	}
};

void TestWorkerDevicePlacement()
{
	using namespace cmt::cu;
	WorkerEnvironment environment;
	static unsigned queries = 0;
	static CUcontext current = reinterpret_cast<CUcontext>(0x7000);
	static CUdevice queried_device = -1;
	static CUresult pci_status = kSuccess;
	DriverApi api;
	api.cuCtxGetCurrent = [](CUcontext* context)
	{
		++queries;
		*context = current;
		return kSuccess;
	};
	api.cuCtxGetDevice = [](CUdevice* device)
	{
		++queries;
		*device = 7;
		return kSuccess;
	};
	api.cuDeviceGetPCIBusId = [](char* bus, int length, CUdevice device)
	{
		++queries;
		queried_device = device;
		std::snprintf(bus, static_cast<std::size_t>(length), "ffffffff:ff:1f.7");
		return pci_status;
	};

	// Disabled means no CUDA query, including on systems without libcuda (the default on one node is the
	// same: test_sync, a_single_node_host_is_unchanged_and_asks_cuda_nothing).
	CHECK(setenv("CMT_WORKER_NUMA", "0", 1) == 0);
	auto disabled = CurrentWorkerPlacement(api);
	CHECK(disabled && disabled->cpus.empty() && !disabled->numa_node);
	CHECK(queries == 0);
	// The default asks CUDA only on a host with two or more CPU nodes: the fixture's, and no NVML.
	CHECK(unsetenv("CMT_WORKER_NUMA") == 0);
	const check::TwoNodeSysfs sysfs(check::Require(cmt::GetThreadAffinity()));
	const auto place = [&](const DriverApi& driver, CUcontext expected = nullptr)
	{ return CurrentWorkerPlacement(driver, {}, expected, nullptr, sysfs.root, {}); };
	auto unavailable = place(DriverApi{});
	CHECK(unavailable && unavailable->cpus.empty() && !unavailable->numa_node);
	current = nullptr;
	auto no_context = place(api);
	CHECK(no_context && no_context->cpus.empty() && !no_context->numa_node);
	CHECK(queries == 1);
	current = reinterpret_cast<CUcontext>(0x7000);
	auto unknown_node = place(api); // ffffffff:ff:1f.7 is not in the fixture
	CHECK(unknown_node && !unknown_node->numa_node && unknown_node->cpus.empty());
	CHECK(queried_device == 7); // Never silently select ordinal zero.
	pci_status = 123;
	auto failed = place(api);
	CHECK(!failed);
	if (!failed)
	{
		CHECK(failed.error().GetKind() == cmt::Error::Kind::Cuda);
		CHECK(failed.error().GetCode() == pci_status);
	}
	auto wrong_context = place(api, reinterpret_cast<CUcontext>(0x8000));
	CHECK(!wrong_context && wrong_context.error().IsUsage());

	// Explicit lists override automatic topology lookup, even if that lookup would fail.
	const auto before = queries;
	CHECK(setenv("CMT_WORKER_CPUS", "2", 1) == 0);
	auto mask = CurrentWorkerPlacement(api);
	CHECK(mask && mask->cpus == std::vector<unsigned>{2} && !mask->numa_node);
	auto specific = CurrentWorkerPlacement(api, {3});
	CHECK(specific && specific->cpus == std::vector<unsigned>{3} && !specific->numa_node);
	CHECK(queries == before);
	CHECK(unsetenv("CMT_WORKER_CPUS") == 0);
	CHECK(setenv("CMT_WORKER_NUMA", "yes", 1) == 0);
	auto invalid = CurrentWorkerPlacement(api);
	CHECK(!invalid && invalid.error().IsUsage());

	// Under a table only its record counts: a recorded placement, or none, which is the default (unplaced on
	// this fake GPU with an unknown node), never the application's CMT_WORKER_*.
	CHECK(setenv("CMT_WORKER_NUMA", "0", 1) == 0);
	CHECK(setenv("CMT_WORKER_CPUS", "5", 1) == 0);
	cmt::WorkerPlacement recorded;
	recorded.cpus = {2};
	auto table = CurrentWorkerPlacement(api, {}, nullptr, &recorded);
	CHECK(table && table->cpus == std::vector<unsigned>{2});
	const cmt::WorkerPlacement none;
	auto unrecorded = CurrentWorkerPlacement(api, {}, nullptr, &none);
	CHECK(unrecorded && unrecorded->cpus.empty() && !unrecorded->numa_node);
	auto from_environment = CurrentWorkerPlacement(api); // no table: the environment
	CHECK(from_environment && from_environment->cpus == std::vector<unsigned>{5});
}

using namespace cmt::cu;

// A driver just able to create a pipeline; cuMemHostAlloc records the thread it ran on, with its affinity
// and current context.
struct Allocation
{
	pthread_t thread{};
	std::vector<unsigned> affinity;
	CUcontext context = nullptr;
	unsigned flags = 0;
};

thread_local CUcontext t_current = nullptr;
std::vector<Allocation> g_allocations; // written by the allocating thread before the caller joins it

DriverApi PipelineApi()
{
	DriverApi api;
	const auto ok = [](auto...) { return kSuccess; };
	api.cuCtxGetCurrent = [](CUcontext* context)
	{
		*context = t_current;
		return kSuccess;
	};
	api.cuCtxSetCurrent = [](CUcontext context)
	{
		t_current = context;
		return kSuccess;
	};
	api.cuEventCreate = [](CUevent* event, unsigned)
	{
		*event = reinterpret_cast<CUevent>(0x10);
		return kSuccess;
	};
	api.cuStreamCreate = [](CUstream* stream, unsigned)
	{
		*stream = reinterpret_cast<CUstream>(0x20);
		return kSuccess;
	};
	api.cuMemHostAlloc = [](void** p, std::size_t bytes, unsigned flags)
	{
		g_allocations.push_back({pthread_self(), check::Require(cmt::GetThreadAffinity()), t_current, flags});
		*p = std::aligned_alloc(4096, bytes);
		return kSuccess;
	};
	api.cuMemFreeHost = [](void* p)
	{
		std::free(p);
		return kSuccess;
	};
	api.cuEventRecord = ok;
	api.cuEventSynchronize = ok;
	api.cuEventDestroy = ok;
	api.cuStreamWaitEvent = ok;
	api.cuStreamDestroy = ok;
	api.cuStreamSynchronize = ok;
	api.cuMemcpyHtoDAsync_v2 = ok;
	api.cuMemcpyDtoHAsync_v2 = ok;
	return api;
}

// The CPU masks of this process's threads, by thread ID.
std::map<std::string, std::vector<unsigned>> ThreadMasks()
{
	std::map<std::string, std::vector<unsigned>> masks; // sorted: compared as a whole below
	for (const auto& task : std::filesystem::directory_iterator("/proc/self/task"))
	{
		const auto tid = task.path().filename().string();
		const auto status = cmt::ReadTextFile(task.path().string() + "/status");
		const auto at = status ? status->find("Cpus_allowed_list:\t") : std::string::npos;
		if (at != std::string::npos)
		{
			const auto start = at + std::strlen("Cpus_allowed_list:\t");
			masks[tid] = check::Require(
				cmt::ParseThreadCpus(status->substr(start, status->find('\n', start) - start)));
		}
	}
	return masks;
}

// `--numa` reaches the hook as the table's recorded placement (tuning::Load, test_tune_table_e2e). On a
// two-node host the default puts the helpers and the pool on the GPU's node; off leaves them on the caller's
// CPUs. Both through the hook's own path: CurrentWorkerPlacement, then ChunkPipeline::Create and Start. Pages
// land on the node of the thread that faults them in, so a placed pool is allocated by another thread, with
// the placement and the pipeline's context current; an unplaced one by the caller.
void TestNumaSwitchPlacesOrLeavesThreads()
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  numa switch: needs two allowed CPUs; skipped\n");
		return;
	}
	check::TwoNodeSysfs sysfs(allowed);
	const DriverApi saved = g_api;
	g_api = PipelineApi();
	g_api.cuCtxGetDevice = [](CUdevice* device)
	{
		*device = 0;
		return kSuccess;
	};
	g_api.cuDeviceGetPCIBusId = [](char* bus, int length, CUdevice)
	{
		std::snprintf(bus, static_cast<std::size_t>(length), "%s", check::kFixtureGpu);
		return kSuccess;
	};
	const auto context = reinterpret_cast<CUcontext>(0x7000);
	t_current = context;
	for (const auto numa : {std::optional<bool>{}, std::optional<bool>{false}, std::optional<bool>{true}})
	{
		cmt::WorkerPlacement recorded;
		recorded.gpu_numa = numa;
		const auto placement =
			check::Require(CurrentWorkerPlacement(g_api, {}, context, &recorded, sysfs.root, {}));
		g_allocations.clear();
		cmt::HookConfig config;
		config.slot_alloc = numa == true ? cmt::StagingAllocation::WriteCombinedPinned
										 : cmt::StagingAllocation::CachedPinned; // both allocators
		auto pipeline =
			check::Require(cmt::ring::ChunkPipeline::Create(context, config, {3, 1, 4096}, placement));
		const auto before = ThreadMasks();
		CHECK(pipeline->Start(placement));
		std::vector<std::vector<unsigned>> helpers;
		for (const auto& [tid, mask] : ThreadMasks())
		{
			if (before.count(tid) == 0)
			{
				helpers.push_back(mask);
			}
		}
		const auto& expected = numa == false ? allowed : sysfs.node1;
		CHECK(helpers == (std::vector<std::vector<unsigned>>{expected, expected}));
		CHECK_EQ(std::size(g_allocations), std::size_t{1});
		if (std::size(g_allocations) == 1)
		{
			const auto& pool = g_allocations[0];
			CHECK(pool.affinity == expected && pool.context == context);
			CHECK((pthread_equal(pool.thread, pthread_self()) != 0) == (numa == false));
			CHECK_EQ(pool.flags, numa == true ? kMemHostAllocWriteCombined : 0u);
		}
	}
	CHECK(check::Require(cmt::GetThreadAffinity()) == allowed);
	g_api = saved;
}

// The staging helpers and the pinning reaper take their placement from one function: under a table only its
// record, never STAGING_THREADS_COPY_CPUS or CMT_WORKER_*. The reaper is never GPU-local, even where the
// helpers would be.
void TestReaperFollowsTheTable()
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  reaper placement: needs two allowed CPUs; skipped\n");
		return;
	}
	check::TwoNodeSysfs sysfs(allowed);
	const DriverApi saved = g_api;
	g_api = PipelineApi();
	g_api.cuCtxGetDevice = [](CUdevice* device)
	{
		*device = 0;
		return kSuccess;
	};
	g_api.cuDeviceGetPCIBusId = [](char* bus, int length, CUdevice)
	{
		std::snprintf(bus, static_cast<std::size_t>(length), "%s", check::kFixtureGpu);
		return kSuccess;
	};
	t_current = reinterpret_cast<CUcontext>(0x7000);
	CHECK(setenv("CMT_WORKER_CPUS", std::to_string(allowed.back()).c_str(), 1) == 0);
	cmt::HookConfig config;
	config.pool_cpus = {allowed.back()};
	auto table = std::make_shared<cmt::Table>();
	table->worker_placement.cpus = {allowed.front()};
	config.tune_table = table;
	const auto inputs = cmt::HookPlacementInputs(config);
	CHECK(inputs.recorded == &table->worker_placement && inputs.explicit_cpus.empty());
	const auto reaper =
		check::Require(BackgroundPlacement(inputs.explicit_cpus, inputs.recorded, sysfs.root));
	CHECK(reaper.cpus == std::vector<unsigned>{allowed.front()});
	// A table that recorded nothing: the helpers go to the GPU's node 1, the reaper stays unplaced.
	table->worker_placement = {};
	const auto helpers =
		check::Require(CurrentWorkerPlacement(g_api, {}, nullptr, &table->worker_placement, sysfs.root, {}));
	CHECK(helpers.cpus == sysfs.node1);
	const auto unplaced =
		check::Require(BackgroundPlacement(inputs.explicit_cpus, inputs.recorded, sysfs.root));
	CHECK(unplaced.cpus.empty() && !unplaced.numa_node);
	g_api = saved;
}

} // namespace

int main()
{
	test_defaults();
	test_numbers();
	test_flags();
	test_gates();
	test_impls();
	test_copy_isa();
	test_cpu_list();
	test_empty_and_secure_variables();
	test_new_knobs();
	test_pipeline_modes();
	TestRemovedControls();
	TestProgressivePinningPolicy();
	TestPinningLimit();
	test_tune_table();
	test_knobs_fit();
	test_slot_alloc();
	TestWorkerDevicePlacement();
	TestNumaSwitchPlacesOrLeavesThreads();
	TestReaperFollowsTheTable();
	std::fflush(stdout);
	cmt::PrintConfig(parse({{"STAGING_THREADS_COPY_CPUS", "0-3"}}), STDOUT_FILENO);
	return check::exit_code();
}
