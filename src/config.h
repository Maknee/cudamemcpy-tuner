#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/error.h"
#include "tune_table.h"

namespace cmt {

struct HookConfig
{
	bool valid = false;
	bool fixed_benchmark = false;
	std::optional<PinningSettings> benchmark_pinning;
	unsigned num_buffers = 0;
	int staging_buffer_kib = 0;
	uint64_t pipeline_chunk_bytes = kDefaultPipelineChunk;
	ProgressivePinning progressive_pinning = ProgressivePinning::Auto;
	uint64_t pinning_min_bytes = 4ull << 20;
	uint64_t pinning_max_bytes = 0;
	uint64_t pinning_step_bytes = 64ull << 20;
	uint64_t pinning_first_bytes = 4ull << 20;
	bool pinning_read_only = false;
	StagingAllocation slot_alloc = StagingAllocation::CachedPinned;
	unsigned pipeline_workers = 2;

	std::optional<std::string> impl;
	std::optional<std::string> impl_h2d;
	std::optional<std::string> impl_d2h;
	std::optional<cmt::Isa> copy_isa;

	std::vector<unsigned> pool_cpus;

	std::size_t copy_flush_bytes = CopyCtx{}.flush_batch;

	std::string tune_table_path;
	std::shared_ptr<const cmt::Table> tune_table;

	bool bench_dma = false;
	bool debug = false;
	bool no_eager = false;
	bool remote_config = false;
	bool cuda_host_alloc_hook = false;
	static constexpr unsigned cuda_host_alloc_threads = 8;
	bool cuda_host_alloc_private = false;

	bool allow_unverified_rows = false;

	bool WantH2HPatch() const;
	bool WantInternalMemcpyPatch() const;

	bool WantStagingSizeWrap() const
	{
		return !fixed_benchmark && tune_table_path.empty() && staging_buffer_kib > 0;
	}
};

struct PlacementInputs
{
	std::vector<unsigned> explicit_cpus;
	const WorkerPlacement* recorded = nullptr;
};

PlacementInputs HookPlacementInputs(const HookConfig& config);

using EnvLookup = Function<const char*(const char*) const>;

Result<HookConfig> ParseConfig(const EnvLookup& getenv_fn, const cmt::CpuIsas& cpu);
const char* PathVariable(const EnvLookup& getenv_fn, const char* name, bool secure);
Result<HookConfig> ParseConfig(const EnvLookup& getenv_fn);

const HookConfig& GetConfig();

Result<std::vector<unsigned>> ParseCpuList(const char* text);

void PrintConfig(const HookConfig& cfg, int fd);

} // namespace cmt
