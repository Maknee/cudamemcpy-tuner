#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/defines.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "config.h"
#include "copying/chunk_pipeline.h"
#include "measure/metrics.h"
#include "cuda/cuda.h"
#include "machine.h"

namespace cmt {

struct StagedBenchmarkOptions
{
	int device = 0;
	int cpu = -1;
	std::size_t threads = 1;
	std::size_t slots = 4;
	std::size_t slot_bytes = 1u << 20;
	LoopName cpu_loop;
	bool cpu_copy = true;
	bool wc_slots = false;
	bool dma = true;
	std::vector<unsigned> worker_allowed_cpus{};
};

class StagedCopyBenchmark final
{
public:
	StagedCopyBenchmark();
	~StagedCopyBenchmark();
	StagedCopyBenchmark(const StagedCopyBenchmark&) = delete;
	StagedCopyBenchmark& operator=(const StagedCopyBenchmark&) = delete;

	[[nodiscard]] Result<void> Setup(const StagedBenchmarkOptions&);
	[[nodiscard]] Result<double> Copy(Dir, unsigned char* host, unsigned char* device, std::size_t bytes,
									  std::uint32_t reps);
	[[nodiscard]] Result<void> Finish();
	void Teardown() noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

Result<std::optional<DeviceInfo>> PrepareBenchmark(Config& config);

Result<std::vector<std::size_t>> SizesThatFit(const Config&, const PathSpec&,
											  SizeCoverage* coverage = nullptr);

Result<void> Run(const Config& cfg, const DeviceInfo& dev,
				 const Function<Result<void>(const MeasurementRow&) const>& emit);

Result<int> RunTuneProbe(const std::string& request_path, const std::string& result_path);

Result<int> RunTuneHostAlloc(const std::string& request_path, const std::string& result_path);

} // namespace cmt
