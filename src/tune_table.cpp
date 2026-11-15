#include "cuda/kernel_status.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/defines.h"
#include "common/error.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/log.h"

#include "tune_table.h"
#include "share.h"

namespace cmt {
namespace {
constexpr unsigned kProgressivePinningAuditVersion = 3;
} // namespace

namespace tuning {

namespace {

cmt::Result<std::string> ReadTableFile(const char* path, bool first_line, std::uint64_t& file_bytes)
{
	std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path, "rb"), &std::fclose);
	if (!file)
	{
		const int code = errno;
		return cmt::Failure(cmt::Error::Kind::Os, "cannot open tuning table", code);
	}
	struct stat status{};
	if (fstat(fileno(file.get()), &status) != 0)
	{
		return OsError("inspect tuning configuration");
	}
	if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
		static_cast<std::uint64_t>(status.st_size) > kMaxConfigBytes)
	{
		return UsageError("tune.json must be a regular file no larger than 256 MiB");
	}
	file_bytes = static_cast<std::uint64_t>(status.st_size);
	std::string text;
	text.reserve(first_line ? 0 : static_cast<std::size_t>(status.st_size));
	char buf[4096];
	for (;;)
	{
		errno = 0;
		const auto count = std::fread(buf, 1, sizeof buf, file.get());
		const int code = errno;
		if (std::ferror(file.get()))
		{
			return cmt::Failure(cmt::Error::Kind::Os, "cannot read tuning table", code ? code : EIO);
		}
		if (count > kMaxConfigBytes - text.size())
		{
			return cmt::Failure(cmt::Error::Kind::Usage, "tune.json exceeds the 256 MiB input size limit");
		}
		text.append(buf, count);
		if (count < sizeof buf || (first_line && (std::memchr(buf, '\n', count) != nullptr ||
												  text.size() >= kMaxSelectionLineBytes)))
		{
			break;
		}
	}
	if (std::fclose(file.release()) != 0)
	{
		const int code = errno;
		return cmt::Failure(cmt::Error::Kind::Os, "cannot close tuning table", code);
	}
	return text;
}

Result<void> ApplyStagingOverrides(Table& table, const RunOverrides& overrides)
{
	if (!overrides.HasStaging())
	{
		return {};
	}
	if (!table.HasStagedCopies())
	{
		return UsageError("overrides change existing staged selections only; this file selects none");
	}
	for (auto& row : table.rows)
	{
		if (row.mode != Mode::Chunks)
		{
			continue;
		}
		row.settings.workers = overrides.pipeline_workers.value_or(row.settings.workers);
		row.settings.slots = overrides.slots_per_worker.value_or(row.settings.slots);
		row.settings.chunk_bytes = overrides.chunk_bytes.value_or(row.settings.chunk_bytes);
		row.settings.wc_slots = overrides.wc_slots.value_or(row.settings.wc_slots);
		if (overrides.cpu_loop)
		{
			row.settings.loop = *overrides.cpu_loop;
			row.loop_fell_back = false;
		}
	}
	if (table.MaxPinnedBytes() > table.max_pinned_bytes)
	{
		return UsageError("overridden staging pools exceed the recorded or local pinned-memory budget");
	}
	return {};
}

} // namespace

cmt::Result<std::shared_ptr<const cmt::Table>> Load(const char* path, const RunOverrides& overrides)
{
	std::uint64_t size = 0;
	auto text = ReadTableFile(path, true, size);
	if (!text)
	{
		return cmt::Failure(std::move(text.error()));
	}
	LocalCapabilities capabilities;
	const auto available = HostMemoryAvailable(Pages::Small);
	capabilities.max_pinned_bytes = available.value_or(0);
	capabilities.progressive_pinning =
		overrides.progressive_pinning.value_or(ProgressivePinning::Auto) != ProgressivePinning::Off &&
		ProgressivePinningAvailable();
	std::uint64_t file_bytes = 0;
	auto table = ReadSelection(*text, capabilities, file_bytes);
	if (table && size != file_bytes)
	{
		LogLine(
			"config: WARNING: {} has {} bytes, its selection line was written for {}: the file changed after "
			"tuning; `cmt check {}` compares the line with the runs",
			path, size, file_bytes, path);
	}
	else if (!table && SelectionLineBytes(*text))
	{
		LogLine("config: {}: selection line not used ({}); parsing every run", path, table.error().Message());
	}
	else if (!table)
	{
		LogDebug("config: {}: no selection line (an older or reformatted file); parsing every run", path);
	}
	if (!table)
	{
		const auto line_error = table.error();
		text = ReadTableFile(path, false, size);
		table = text ? cmt::ParseTuningResult(*text, capabilities) : cmt::Failure(text.error());
		if (!table && text && share::IsShare(*text))
		{
			LogLine(
				"config: {} is a shared tune, and its selection line is not usable here; `cmt tune` measures "
				"this machine",
				path);
			return cmt::Failure(line_error);
		}
		if (!table)
		{
			return cmt::Failure(std::move(table.error()));
		}
	}
	for (const Row& row : table->rows)
	{
		if (row.mode == Mode::Chunks)
		{
			LogDebug("tune table: row {} {} {} runs {}", DirName(row.dir), row.min_bytes,
					 row.max_bytes == UINT64_MAX ? std::string("inf") : std::to_string(row.max_bytes),
					 hwcopy::Describe(row.settings.loop));
		}
	}
	if (!available)
	{
		LogLine("config: host-memory headroom unavailable; recorded staging is excluded");
	}
	else if (table->max_pinned_bytes == *available)
	{
		LogLine("config: local host-memory headroom limits pinned pools to {} bytes", *available);
	}
	CMT_RETURN_IF_ERROR(ApplyStagingOverrides(*table, overrides));
	if (overrides.numa)
	{
		table->worker_placement.gpu_numa = *overrides.numa ? std::nullopt : std::optional<bool>{false};
	}
	return std::make_shared<const cmt::Table>(std::move(*table));
}

void Print(const cmt::Table& table, const std::string& path, int fd)
{
	LogTo(fd, "config: {} derived ranges from {}", table.rows.size(), path);
}

} // namespace tuning
} // namespace cmt

namespace cmt {
namespace {
Result<void> RequireJson(bool condition)
{
	if (!condition)
	{
		return UsageError("invalid measured-copy JSON field");
	}
	return {};
}

Result<void> UniqueFields(Json value)
{
	if (value.is_object())
	{
		std::set<std::string_view> keys;
		const auto object = value.get_object().value_unsafe();
		for (auto field : object)
		{
			CMT_RETURN_IF_ERROR(RequireJson(keys.insert(field.key).second));
			CMT_RETURN_IF_ERROR(UniqueFields(field.value));
		}
	}
	else if (value.is_array())
	{
		const auto array = value.get_array().value_unsafe();
		for (Json item : array)
		{
			CMT_RETURN_IF_ERROR(UniqueFields(item));
		}
	}
	return {};
}

bool OnlyFields(Json value, std::initializer_list<std::string_view> known)
{
	simdjson::dom::object object;
	if (value.get(object) != simdjson::SUCCESS)
	{
		return false;
	}
	for (auto field : object)
	{
		if (std::find(known.begin(), known.end(), field.key) == known.end())
		{
			return false;
		}
	}
	return true;
}

Result<std::string> StringField(Json object, std::string_view name)
{
	std::string_view value;
	CMT_RETURN_IF_ERROR(RequireJson(object[name].get(value) == simdjson::SUCCESS));
	return std::string(value);
}

Result<std::uint64_t> UnsignedField(Json object, std::string_view name)
{
	std::uint64_t value = 0;
	CMT_RETURN_IF_ERROR(RequireJson(object[name].get(value) == simdjson::SUCCESS));
	return value;
}

std::string SettingsKey(const CopyRun& run)
{
	if (run.mode == Mode::ProgressivePinning)
	{
		return fmt::format("pinning:{}:{}:{}:{}", run.pinning.first_chunk_bytes, run.pinning.max_chunk_bytes,
						   run.pinning.max_registered_bytes
							   ? std::to_string(*run.pinning.max_registered_bytes)
							   : std::string("none"),
						   run.pinning.read_only);
	}
	return fmt::format("{}:{}:{}:{}:{}:{}", static_cast<int>(run.mode), run.settings.workers,
					   run.settings.slots, run.settings.slot_bytes, run.settings.loop, run.settings.wc_slots);
}

} // namespace

double Median(std::vector<double> values)
{
	assert(!values.empty());
	if (values.empty())
	{
		return std::numeric_limits<
			double>::quiet_NaN();
	}
	std::sort(values.begin(), values.end());
	const auto middle = values.size() / 2;
	return values.size() % 2 ? values[middle]
							 : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

const char* ModeName(Mode mode)
{
	switch (mode)
	{
	case Mode::Driver:
		return "driver";
	case Mode::Chunks:
		return "staged";
	case Mode::ProgressivePinning:
		return "progressive_pinning";
	}
	return "invalid";
}

const Row* Table::Find(Dir direction, std::uint64_t bytes) const
{
	for (const auto& row : rows)
	{
		if (row.dir == direction && bytes >= row.min_bytes && bytes < row.max_bytes)
		{
			return &row;
		}
	}
	return nullptr;
}

bool Table::HasStagedCopies() const
{
	return std::any_of(rows.begin(), rows.end(), [](const Row& row) { return row.mode == Mode::Chunks; });
}

bool Table::HasProgressivePinning() const
{
	return std::any_of(rows.begin(), rows.end(),
					   [](const Row& row) { return row.mode == Mode::ProgressivePinning; });
}

std::size_t Table::MaxPinnedBytes(bool wc_slots) const
{
	std::size_t result = 0;
	for (const auto& row : rows)
	{
		if (row.mode == Mode::Chunks && row.settings.wc_slots == wc_slots)
		{
			result = std::max(result, static_cast<std::size_t>(row.settings.workers) * row.settings.slots *
										  row.settings.chunk_bytes);
		}
	}
	return result;
}

std::size_t Table::MaxPinnedBytes() const
{
	return MaxPinnedBytes(false) + MaxPinnedBytes(true);
}

namespace {
Result<void> ValidatePinningSettings(const PinningSettings& settings)
{
	const auto& limit = settings.max_registered_bytes;
	if (settings.first_chunk_bytes == 0 || settings.first_chunk_bytes > settings.max_chunk_bytes ||
		settings.max_chunk_bytes > kMaxBytes ||
		(limit && (settings.max_chunk_bytes > *limit || *limit > kMaxBytes)))
	{
		return UsageError(
			"pinning requires 0 < first chunk <= maximum chunk <= registration limit (if set) <= 1 TiB");
	}
	return {};
}
} // namespace

Result<void> ValidateCopyRun(const CopyRun& run)
{
	if (run.bytes == 0 || run.process_round == 0 ||
		(run.direction != Dir::HtoD && run.direction != Dir::DtoH))
	{
		return UsageError("copy run requires positive bytes/round and a host-device direction");
	}
	if (run.mode == Mode::Chunks)
	{
		const auto& settings = run.settings;
		if (settings.workers < 1 || settings.workers > static_cast<int>(kMaxPipelineWorkers) ||
			settings.slots < 1 || settings.slots > static_cast<int>(kMaxStagingBuffers) ||
			settings.slot_bytes < kMinSlotBytes || settings.slot_bytes > kMaxSlotBytes ||
			settings.slot_bytes % kChunkSizeGranularity != 0 || !ParseLoopName(NativeArch(), settings.loop))
		{
			return UsageError("invalid staged copy settings");
		}
	}
	else if (run.mode == Mode::ProgressivePinning)
	{
		CMT_RETURN_IF_ERROR(ValidatePinningSettings(run.pinning));
		if (run.coverage != "measured" ||
			(run.status == "complete" &&
			 (run.execution != "hooked_copy" || run.timing_boundary != "completed_pageable_copy")))
		{
			return UsageError("progressive pinning requires a measured completed-pageable-copy hook run");
		}
	}
	else if (run.mode != Mode::Driver)
	{
		return UsageError("unknown copy method");
	}
	if (run.coverage != "measured" && run.coverage != "instant")
	{
		return UsageError("unknown copy-run coverage");
	}
	if (run.status == "complete")
	{
		if (!run.verified || !(run.seconds > 0) || !(run.gbps > 0) || !std::isfinite(run.seconds) ||
			!std::isfinite(run.gbps))
		{
			return UsageError("completed copy requires verified positive finite measurements");
		}
		const double derived = static_cast<double>(run.bytes) / run.seconds / 1e9;
		if (std::abs(derived - run.gbps) > derived * 1e-5)
		{
			return UsageError("copy bandwidth disagrees with bytes and elapsed time");
		}
		if (run.sample_gbps.empty() ||
			std::any_of(run.sample_gbps.begin(), run.sample_gbps.end(),
						[](double gbps) { return !std::isfinite(gbps) || gbps <= 0; }))
		{
			return UsageError("completed copy requires positive finite per-sample rates");
		}
	}
	else if (run.status != "failed" && run.status != "unsupported" && run.status != "capacity")
	{
		return UsageError("unknown copy run status");
	}
	return {};
}

Result<StagingSettings> ParseChunksSettings(Json settings)
{
	CMT_RETURN_IF_ERROR(UniqueFields(settings));
	simdjson::dom::object fields;
	CMT_RETURN_IF_ERROR(RequireJson(settings.get(fields) == simdjson::SUCCESS));
	for (auto field : fields)
	{
		if (field.key != "method" && field.key != "cpu_loop" && field.key != "pipeline_workers" &&
			field.key != "slots_per_worker" && field.key != "chunk_bytes" && field.key != "staging_memory")
		{
			return UsageError("staged copy settings contain an unknown or pinning field");
		}
	}
	auto method = StringField(settings, "method");
	if (!method || *method != "staged")
	{
		return UsageError("staged copy settings require method staged");
	}
	StagingSettings result;
	auto workers = UnsignedField(settings, "pipeline_workers");
	auto slots = UnsignedField(settings, "slots_per_worker");
	auto chunk = UnsignedField(settings, "chunk_bytes");
	auto loop = StringField(settings, "cpu_loop");
	auto memory = StringField(settings, "staging_memory");
	if (!workers || !slots || !chunk || !loop || !memory || *workers > kMaxPipelineWorkers ||
		*slots > kMaxStagingBuffers || *chunk > kMaxSlotBytes)
	{
		return UsageError("invalid staged copy JSON settings");
	}
	CMT_RETURN_IF_ERROR(RequireJson(*memory == "cached_pinned" || *memory == "wc_pinned"));
	result.workers = static_cast<int>(*workers);
	result.slots = static_cast<int>(*slots);
	result.slot_bytes = static_cast<std::size_t>(*chunk);
	result.loop = *loop;
	result.wc_slots = *memory == "wc_pinned";
	if (result.workers < 1 || result.slots < 1 || result.slot_bytes < kMinSlotBytes ||
		result.slot_bytes % kChunkSizeGranularity != 0 || !ParseLoopName(NativeArch(), result.loop))
	{
		return UsageError("invalid staged copy settings limits");
	}
	return result;
}

Result<PinningSettings> ParsePinningSettings(Json settings)
{
	CMT_RETURN_IF_ERROR(UniqueFields(settings));
	simdjson::dom::object fields;
	CMT_RETURN_IF_ERROR(RequireJson(settings.get(fields) == simdjson::SUCCESS));
	for (auto field : fields)
	{
		if (field.key != "method" && field.key != "first_chunk_bytes" && field.key != "max_chunk_bytes" &&
			field.key != "max_registered_bytes" && field.key != "read_only")
		{
			return UsageError("progressive pinning settings contain an unknown or staging field");
		}
	}
	auto first = UnsignedField(settings, "first_chunk_bytes");
	auto maximum = UnsignedField(settings, "max_chunk_bytes");
	const bool unlimited = settings["max_registered_bytes"].is_null();
	auto limit = unlimited ? Result<std::uint64_t>(0) : UnsignedField(settings, "max_registered_bytes");
	auto method = StringField(settings, "method");
	PinningSettings result;
	if (!method || *method != "progressive_pinning" || !first || !maximum || !limit ||
		settings["read_only"].get(result.read_only) != simdjson::SUCCESS)
	{
		return UsageError(
			"progressive pinning requires chunk sizes, a registration limit or null, and read_only");
	}
	result.first_chunk_bytes = *first;
	result.max_chunk_bytes = *maximum;
	result.max_registered_bytes = unlimited ? std::nullopt : std::optional<std::uint64_t>(*limit);
	CMT_RETURN_IF_ERROR(ValidatePinningSettings(result));
	return result;
}

Result<std::optional<PinningSettings>> ReadBenchmarkPinning(const char* path)
{
	constexpr std::size_t kMaxBenchmarkRequestBytes = 64 * 1024;
	auto text = ReadTextFile(path, kMaxBenchmarkRequestBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json document;
	const auto error = parser.parse(*text).get(document);
	if (error == simdjson::MEMALLOC)
	{
		Panic("benchmark request allocation failed");
	}
	if (error != simdjson::SUCCESS || !document.is_object())
	{
		return UsageError("invalid fixed benchmark request JSON");
	}
	CMT_RETURN_IF_ERROR(UniqueFields(document));
	auto schema = StringField(document, "schema");
	auto version = UnsignedField(document, "version");
	auto direction = StringField(document, "direction");
	auto round = UnsignedField(document, "process_round");
	Json settings;
	simdjson::dom::array sizes;
	if (!schema || *schema != "cmt.benchmark-run" || !version || *version != 2 || !direction ||
		(*direction != "htod" && *direction != "dtoh") ||
		document["sizes_bytes"].get(sizes) != simdjson::SUCCESS || sizes.size() == 0 || !round ||
		*round == 0 || *round > UINT_MAX || document["settings"].get(settings) != simdjson::SUCCESS)
	{
		return UsageError("fixed benchmark request has invalid identity fields");
	}
	auto method = StringField(settings, "method");
	if (!method)
	{
		return Failure(method.error());
	}
	if (*method == "progressive_pinning")
	{
		auto pinning = ParsePinningSettings(settings);
		if (!pinning)
		{
			return Failure(pinning.error());
		}
		return std::optional<PinningSettings>{*pinning};
	}
	if (*method == "staged")
	{
		CMT_RETURN_IF_ERROR(ParseChunksSettings(settings));
	}
	else if (*method != "driver")
	{
		return UsageError("fixed benchmark request has an unknown method");
	}
	return std::optional<PinningSettings>{};
}

Result<std::vector<CopyRun>> ReadCopyRuns(Json document)
{
	CMT_RETURN_IF_ERROR(UniqueFields(document));
	simdjson::dom::array values;
	CMT_RETURN_IF_ERROR(RequireJson(document["runs"].get(values) == simdjson::SUCCESS));
	std::vector<CopyRun> runs;
	for (Json value : values)
	{
		CopyRun run;
		auto direction = StringField(value, "direction");
		auto bytes = UnsignedField(value, "bytes");
		auto round = UnsignedField(value, "process_round");
		auto status = StringField(value, "status");
		auto execution = StringField(value, "execution");
		auto boundary = StringField(value, "timing_boundary");
		if (!direction || !bytes || !round || !status || !execution || !boundary)
		{
			return UsageError("copy run is missing required identity fields");
		}
		CMT_RETURN_IF_ERROR(RequireJson(*direction == "htod" || *direction == "dtoh"));
		CMT_RETURN_IF_ERROR(RequireJson(*round <= UINT_MAX));
		run.direction = *direction == "htod" ? Dir::HtoD : Dir::DtoH;
		run.bytes = *bytes;
		run.process_round = static_cast<unsigned>(*round);
		run.status = *status;
		run.execution = *execution;
		run.timing_boundary = *boundary;
		Json optional;
		if (value["measurement_class"].get(optional) == simdjson::SUCCESS)
		{
			auto classification = StringField(value, "measurement_class");
			if (!classification)
			{
				return Failure(classification.error());
			}
			run.measurement_class = *classification;
		}
		if (value["coverage"].get(optional) == simdjson::SUCCESS)
		{
			auto coverage = StringField(value, "coverage");
			if (!coverage)
			{
				return Failure(coverage.error());
			}
			run.coverage = *coverage;
		}
		if (value["reason"].get(optional) == simdjson::SUCCESS)
		{
			auto reason = StringField(value, "reason");
			if (!reason)
			{
				return Failure(reason.error());
			}
			run.reason = *reason;
		}
		Json settings;
		CMT_RETURN_IF_ERROR(RequireJson(value["settings"].get(settings) == simdjson::SUCCESS));
		auto method = StringField(settings, "method");
		if (!method)
		{
			return Failure(method.error());
		}
		CMT_RETURN_IF_ERROR(
			RequireJson(*method == "driver" || *method == "staged" || *method == "progressive_pinning"));
		run.mode = *method == "driver"	 ? Mode::Driver
				   : *method == "staged" ? Mode::Chunks
										 : Mode::ProgressivePinning;
		if (run.mode == Mode::Chunks)
		{
			auto parsed = ParseChunksSettings(settings);
			if (!parsed)
			{
				return Failure(parsed.error());
			}
			run.settings = std::move(*parsed);
		}
		else if (run.mode == Mode::ProgressivePinning)
		{
			auto parsed = ParsePinningSettings(settings);
			if (!parsed)
			{
				return Failure(parsed.error());
			}
			run.pinning = *parsed;
		}
		if (run.status == "complete")
		{
			CMT_RETURN_IF_ERROR(RequireJson(value["seconds"].get(run.seconds) == simdjson::SUCCESS &&
											value["gbps"].get(run.gbps) == simdjson::SUCCESS &&
											value["verified"].get(run.verified) == simdjson::SUCCESS));
		}
		if (run.status == "complete")
		{
			auto count = UnsignedField(value, "sample_count");
			auto median = UnsignedField(value, "median_sample_index");
			simdjson::dom::array samples;
			if (!count || !median || *count == 0 || *median >= *count ||
				value["samples"].get(samples) != simdjson::SUCCESS || samples.size() != *count)
			{
				return UsageError("completed run needs all raw samples and its median sample index");
			}
			std::size_t sample_index = 0;
			std::uint64_t copies_per_sample = 0;
			std::vector<double> elapsed_samples;
			for (Json sample : samples)
			{
				double seconds = 0;
				double elapsed = 0;
				auto copies = UnsignedField(sample, "copies_per_sample");
				if (!copies || *copies == 0 || (sample_index > 0 && *copies != copies_per_sample) ||
					sample["seconds"].get(seconds) != simdjson::SUCCESS ||
					sample["elapsed_seconds"].get(elapsed) != simdjson::SUCCESS || !std::isfinite(seconds) ||
					!std::isfinite(elapsed) || seconds <= 0 || elapsed <= 0 ||
					std::abs(seconds * static_cast<double>(*copies) - elapsed) > elapsed * 1e-5 ||
					(sample_index == *median && std::abs(seconds - run.seconds) > seconds * 1e-5))
				{
					return UsageError("raw copy sample disagrees with aggregate timing");
				}
				copies_per_sample = *copies;
				elapsed_samples.push_back(elapsed);
				run.sample_gbps.push_back(static_cast<double>(run.bytes) / seconds / 1e9);
				++sample_index;
			}
			const double selected_elapsed = elapsed_samples[*median];
			std::sort(elapsed_samples.begin(), elapsed_samples.end());
			if (selected_elapsed != elapsed_samples[elapsed_samples.size() / 2])
			{
				return UsageError("selected sample is not the median completed-copy sample");
			}
		}
		if (run.status == "complete" && run.mode != Mode::Driver && run.execution == "hooked_copy")
		{
			const auto observed = [&](const char* name, double expected)
			{
				double actual = 0;
				return value[name].get(actual) == simdjson::SUCCESS && actual == expected;
			};
			double calls = 0;
			if (!observed("tune_audit_valid", 1) || !observed("tune_audit_measured_rejected", 0) ||
				!observed("tune_audit_measured_errors", 0) || !observed("tune_audit_measured_mixed", 0) ||
				value["tune_audit_measured_calls"].get(calls) != simdjson::SUCCESS || !std::isfinite(calls) ||
				calls < 1 || std::floor(calls) != calls || !observed("tune_audit_measured_accepted", calls) ||
				!observed("tune_audit_measured_bytes", calls * static_cast<double>(run.bytes)))
			{
				return UsageError("completed hooked copy lacks matching accepted-call observations");
			}
			if (run.mode == Mode::Chunks)
			{
				const auto loop = ParseLoopName(NativeArch(), run.settings.loop);
				if (!loop || !observed("tune_audit_write_back", static_cast<unsigned>(loop->write_back)) ||
					!observed("tune_audit_flush_bytes", CopyCtx{}.flush_batch) ||
					!observed("tune_audit_requested_workers", run.settings.workers) ||
					!observed("tune_audit_slots", run.settings.slots) ||
					!observed("tune_audit_slot_bytes", static_cast<double>(run.settings.slot_bytes)) ||
					!observed("tune_audit_wc_slots", run.settings.wc_slots ? 1 : 0) ||
					!observed("tune_audit_bench_dma", 0) || !observed("tune_audit_pace_gbps", 0))
				{
					return UsageError("completed hooked copy lacks matching staging and loop observations");
				}
			}
			else if (run.mode == Mode::ProgressivePinning)
			{
				double registered_bytes = 0;
				double chunks = 0;
				if (!observed("tune_audit_version", kProgressivePinningAuditVersion) ||
					!observed("tune_audit_row_mode", static_cast<unsigned>(Mode::ProgressivePinning)) ||
					!observed("tune_audit_row_min_bytes", 0) || !observed("tune_audit_row_max_bytes", 0) ||
					!observed("tune_audit_pinning_completed", 1) ||
					value["tune_audit_pinning_registered_bytes"].get(registered_bytes) != simdjson::SUCCESS ||
					!std::isfinite(registered_bytes) || std::floor(registered_bytes) != registered_bytes ||
					registered_bytes < static_cast<double>(run.bytes) ||
					(run.pinning.max_registered_bytes &&
					 registered_bytes > static_cast<double>(*run.pinning.max_registered_bytes)) ||
					!observed("tune_audit_pinning_released_bytes", registered_bytes) ||
					!observed("tune_audit_pinning_registered_dma_bytes", static_cast<double>(run.bytes)) ||
					!observed("tune_audit_pinning_pageable_dma_bytes", 0) ||
					!observed("tune_audit_pinning_borrowed_bytes", 0) ||
					value["tune_audit_pinning_chunks"].get(chunks) != simdjson::SUCCESS ||
					!std::isfinite(chunks) || std::floor(chunks) != chunks || chunks < 1 ||
					!observed("tune_audit_pinning_first_chunk_bytes",
							  static_cast<double>(run.pinning.first_chunk_bytes)) ||
					!observed("tune_audit_pinning_max_chunk_bytes",
							  static_cast<double>(run.pinning.max_chunk_bytes)) ||
					!observed("tune_audit_pinning_max_registered_bytes",
							  static_cast<double>(run.pinning.max_registered_bytes.value_or(0))) ||
					!observed("tune_audit_pinning_read_only", run.pinning.read_only ? 1 : 0))
				{
					return UsageError(
						"completed pinning lacks matching registration and release observations");
				}
			}
		}
		CMT_RETURN_IF_ERROR(ValidateCopyRun(run));
		runs.push_back(std::move(run));
	}
	return runs;
}

void WriteCopyRun(JsonWriter& writer, const CopyRun& run, std::optional<Pages> pages)
{
	writer.start_object();
	JsonField(writer, "direction", DirName(run.direction), true);
	JsonField(writer, "bytes", run.bytes);
	JsonField(writer, "process_round", run.process_round);
	JsonField(writer, "status", run.status);
	JsonField(writer, "execution", run.execution);
	JsonField(writer, "timing_boundary", run.timing_boundary);
	JsonField(writer, "coverage", run.coverage);
	JsonField(writer, "measurement_class", run.measurement_class);
	JsonField(writer, "reason", run.reason);
	JsonField(writer, "verified", run.verified);
	if (pages)
	{
		JsonField(writer, "pages_requested", PagesName(*pages));
	}
	if (run.status == "complete")
	{
		JsonField(writer, "seconds", run.seconds);
		JsonField(writer, "gbps", run.gbps);
	}
	JsonKey(writer, "settings");
	WriteSettings(writer, run.mode, run.settings, run.pinning);
	writer.end_object();
}

void WriteSettings(JsonWriter& writer, Mode mode, const StagingSettings& settings,
				   const PinningSettings& pinning)
{
	writer.start_object();
	JsonField(writer, "method", ModeName(mode), true);
	if (mode == Mode::Chunks)
	{
		JsonField(writer, "cpu_loop", settings.loop);
		JsonField(writer, "pipeline_workers", settings.workers);
		JsonField(writer, "slots_per_worker", settings.slots);
		JsonField(writer, "chunk_bytes", settings.slot_bytes);
		JsonField(writer, "staging_memory", settings.wc_slots ? "wc_pinned" : "cached_pinned");
	}
	else if (mode == Mode::ProgressivePinning)
	{
		JsonField(writer, "first_chunk_bytes", pinning.first_chunk_bytes);
		JsonField(writer, "max_chunk_bytes", pinning.max_chunk_bytes);
		JsonField(writer, "max_registered_bytes", pinning.max_registered_bytes);
		JsonField(writer, "read_only", pinning.read_only);
	}
	writer.end_object();
}

namespace {
bool LoopRuns(const std::string& name, const CpuIsas& cpu)
{
	const auto loop = ParseLoopName(NativeArch(), name);
	return loop && Resolve(NativeArch(), *loop, cpu, std::nullopt);
}

void SetStaged(Row& row, const StagingSettings& recorded, const CpuIsas& cpu)
{
	CopyCtx context{};
	const auto selected = hwcopy::Select(recorded.loop, context, std::nullopt, cpu);
	row.mode = Mode::Chunks;
	row.settings = {
		static_cast<unsigned>(recorded.workers), static_cast<uint32_t>(recorded.slots), recorded.slot_bytes,
		selected ? selected->loop : hwcopy::DefaultLoop(context, std::nullopt, cpu), recorded.wc_slots};
	row.loop_fell_back = !selected || selected->fell_back;
	row.loop_name = recorded.loop;
}
} // namespace

Result<Table> BuildCopyLookup(const std::vector<CopyRun>& runs, const SelectionPolicy& policy,
							  const LocalCapabilities& capabilities)
{
	if (policy.version != kSelectionPolicyVersion || policy.process_rounds == 0 ||
		!std::isfinite(policy.minimum_gain) || policy.minimum_gain < 0 || !(policy.tie_band >= 0) ||
		policy.tie_band > 1)
	{
		return UsageError("unsupported or invalid selection policy");
	}
	for (const auto& run : runs)
	{
		CMT_RETURN_IF_ERROR(ValidateCopyRun(run));
	}
	Table table;
	table.max_pinned_bytes = capabilities.max_pinned_bytes;
	for (Dir direction : {Dir::HtoD, Dir::DtoH})
	{
		std::map<std::uint64_t, Row> measured;
		std::map<std::uint64_t, Row> instant;
		std::map<std::uint64_t, Row> measured_unpinned;
		std::map<std::uint64_t, Row> instant_unpinned;
		std::set<std::uint64_t> requested_measured_sizes;
		for (const auto& run : runs)
		{
			if (run.direction == direction && run.coverage == "measured" &&
				run.measurement_class == "primary" && run.timing_boundary == "completed_pageable_copy")
			{
				requested_measured_sizes.insert(run.bytes);
			}
		}
		for (const std::string coverage : {"measured", "instant"})
		{
			std::map<std::uint64_t, std::map<std::string, std::vector<const CopyRun*>>> points;
			for (const auto& run : runs)
			{
				if (run.direction != direction || run.coverage != coverage || run.status != "complete" ||
					run.timing_boundary != "completed_pageable_copy" || run.measurement_class != "primary")
				{
					continue;
				}
				const char* execution = run.mode == Mode::Driver ? "bare_driver"
										: coverage == "instant"	 ? "direct_engine"
																 : "hooked_copy";
				if (run.execution != execution)
				{
					continue;
				}
				if (run.mode == Mode::Chunks)
				{
					const auto pinned = static_cast<std::size_t>(run.settings.workers) * run.settings.slots *
										run.settings.slot_bytes;
					if (!LoopRuns(run.settings.loop, capabilities.cpu) ||
						pinned > capabilities.max_pinned_bytes)
					{
						continue;
					}
				}
				else if (run.mode == Mode::ProgressivePinning &&
						 (!capabilities.progressive_pinning || run.bytes < run.pinning.first_chunk_bytes ||
						  (run.pinning.max_registered_bytes &&
						   run.bytes > *run.pinning.max_registered_bytes)))
				{
					continue;
				}
				points[run.bytes][SettingsKey(run)].push_back(&run);
			}
			for (const auto& [bytes, configurations] : points)
			{
				struct ConfigurationSummary
				{
					const CopyRun* run = nullptr;
					double median_gbps = 0;
					double minimum_gbps = 0;
					double maximum_gbps = 0;
				};

				ConfigurationSummary driver;
				std::vector<ConfigurationSummary> optimized_configurations;
				for (const auto& [key, processes] : configurations)
				{
					(void)key;
					std::set<unsigned> rounds;
					std::vector<double> rates;
					for (const auto* process : processes)
					{
						if (!rounds.insert(process->process_round).second)
						{
							return UsageError("duplicate copy run round");
						}
						rates.insert(rates.end(), process->sample_gbps.begin(), process->sample_gbps.end());
					}
					const unsigned required = coverage == "instant" ? 1 : policy.process_rounds;
					if (rounds.size() != required || *rounds.begin() != 1 || *rounds.rbegin() != required ||
						(coverage == "measured" && rates.size() < kMinMeasuredSamples))
					{
						continue;
					}
					ConfigurationSummary summary;
					summary.run = processes.front();
					const auto extremes = std::minmax_element(rates.begin(), rates.end());
					summary.minimum_gbps = *extremes.first;
					summary.maximum_gbps = *extremes.second;
					summary.median_gbps = Median(std::move(rates));
					if (processes.front()->mode == Mode::Driver)
					{
						driver = summary;
					}
					else
					{
						optimized_configurations.push_back(summary);
					}
				}
				if (!driver.run || (coverage == "measured" && optimized_configurations.empty()))
				{
					continue;
				}
				const auto order = [size = bytes](const ConfigurationSummary* summary)
				{
					const auto& settings = summary->run->settings;
					const bool staged = summary->run->mode == Mode::Chunks;
					return std::tuple{staged && settings.slot_bytes >= size,
									  staged ? static_cast<std::size_t>(settings.workers) * settings.slots *
												   settings.slot_bytes
											 : 0,
									  settings.workers, -summary->median_gbps};
				};
				const auto choose = [&](bool with_pinning)
				{
					const ConfigurationSummary* best = nullptr;
					std::vector<const ConfigurationSummary*> eligible;
					for (const auto& optimized : optimized_configurations)
					{
						if (!with_pinning && optimized.run->mode == Mode::ProgressivePinning)
						{
							continue;
						}
						if (coverage == "measured" && optimized.minimum_gbps <= driver.maximum_gbps)
						{
							continue;
						}
						if (optimized.median_gbps <= driver.median_gbps * (1 + policy.minimum_gain))
						{
							continue;
						}
						eligible.push_back(&optimized);
						if (!best || optimized.median_gbps > best->median_gbps)
						{
							best = &optimized;
						}
					}
					const double tie_floor = best ? best->median_gbps * (1 - policy.tie_band) : 0;
					for (const auto* tied : eligible)
					{
						if (policy.tie_band > 0 && tied->median_gbps >= tie_floor &&
							order(tied) < order(best))
						{
							best = tied;
						}
					}
					return best;
				};
				const auto make_row = [&, size = bytes](const ConfigurationSummary* best)
				{
					Row row;
					row.dir = direction;
					row.min_bytes = size;
					row.max_bytes = size == UINT64_MAX ? UINT64_MAX : size + 1;
					if (best != nullptr && best->run->mode == Mode::ProgressivePinning)
					{
						row.mode = Mode::ProgressivePinning;
						row.pinning = best->run->pinning;
					}
					else if (best != nullptr)
					{
						SetStaged(row, best->run->settings, capabilities.cpu);
					}
					return row;
				};
				const ConfigurationSummary* best = choose(true);
				(coverage == "instant" ? instant : measured)[bytes] = make_row(best);
				const bool others_compared =
					coverage == "instant" ||
					std::any_of(optimized_configurations.begin(), optimized_configurations.end(),
								[](const ConfigurationSummary& summary)
								{ return summary.run->mode != Mode::ProgressivePinning; });
				if (best != nullptr && best->run->mode == Mode::ProgressivePinning && others_compared)
				{
					(coverage == "instant" ? instant_unpinned : measured_unpinned)[bytes] =
						make_row(choose(false));
				}
			}
		}
		std::set<std::uint64_t> boundaries{0, UINT64_MAX};
		boundaries.insert(requested_measured_sizes.begin(), requested_measured_sizes.end());
		for (const auto* points : {&instant, &measured})
		{
			for (const auto& [bytes, row] : *points)
			{
				boundaries.insert(bytes);
				if (row.mode == Mode::ProgressivePinning && row.pinning.max_registered_bytes)
				{
					boundaries.insert(*row.pinning.max_registered_bytes + 1);
				}
			}
		}
		for (auto it = boundaries.begin(); std::next(it) != boundaries.end(); ++it)
		{
			Row row;
			row.dir = direction;
			row.min_bytes = *it;
			row.max_bytes = *std::next(it);
			for (const auto* points : {&instant, &measured})
			{
				if (points->empty() || *it < points->begin()->first)
				{
					continue;
				}
				auto found = points->upper_bound(*it);
				--found;
				if (points == &measured)
				{
					auto requested = requested_measured_sizes.upper_bound(*it);
					if (requested == requested_measured_sizes.begin())
					{
						continue;
					}
					--requested;
					if (found->first != *requested)
					{
						continue;
					}
				}
				const Row* chosen = &found->second;
				if (chosen->mode == Mode::ProgressivePinning &&
					(*it < chosen->pinning.first_chunk_bytes ||
					 (chosen->pinning.max_registered_bytes && *it > *chosen->pinning.max_registered_bytes)))
				{
					const auto& unpinned = points == &measured ? measured_unpinned : instant_unpinned;
					const auto alternative = unpinned.find(found->first);
					if (alternative == unpinned.end())
					{
						continue;
					}
					chosen = &alternative->second;
				}
				row.mode = chosen->mode;
				row.settings = chosen->settings;
				row.pinning = chosen->pinning;
				row.loop_fell_back = chosen->loop_fell_back;
				row.loop_name = chosen->loop_name;
			}
			table.rows.push_back(std::move(row));
		}
	}
	if (table.MaxPinnedBytes() > capabilities.max_pinned_bytes)
	{
		return Unsupported("combined cached/WC selected pools exceed the memory budget");
	}
	return table;
}

namespace {
struct TuningResult
{
	std::vector<CopyRun> runs;
	std::string_view policy;
	Json document;
	Json request;
	Json source_machine;
};

Result<TuningResult> ReadTuningResult(JsonParser& parser, const std::string& text)
{
	TuningResult result;
	const auto error = parser.parse(text).get(result.document);
	if (error == simdjson::MEMALLOC)
	{
		Panic("tuning JSON allocation failed");
	}
	CMT_RETURN_IF_ERROR(RequireJson(error == simdjson::SUCCESS));
	auto schema = StringField(result.document, "schema");
	auto version = UnsignedField(result.document, "version");
	auto state = StringField(result.document, "state");
	if (!schema || *schema != "cmt.tuning" || !version || *version != 1 || !state ||
		(*state != "complete" && *state != "partial") ||
		result.document["selection_policy"].get(result.policy) != simdjson::SUCCESS)
	{
		return UsageError("expected a published cmt.tuning version 1 result");
	}
	auto runs = ReadCopyRuns(result.document);
	if (!runs)
	{
		return Failure(runs.error());
	}
	result.runs = std::move(*runs);
	CMT_RETURN_IF_ERROR(RequireJson(result.document["request"].get(result.request) == simdjson::SUCCESS));
	if (result.document["source_machine"].get(result.source_machine) != simdjson::SUCCESS)
	{
		return UsageError("tuning result needs source CPU and GPU model evidence");
	}
	return result;
}

Result<void> ReadRequest(Table& table, Json request, Json source_machine, std::size_t available)
{
	auto pinned_mib = UnsignedField(request, "max_pinned_mib");
	if (!pinned_mib || *pinned_mib == 0 || *pinned_mib > 32768)
	{
		return UsageError("tuning result needs a valid pinned-memory limit");
	}
	table.max_pinned_bytes = std::min(available, static_cast<std::size_t>(*pinned_mib) << 20);
	simdjson::dom::array environment;
	const auto environment_error = request["environment"].get(environment);
	if (environment_error != simdjson::NO_SUCH_FIELD && environment_error != simdjson::SUCCESS)
	{
		return UsageError("tuning request environment must be an array");
	}
	if (environment_error == simdjson::SUCCESS)
	{
		for (auto entry : environment)
		{
			std::string_view text;
			if (entry.get(text) != simdjson::SUCCESS || text.find('=') == std::string_view::npos)
			{
				return UsageError("tuning environment entries must be KEY=value strings");
			}
			const auto separator = text.find('=');
			auto placement = ParseWorkerPlacement(text.substr(0, separator), text.substr(separator + 1),
												  table.worker_placement);
			if (!placement)
			{
				return Failure(placement.error());
			}
		}
	}
	for (auto [pointer, destination] : {std::pair{"/host/cpu/model", &table.source_cpu_model},
										std::pair{"/host/cpu/arch", &table.source_cpu_arch},
										std::pair{"/gpus/0/name", &table.source_gpu_name}})
	{
		std::string_view field;
		if (source_machine.at_pointer(pointer).get(field) != simdjson::SUCCESS || field.empty())
		{
			return UsageError("tuning result needs source CPU and GPU model evidence");
		}
		*destination = field;
	}
	return {};
}

struct RegistrationLimit
{
	std::optional<std::uint64_t> recorded;
	std::optional<std::uint64_t> applied;
};

Result<RegistrationLimit> RegistrationLimitOf(Json request)
{
	constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
	std::string_view rule;
	const auto rule_error = request[kRegistrationLimitKey].get(rule);
	const bool explicit_only = rule_error == simdjson::SUCCESS && rule == kExplicitOnly;
	if (rule_error != simdjson::NO_SUCH_FIELD && !explicit_only)
	{
		return UsageError("the request's registration_limit must be \"explicit_only\"");
	}
	RegistrationLimit limit;
	if (explicit_only && request["max_registered_mib"].is_null())
	{
		return limit;
	}
	auto mib = UnsignedField(request, "max_registered_mib");
	if (!mib || *mib == 0 || *mib > kMaxBytes / kMiB)
	{
		return UsageError("pinning measurements require a registration limit of 1 MiB to 1 TiB, or null");
	}
	limit.recorded = *mib * kMiB;
	limit.applied = explicit_only || *mib != kLegacyDefaultRegistrationMiB ? limit.recorded : std::nullopt;
	return limit;
}

Result<Table> SelectTable(const std::vector<CopyRun>& runs, std::string_view policy_version, Json request,
						  Json source_machine, const LocalCapabilities& capabilities, bool tie = true)
{
	const std::vector<CopyRun>* selected = &runs;
	std::vector<CopyRun> lifted;
	SelectionPolicy policy;
	policy.version = std::string(policy_version);
	auto rounds = UnsignedField(request, "process_rounds");
	CMT_RETURN_IF_ERROR(RequireJson(rounds && *rounds > 0 && *rounds <= UINT_MAX));
	policy.process_rounds = static_cast<unsigned>(*rounds);
	std::uint64_t tie_percent = 0;
	const auto tie_error = request["tie_percent"].get(tie_percent);
	CMT_RETURN_IF_ERROR(RequireJson(tie_error == simdjson::SUCCESS || tie_error == simdjson::NO_SUCH_FIELD));
	policy.tie_band = tie ? static_cast<double>(tie_percent) / 100 : 0;
	if (std::any_of(runs.begin(), runs.end(),
					[](const CopyRun& run) { return run.mode == Mode::ProgressivePinning; }))
	{
		bool requested = false;
		auto search = StringField(request, "search");
		Json settings;
		if (request["progressive_pinning"].get(requested) != simdjson::SUCCESS || !requested || !search ||
			*search != "full_with_kernel" ||
			request["progressive_pinning_settings"].get(settings) != simdjson::SUCCESS)
		{
			return UsageError("pinning measurements require a full_with_kernel request");
		}
		auto limit = RegistrationLimitOf(request);
		if (!limit)
		{
			return Failure(limit.error());
		}
		auto expected = ParsePinningSettings(settings);
		if (!expected || expected->max_registered_bytes != limit->recorded)
		{
			return UsageError("requested pinning settings disagree with the registration limit");
		}
		for (const auto& run : runs)
		{
			if (run.mode == Mode::ProgressivePinning &&
				(run.pinning.first_chunk_bytes != expected->first_chunk_bytes ||
				 run.pinning.max_chunk_bytes != expected->max_chunk_bytes ||
				 run.pinning.max_registered_bytes != expected->max_registered_bytes ||
				 run.pinning.read_only != expected->read_only))
			{
				return UsageError("pinning measurement settings differ from the declared FULL request");
			}
		}
		if (limit->applied != limit->recorded)
		{
			lifted = runs;
			for (auto& run : lifted)
			{
				if (run.mode == Mode::ProgressivePinning)
				{
					run.pinning.max_registered_bytes = limit->applied;
				}
			}
			selected = &lifted;
		}
	}
	Table table;
	CMT_RETURN_IF_ERROR(ReadRequest(table, request, source_machine, capabilities.max_pinned_bytes));
	LocalCapabilities limits = capabilities;
	limits.max_pinned_bytes = table.max_pinned_bytes;
	auto lookup = BuildCopyLookup(*selected, policy, limits);
	if (!lookup)
	{
		return Failure(lookup.error());
	}
	table.rows = std::move(lookup->rows);
	return table;
}
} // namespace

Result<Table> ParseTuningResult(const std::string& text, const LocalCapabilities& capabilities)
{
	JsonParser parser;
	auto result = ReadTuningResult(parser, text);
	if (!result)
	{
		return Failure(result.error());
	}
	return SelectTable(result->runs, result->policy, result->request, result->source_machine, capabilities);
}

namespace {
constexpr std::string_view kSelectionKey = "{\"selection\":";
constexpr std::string_view kAbovePinningBudget = "above_pinning_budget";
constexpr std::string_view kUnpinnedChoice = "unpinned";

struct SelectionParts
{
	std::string policy;
	std::string request;
	std::string machine;
	std::string rows;
	std::optional<std::string> reference_rows;
	bool above_pinning_budget = false;
	bool registration_limit = false;
};

Result<std::string> EncodeSelection(const SelectionParts& parts, std::size_t rest_bytes)
{
	std::string line;
	for (std::size_t file_bytes = 0; std::size(line) + rest_bytes != file_bytes;)
	{
		file_bytes = std::size(line) + rest_bytes;
		JsonWriter writer;
		writer.start_object();
		JsonField(writer, "policy", std::string_view(parts.policy), true);
		JsonField(writer, "file_bytes", file_bytes);
		JsonKey(writer, "request");
		writer.append_raw(parts.request);
		JsonKey(writer, "source_machine");
		writer.append_raw(parts.machine);
		JsonKey(writer, "rows");
		writer.append_raw(parts.rows);
		if (parts.reference_rows)
		{
			JsonKey(writer, "tie_reference_rows");
			writer.append_raw(*parts.reference_rows);
		}
		if (parts.above_pinning_budget)
		{
			JsonField(writer, kAbovePinningBudget, kUnpinnedChoice);
		}
		if (parts.registration_limit)
		{
			JsonField(writer, kRegistrationLimitKey, kExplicitOnly);
		}
		writer.end_object();
		auto value = FinishJson(writer);
		if (!value)
		{
			return Failure(value.error());
		}
		value->pop_back();
		line = std::string(kSelectionKey) + *value + ",\n";
	}
	return line;
}

Result<std::string> JsonText(const JsonWriter& writer)
{
	auto text = FinishJson(writer);
	if (text)
	{
		text->pop_back();
	}
	return text;
}
} // namespace

Result<std::string> SelectionRequest(Json request)
{
	static constexpr std::string_view kRead[] = {"search",
												 "process_rounds",
												 "tie_percent",
												 "max_pinned_mib",
												 "progressive_pinning",
												 "progressive_pinning_settings",
												 "max_registered_mib",
												 kRegistrationLimitKey,
												 "environment"};
	simdjson::dom::object fields;
	if (request.get(fields))
	{
		return UsageError("tuning request must be an object");
	}
	JsonWriter writer;
	writer.start_object();
	bool first = true;
	for (auto field : fields)
	{
		if (std::find(std::begin(kRead), std::end(kRead), field.key) == std::end(kRead))
		{
			continue;
		}
		JsonKey(writer, field.key, first);
		first = false;
		simdjson::dom::array entries;
		if (field.key != "environment" || field.value.get(entries))
		{
			writer.append_raw(simdjson::to_string(field.value));
			continue;
		}
		writer.start_array();
		bool first_entry = true;
		for (Json entry : entries)
		{
			std::string_view text;
			if (!entry.get(text) && text.rfind("CMT_WORKER_", 0) != 0)
			{
				continue;
			}
			writer.append_raw(first_entry ? "" : ",");
			first_entry = false;
			writer.append_raw(simdjson::to_string(entry));
		}
		writer.end_array();
	}
	writer.end_object();
	return JsonText(writer);
}

Result<std::string> SelectionMachine(Json machine)
{
	const auto value = [&](std::string_view pointer) -> std::optional<std::string>
	{
		Json found;
		return machine.at_pointer(pointer).get(found)
				   ? std::nullopt
				   : std::optional<std::string>(simdjson::to_string(found));
	};
	const auto model = value("/host/cpu/model");
	const auto arch = value("/host/cpu/arch");
	const auto gpu = value("/gpus/0/name");
	const auto driver = value("/driver/kernel_module/version");
	JsonWriter writer;
	writer.start_object();
	JsonKey(writer, "host", true);
	writer.start_object();
	JsonKey(writer, "cpu", true);
	writer.start_object();
	bool first = true;
	for (const auto& [key, text] : {std::pair{"model", &model}, std::pair{"arch", &arch}})
	{
		if (*text)
		{
			JsonKey(writer, key, first);
			writer.append_raw(**text);
			first = false;
		}
	}
	writer.end_object();
	writer.end_object();
	JsonKey(writer, "gpus");
	writer.start_array();
	if (gpu)
	{
		writer.start_object();
		JsonKey(writer, "name", true);
		writer.append_raw(*gpu);
		writer.end_object();
	}
	writer.end_array();
	if (driver)
	{
		JsonKey(writer, "driver");
		writer.start_object();
		JsonKey(writer, "kernel_module", true);
		writer.start_object();
		JsonKey(writer, "version", true);
		writer.append_raw(*driver);
		writer.end_object();
		writer.end_object();
	}
	writer.end_object();
	return JsonText(writer);
}

std::optional<std::size_t> SelectionLineBytes(std::string_view text)
{
	const auto end = text.substr(0, kMaxSelectionLineBytes).find('\n');
	if (text.compare(0, std::size(kSelectionKey), kSelectionKey) != 0 || end == std::string_view::npos ||
		text[end - 1] != ',')
	{
		return std::nullopt;
	}
	return end + 1;
}

Result<std::string> WithSelection(std::string_view rest, const std::vector<CopyRun>& runs,
								  std::string_view policy, Json request, Json source_machine)
{
	const auto any = [&](std::string_view coverage)
	{
		return std::any_of(runs.begin(), runs.end(),
						   [&](const CopyRun& run) { return run.coverage == coverage; });
	};
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	everything.progressive_pinning = true;
	auto table = SelectTable(runs, policy, request, source_machine, everything);
	auto reference = SelectTable(runs, policy, request, source_machine, everything, false);
	if (!table || !reference || (any("instant") && any("measured")))
	{
		return "{" + std::string(rest);
	}
	const auto rows_json = [](const Table& selected)
	{
		JsonWriter writer;
		writer.start_array();
		for (std::size_t i = 0; i < std::size(selected.rows); i++)
		{
			const Row& row = selected.rows[i];
			writer.append_raw(i == 0 ? "" : ",");
			writer.start_object();
			JsonField(writer, "direction", DirName(row.dir), true);
			JsonField(writer, "min_bytes", row.min_bytes);
			JsonField(writer, "max_bytes", row.max_bytes);
			JsonKey(writer, "settings");
			WriteSettings(writer, row.mode,
						  {static_cast<int>(row.settings.workers), static_cast<int>(row.settings.slots),
						   row.settings.chunk_bytes, row.loop_name, row.settings.wc_slots},
						  row.pinning);
			writer.end_object();
		}
		writer.end_array();
		return JsonText(writer);
	};
	auto rows = rows_json(*table);
	auto reference_rows = rows_json(*reference);
	auto request_text = SelectionRequest(request);
	auto machine_text = SelectionMachine(source_machine);
	for (const auto* part : {&rows, &reference_rows, &request_text, &machine_text})
	{
		if (!*part)
		{
			return Failure(part->error());
		}
	}
	SelectionParts parts;
	parts.policy = policy;
	parts.request = *request_text;
	parts.machine = *machine_text;
	parts.rows = *rows;
	if (*reference_rows != *rows)
	{
		parts.reference_rows = *reference_rows;
	}
	parts.above_pinning_budget = parts.registration_limit =
		table->HasProgressivePinning() || reference->HasProgressivePinning();
	auto line = EncodeSelection(parts, std::size(rest));
	if (!line)
	{
		return Failure(line.error());
	}
	return (std::size(*line) <= kMaxSelectionLineBytes ? *line : "{") + std::string(rest);
}

Result<std::string> RewriteSelection(std::string_view text, std::string_view rest)
{
	const auto line = SelectionLineBytes(text);
	if (!line)
	{
		return UsageError("line 1 is not a selection line");
	}
	const auto value = text.substr(std::size(kSelectionKey), *line - std::size(kSelectionKey) - 2);
	JsonParser parser;
	simdjson::dom::object selection;
	const auto error = parser.parse(std::data(value), std::size(value)).get(selection);
	if (error == simdjson::MEMALLOC)
	{
		Panic("selection JSON allocation failed");
	}
	if (error)
	{
		return UsageError("the selection line is not a JSON object");
	}
	SelectionParts parts;
	std::set<std::string_view> seen;
	for (auto field : selection)
	{
		std::string_view text_value;
		std::uint64_t number = 0;
		const auto raw = simdjson::to_string(field.value);
		bool valid = seen.insert(field.key).second;
		if (field.key == "policy")
		{
			valid = valid && !field.value.get(text_value);
			parts.policy = text_value;
		}
		else if (field.key == "file_bytes")
		{
			valid = valid && !field.value.get(number);
		}
		else if (field.key == "request" || field.key == "source_machine")
		{
			valid = valid && field.value.is_object();
			(field.key == "request" ? parts.request : parts.machine) = raw;
		}
		else if (field.key == "rows" || field.key == "tie_reference_rows")
		{
			valid = valid && field.value.is_array();
			if (field.key == "rows")
			{
				parts.rows = raw;
			}
			else
			{
				parts.reference_rows = raw;
			}
		}
		else if (field.key == kAbovePinningBudget || field.key == kRegistrationLimitKey)
		{
			const bool above = field.key == kAbovePinningBudget;
			valid = valid && !field.value.get(text_value) &&
					text_value == (above ? kUnpinnedChoice : kExplicitOnly);
			(above ? parts.above_pinning_budget : parts.registration_limit) = true;
		}
		else
		{
			valid = false;
		}
		if (!valid)
		{
			return UsageError("the selection line has an unknown, repeated or invalid field");
		}
	}
	if (!seen.count("policy") || !seen.count("file_bytes") || !seen.count("request") ||
		!seen.count("source_machine") || !seen.count("rows"))
	{
		return UsageError("the selection line lacks a valid field");
	}
	auto encoded = EncodeSelection(parts, std::size(rest));
	if (!encoded)
	{
		return Failure(encoded.error());
	}
	if (std::size(*encoded) > kMaxSelectionLineBytes)
	{
		return UsageError("the selection line would exceed kMaxSelectionLineBytes");
	}
	return *encoded + std::string(rest);
}

Result<std::string> Reselect(const std::string& text)
{
	JsonParser parser;
	auto result = ReadTuningResult(parser, text);
	if (!result)
	{
		return Failure(result.error());
	}
	const auto line = SelectionLineBytes(text);
	if (!line && (text.front() != '{' || result->document["selection"].error() != simdjson::NO_SUCH_FIELD))
	{
		return UsageError("tune.json is not laid out as cmt writes it; no selection line can be placed");
	}
	return WithSelection(std::string_view(text).substr(line.value_or(1)), result->runs, result->policy,
						 result->request, result->source_machine);
}

namespace tuning {

Result<Table> ReadSelection(std::string_view text, const LocalCapabilities& local, std::uint64_t& file_bytes)
{
	const auto line = SelectionLineBytes(text);
	if (!line)
	{
		return UsageError("line 1 is not a selection line: an older or reformatted file");
	}
	const auto value = text.substr(std::size(kSelectionKey), *line - std::size(kSelectionKey) - 2);
	JsonParser parser;
	Json selection;
	const auto error = parser.parse(std::data(value), std::size(value)).get(selection);
	if (error == simdjson::MEMALLOC)
	{
		Panic("selection JSON allocation failed");
	}
	std::string_view policy;
	Json request;
	Json source_machine;
	simdjson::dom::array rows;
	simdjson::dom::array reference_rows;
	const auto reference_error = error ? error : selection["tie_reference_rows"].get(reference_rows);
	std::string_view above_pinning_budget;
	const auto above_error = error ? error : selection[kAbovePinningBudget].get(above_pinning_budget);
	std::string_view registration_limit;
	const auto limit_error = error ? error : selection[kRegistrationLimitKey].get(registration_limit);
	if (error || selection["policy"].get(policy) || selection["file_bytes"].get(file_bytes) ||
		selection["request"].get(request) || selection["source_machine"].get(source_machine) ||
		selection["rows"].get(rows) || (reference_error && reference_error != simdjson::NO_SUCH_FIELD) ||
		(above_error && above_error != simdjson::NO_SUCH_FIELD) ||
		(!above_error && above_pinning_budget != kUnpinnedChoice) ||
		(limit_error && limit_error != simdjson::NO_SUCH_FIELD) ||
		(!limit_error && registration_limit != kExplicitOnly))
	{
		return UsageError("the selection line lacks a valid field");
	}
	if (policy != kSelectionPolicyVersion)
	{
		return Unsupported("the selection line was selected under another selection policy");
	}
	if (!UniqueFields(selection) ||
		!OnlyFields(selection, {"policy", "file_bytes", "request", "source_machine", "rows",
								"tie_reference_rows", kAbovePinningBudget, kRegistrationLimitKey}))
	{
		return UsageError("the selection line has a repeated or unknown field");
	}
	Table table;
	CMT_RETURN_IF_ERROR(ReadRequest(table, request, source_machine, local.max_pinned_bytes));
	const auto read_rows = [&](simdjson::dom::array entries, Table& into) -> Result<void>
	{
		std::optional<std::uint64_t> ends[2];
		for (Json entry : entries)
		{
			Row row;
			std::string_view direction;
			std::string_view method;
			Json settings;
			if (!OnlyFields(entry, {"direction", "min_bytes", "max_bytes", "settings"}) ||
				entry["direction"].get(direction) || (direction != "htod" && direction != "dtoh") ||
				entry["min_bytes"].get(row.min_bytes) || entry["max_bytes"].get(row.max_bytes) ||
				entry["settings"].get(settings) || settings["method"].get(method))
			{
				return UsageError("invalid selection row");
			}
			row.dir = direction == "htod" ? Dir::HtoD : Dir::DtoH;
			auto& end = ends[row.dir == Dir::HtoD ? 0 : 1];
			if (row.min_bytes >= row.max_bytes || (end && row.min_bytes < *end))
			{
				return UsageError("selection rows must be nonempty, ascending and not overlap");
			}
			end = row.max_bytes;
			if (method == "staged")
			{
				auto recorded = ParseChunksSettings(settings);
				if (!recorded)
				{
					return Failure(recorded.error());
				}
				if (!LoopRuns(recorded->loop, local.cpu))
				{
					return Unsupported("a selected copy loop does not run natively on this CPU");
				}
				SetStaged(row, *recorded, local.cpu);
			}
			else if (method == "progressive_pinning")
			{
				auto pinning = ParsePinningSettings(settings);
				if (!pinning || !local.progressive_pinning)
				{
					return pinning ? Unsupported("a selected row needs progressive pinning, unavailable here")
								   : Failure(pinning.error());
				}
				row.mode = Mode::ProgressivePinning;
				row.pinning = *pinning;
			}
			else if (method != "driver" || !OnlyFields(settings, {"method"}))
			{
				return UsageError("invalid selection row");
			}
			into.rows.push_back(std::move(row));
		}
		return {};
	};
	Table reference;
	CMT_RETURN_IF_ERROR(read_rows(rows, table));
	if (reference_error == simdjson::SUCCESS)
	{
		CMT_RETURN_IF_ERROR(read_rows(reference_rows, reference));
	}
	if (above_error && (table.HasProgressivePinning() || reference.HasProgressivePinning()))
	{
		return Unsupported("the selection line predates the unpinned rows above a pinning budget; `cmt check "
						   "--write` rewrites it");
	}
	if (limit_error && (table.HasProgressivePinning() || reference.HasProgressivePinning()))
	{
		return Unsupported("the selection line predates registration limits being explicit only; `cmt check "
						   "--write` rewrites it");
	}
	if (std::max(table.MaxPinnedBytes(), reference.MaxPinnedBytes()) > table.max_pinned_bytes)
	{
		return Unsupported("the selected staging pools exceed the local pinned-memory headroom");
	}
	return table;
}

bool HostAllocClearlyFaster(const std::vector<double>& faster, const std::vector<double>& baseline,
							double minimum_gain)
{
	if (std::size(faster) < 2 || std::size(baseline) < 2)
	{
		return false;
	}
	return *std::max_element(std::begin(faster), std::end(faster)) <
			   *std::min_element(std::begin(baseline), std::end(baseline)) &&
		   Median(faster) * (1 + minimum_gain) < Median(baseline);
}

namespace {

template <class T>
void WriteArray(JsonWriter& writer, const std::vector<T>& values)
{
	writer.start_array();
	for (std::size_t i = 0; i < std::size(values); i++)
	{
		if (i != 0)
		{
			writer.append_comma();
		}
		JsonValue(writer, values[i]);
	}
	writer.end_array();
}

void WriteHostAllocArm(JsonWriter& writer, const HostAllocArm& arm)
{
	const auto median = [](const std::vector<double>& values)
	{ return std::empty(values) ? std::nullopt : std::optional<double>(Median(values)); };
	writer.start_object();
	JsonKey(writer, "alloc_seconds", true);
	WriteArray(writer, arm.alloc_seconds);
	JsonKey(writer, "free_seconds");
	WriteArray(writer, arm.free_seconds);
	JsonKey(writer, "huge_kib");
	WriteArray(writer, arm.huge_kib);
	JsonField(writer, "alloc_median_seconds", median(arm.alloc_seconds));
	JsonField(writer, "free_median_seconds", median(arm.free_seconds));
	writer.end_object();
}

Result<HostAllocArm> ReadHostAllocArm(Json value)
{
	HostAllocArm arm;
	simdjson::dom::array alloc_seconds, free_seconds, huge_kib;
	if (value["alloc_seconds"].get(alloc_seconds) || value["free_seconds"].get(free_seconds) ||
		value["huge_kib"].get(huge_kib))
	{
		return UsageError("a host_allocation arm lacks its samples");
	}
	const auto read_seconds = [](simdjson::dom::array samples, std::vector<double>& into) -> Result<void>
	{
		for (Json sample : samples)
		{
			double seconds = 0;
			if (sample.get(seconds) || !std::isfinite(seconds) || seconds < 0)
			{
				return UsageError("a host_allocation sample is not a finite nonnegative number of seconds");
			}
			into.push_back(seconds);
		}
		return {};
	};
	CMT_RETURN_IF_ERROR(read_seconds(alloc_seconds, arm.alloc_seconds));
	CMT_RETURN_IF_ERROR(read_seconds(free_seconds, arm.free_seconds));
	for (Json sample : huge_kib)
	{
		std::uint64_t kib = 0;
		if (sample.get(kib))
		{
			return UsageError("a host_allocation huge_kib sample is not a count");
		}
		arm.huge_kib.push_back(kib);
	}
	if (std::empty(arm.alloc_seconds) || std::size(arm.free_seconds) != std::size(arm.alloc_seconds) ||
		std::size(arm.huge_kib) != std::size(arm.alloc_seconds))
	{
		return UsageError("a host_allocation arm needs one alloc, free and huge_kib sample per process");
	}
	return arm;
}

} // namespace

void WriteHostAllocation(JsonWriter& writer, const HostAllocation& section)
{
	writer.start_object();
	JsonField(writer, "state", section.state, true);
	if (section.state != "measured")
	{
		JsonField(writer, "reason", section.reason);
		writer.end_object();
		return;
	}
	JsonField(writer, "bytes", section.bytes);
	JsonField(writer, "threads", section.threads);
	JsonField(writer, "thp", section.thp);
	JsonKey(writer, "driver");
	WriteHostAllocArm(writer, section.driver);
	JsonKey(writer, "shared");
	WriteHostAllocArm(writer, section.shared);
	JsonField(writer, "private_state", section.private_state);
	const bool private_measured = section.private_state == "measured";
	JsonKey(writer, "private");
	if (private_measured)
	{
		WriteHostAllocArm(writer, section.private_memory);
	}
	else
	{
		writer.append_null();
	}
	JsonKey(writer, "decision");
	writer.start_object();
	JsonField(writer, "shared",
			  HostAllocClearlyFaster(section.shared.alloc_seconds, section.driver.alloc_seconds), true);
	JsonField(writer, "private",
			  private_measured &&
				  HostAllocClearlyFaster(section.private_memory.alloc_seconds, section.driver.alloc_seconds));
	writer.end_object();
	writer.end_object();
}

Result<std::optional<HostAllocation>> ReadHostAllocation(Json document)
{
	Json value;
	const auto missing = document["host_allocation"].get(value);
	if (missing == simdjson::NO_SUCH_FIELD)
	{
		return std::optional<HostAllocation>();
	}
	const auto invalid = [] { return UsageError("tune.json's host_allocation is not as cmt writes it"); };
	std::string_view state;
	if (missing || !value.is_object() || value["state"].get(state))
	{
		return invalid();
	}
	HostAllocation section;
	section.state = state;
	if (state == "not_measured")
	{
		std::string_view reason;
		if (value["reason"].get(reason) ||
			std::none_of(std::begin(kHostAllocReasons), std::end(kHostAllocReasons),
						 [&](const char* known) { return reason == known; }))
		{
			return invalid();
		}
		section.reason = reason;
	}
	else if (state == "measured")
	{
		std::uint64_t threads = 0;
		Json thp;
		Json driver_samples;
		Json shared_samples;
		Json private_samples;
		std::string_view private_state;
		if (value["bytes"].get(section.bytes) || section.bytes == 0 || value["threads"].get(threads) ||
			threads < 1 || threads > 64 || value["thp"].get(thp) || (!thp.is_null() && !thp.is_string()) ||
			value["driver"].get(driver_samples) || value["shared"].get(shared_samples) ||
			value["private_state"].get(private_state) || value["private"].get(private_samples) ||
			(private_state != "measured" && private_state != "not_allowed" && private_state != "failed") ||
			(private_state == "measured") == private_samples.is_null())
		{
			return invalid();
		}
		section.threads = static_cast<unsigned>(threads);
		std::string_view thp_value;
		if (!thp.get(thp_value))
		{
			section.thp = std::string(thp_value);
		}
		section.private_state = private_state;
		auto driver = ReadHostAllocArm(driver_samples);
		auto shared = ReadHostAllocArm(shared_samples);
		if (!driver || !shared)
		{
			return invalid();
		}
		section.driver = std::move(*driver);
		section.shared = std::move(*shared);
		if (private_state == "measured")
		{
			auto measured = ReadHostAllocArm(private_samples);
			if (!measured)
			{
				return invalid();
			}
			section.private_memory = std::move(*measured);
		}
	}
	else
	{
		return invalid();
	}
	JsonWriter writer;
	WriteHostAllocation(writer, section);
	auto rewritten = FinishJson(writer);
	JsonParser parser;
	Json canonical;
	if (!rewritten || parser.parse(*rewritten).get(canonical) ||
		simdjson::to_string(canonical) != simdjson::to_string(value))
	{
		return invalid();
	}
	return std::optional<HostAllocation>(std::move(section));
}

} // namespace tuning
} // namespace cmt
