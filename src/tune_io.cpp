#include "tuner_internal.h"
#include "share.h"
#include <picosha2.h>

namespace cmt {
using namespace tune_detail;

namespace {
constexpr std::size_t kHashReadBytes = 64 * kKiB;

Result<std::string> HashFile(const std::string& path)
{
	const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
	if (fd < 0)
	{
		return OsError("tune: cannot open hash input");
	}
	std::FILE* stream = fdopen(fd, "rb");
	if (!stream)
	{
		const int code = errno;
		close(fd);
		return OsError("tune: cannot open hash stream", code);
	}
	std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(stream, &std::fclose);
	picosha2::hash256_one_by_one hasher;
	std::array<unsigned char, kHashReadBytes> buffer;
	for (;;)
	{
		errno = 0;
		const auto count = std::fread(buffer.data(), 1, buffer.size(), file.get());
		const int code = errno;
		hasher.process(buffer.data(), buffer.data() + count);
		if (std::ferror(file.get()))
		{
			if (code == EINTR)
			{
				std::clearerr(file.get());
				continue;
			}
			return OsError("tune: cannot read hash input", code ? code : EIO);
		}
		if (count < buffer.size())
		{
			break;
		}
	}
	if (std::fclose(file.release()) != 0)
	{
		return OsError("tune: cannot close hash input");
	}
	hasher.finish();
	return picosha2::get_hash_hex_string(hasher);
}

Result<std::string> RequiredJson(Json object, std::string_view pointer)
{
	Json value;
	const auto error = object.at_pointer(pointer).get(value);
	if (error != simdjson::SUCCESS)
	{
		return Failure(Error::Kind::Usage, "missing machine identity field", static_cast<int>(error));
	}
	JsonWriter writer;
	WriteSortedJson(writer, value);
	return FinishJson(writer);
}

Result<std::string> EncodeTuning(const TuningData& data)
{
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.tuning", true);
	JsonField(writer, "version", 1);
	JsonField(writer, "state", data.state);
	JsonField(writer, "matrix_state", data.matrix_state);
	JsonField(writer, "selection_policy", kSelectionPolicyVersion);
	JsonKey(writer, "request");
	writer.append_raw(data.request);
	JsonKey(writer, "source_machine");
	writer.append_raw(data.source_machine);
	if (data.estimate)
	{
		JsonKey(writer, "estimate");
		writer.start_object();
		JsonField(writer, "headroom", kEstimateHeadroom, true);
		JsonKey(writer, "threads");
		writer.start_object();
		bool first_direction = true;
		for (const auto& [direction, cpu] : data.estimate->cpu_gbps)
		{
			JsonField(writer, DirName(direction),
					  tune::EstimateThreads(cpu, data.estimate->pinned_gbps.at(direction),
											data.estimate->max_threads),
					  first_direction);
			first_direction = false;
		}
		writer.end_object();
		JsonField(writer, "max_threads", data.estimate->max_threads);
		JsonKey(writer, "cpu_copies");
		writer.append_raw(data.estimate->cpu_copies);
		JsonKey(writer, "pinned_copies");
		writer.append_raw(data.estimate->pinned_copies);
		writer.end_object();
	}
	if (data.host_allocation)
	{
		JsonKey(writer, "host_allocation");
		tuning::WriteHostAllocation(writer, *data.host_allocation);
	}
	JsonKey(writer, "identity");
	writer.start_object();
	bool first = true;
	for (const auto& [key, value] : data.identity)
	{
		JsonField(writer, key, value, first);
		first = false;
	}
	writer.end_object();
	JsonKey(writer, "invocations");
	writer.start_array();
	for (const auto& invocation : data.previous_invocations)
	{
		writer.append_raw(invocation);
		writer.append_comma();
	}
	writer.start_object();
	JsonKey(writer, "timings_seconds", true);
	writer.start_object();
	JsonField(writer, "metadata_child_wall", data.metadata_child_wall, true);
	JsonField(writer, "probe_child_wall", data.probe_child_wall);
	JsonField(writer, "measured_children_wall", data.measured_children_wall);
	JsonField(writer, "total_before_publish",
			  std::chrono::duration<double>(Clock::now() - data.invocation_start).count());
	writer.end_object();
	JsonKey(writer, "metadata_child_timings_seconds");
	writer.append_raw(data.metadata_timings);
	JsonKey(writer, "probe_child_timings_seconds");
	writer.append_raw(data.probe_timings);
	JsonField(writer, "child_placement", data.placement_note);
	writer.end_object();
	writer.end_array();
	JsonKey(writer, "attempts");
	writer.start_array();
	first = true;
	for (const auto& raw : data.attempts)
	{
		if (!first)
		{
			writer.append_comma();
		}
		first = false;
		writer.append_raw(raw);
	}
	writer.end_array();
	JsonKey(writer, "runs");
	writer.start_array();
	first = true;
	for (const auto& raw : data.raw_runs)
	{
		if (!first)
		{
			writer.append_comma();
		}
		first = false;
		writer.append_raw(raw);
	}
	writer.end_array();
	writer.end_object();
	auto document = FinishJson(writer);
	if (!document)
	{
		return Failure(document.error());
	}
	JsonParser request_parser;
	JsonParser machine_parser;
	Json request;
	Json machine;
	if (request_parser.parse(data.request).get(request) ||
		machine_parser.parse(data.source_machine).get(machine))
	{
		Panic("tuning request or source machine is not JSON");
	}
	return WithSelection(std::string_view(*document).substr(1), data.runs, kSelectionPolicyVersion, request,
						 machine);
}

} // namespace

Result<std::map<std::string, std::string>> tune::ResumeIdentity(const TuneArgs& options, Json machine)
{
	TuneArgs measured = options;
	measured.grid.tie_percent = 0;
	std::string kept;
	bool tie = false;
	for (std::size_t start = 0; start < std::size(options.tuning_params);)
	{
		const auto end = std::min(options.tuning_params.find(' ', start), std::size(options.tuning_params));
		const auto entry = std::string_view(options.tuning_params).substr(start, end - start);
		start = end + 1;
		if (entry.rfind("tie=", 0) == 0)
		{
			tie = true;
		}
		else if (!entry.empty())
		{
			kept += (kept.empty() ? "" : " ") + std::string(entry);
		}
	}
	measured.tuning_params = tie ? kept : options.tuning_params;
	auto request = RequestJson(measured);
	if (!request)
	{
		return Failure(request.error());
	}
	std::map<std::string, std::string> identity;
	identity["request"] = *request;
	for (const auto& [key, path] :
		 {std::pair{"hook_sha256", options.hook}, std::pair{"benchmark_sha256", options.benchmark_exe},
		  std::pair{"cmt_sha256", options.executable}})
	{
		auto hash = HashFile(path);
		if (!hash)
		{
			std::fprintf(stderr, "cmt tune: identity %s: %s\n", key, path.c_str());
			return Failure(hash.error());
		}
		identity[key] = *hash;
	}
	auto fingerprint = share::MachineFingerprint(machine, options.env, options.identity_salt);
	if (!fingerprint)
	{
		return Failure(fingerprint.error());
	}
	identity["fingerprint_salt"] = options.identity_salt;
	identity["machine_fingerprint"] = *fingerprint;
	auto shareable = share::ShareableMachine(machine);
	JsonParser shareable_parser;
	Json recorded;
	if (!shareable || shareable_parser.parse(*shareable).get(recorded))
	{
		return shareable ? UsageError("invalid machine record") : Failure(shareable.error());
	}
	for (const char* pointer :
		 {"/host/kernel", "/host/cpu", "/host/memory/thp_enabled", "/host/memory/thp_defrag", "/driver",
		  "/gpus/0/ordinal", "/gpus/0/name", "/gpus/0/compute_capability", "/gpus/0/gcn_arch",
		  "/gpus/0/compute_units", "/gpus/0/memory_bytes", "/gpus/0/numa_node", "/gpus/0/iommu_group_type",
		  "/gpus/0/has_pcie_link", "/gpus/0/host_link", "/gpus/0/pageable_memory_access"})
	{
		auto field = RequiredJson(recorded, pointer);
		if (!field)
		{
			std::fprintf(stderr, "cmt tune: missing identity field /source_machine%s\n", pointer);
			return Failure(field.error());
		}
		identity[pointer] = *field;
	}
	simdjson::dom::array hops;
	if (recorded.at_pointer("/gpus/0/pcie_path").get(hops))
	{
		return UsageError("machine identity requires a PCIe path array");
	}
	identity["pcie_hop_count"] = std::to_string(hops.size());
	for (std::size_t index = 0; index < hops.size(); ++index)
	{
		for (const char* key : {"max_speed", "max_width", "max_gbps"})
		{
			const auto pointer = "/gpus/0/pcie_path/" + std::to_string(index) + "/" + key;
			auto field = RequiredJson(recorded, pointer);
			if (!field)
			{
				return Failure(field.error());
			}
			identity[pointer] = *field;
		}
	}
	auto affinity = GetThreadAffinity();
	if (!affinity)
	{
		return Failure(affinity.error());
	}
	for (auto cpu : *affinity)
	{
		identity["allowed_cpus"] += std::to_string(cpu) + ",";
	}
	identity["child_placement"] = options.placement.decision;
	return identity;
}

Result<std::string> tune::SavedSalt(const std::string& tune_json)
{
	JsonParser parser;
	std::string_view salt;
	if (parser.parse(tune_json).at_pointer("/identity/fingerprint_salt").get(salt) || std::size(salt) != 32)
	{
		return UsageError("resume identity has no fingerprint salt");
	}
	return std::string(salt);
}

namespace tune_detail {
Result<void> ParseDocument(JsonParser& parser, const std::string& text, Json& value,
						   const std::string& source_path)
{
	const auto error = parser.parse(text).get(value);
	if (error == simdjson::MEMALLOC)
	{
		Panic("tuning document allocation failed");
	}
	if (error != simdjson::SUCCESS)
	{
		std::fprintf(stderr, "cmt tune: %s: %s\n", source_path.c_str(), simdjson::error_message(error));
		return Failure(Error::Kind::Usage, "invalid tuning JSON document", static_cast<int>(error));
	}
	if (!value.is_object())
	{
		std::fprintf(stderr, "cmt tune: %s: root value must be an object\n", source_path.c_str());
		return UsageError("tuning JSON document must be an object");
	}
	return {};
}

Result<void> AddRuns(TuningData& data, Json document, const BenchmarkRun* expected, unsigned expected_samples)
{
	std::string_view schema;
	std::uint64_t version = 0;
	if (document["schema"].get(schema) != simdjson::SUCCESS ||
		document["version"].get(version) != simdjson::SUCCESS || version != 1 ||
		(schema != "cmt.measurements" && schema != "cmt.tuning"))
	{
		return UsageError("unsupported measurement envelope");
	}
	auto parsed = ReadCopyRuns(document);
	if (!parsed)
	{
		return Failure(parsed.error());
	}
	if (expected && parsed->size() != expected->sizes.size())
	{
		return Failure(Error::Kind::Verify, "benchmark must return one result per requested size");
	}
	simdjson::dom::array raw;
	if (document["runs"].get(raw) != simdjson::SUCCESS)
	{
		return UsageError("missing run array");
	}
	std::size_t index = 0;
	for (Json record : raw)
	{
		auto run = (*parsed)[index];
		std::string_view pages;
		if (record["pages_requested"].get(pages) || pages != PagesName(data.pages))
		{
			return UsageError("benchmark caller page request differs from the tuning request");
		}
		if (expected &&
			(run.measurement_class != "primary" || run.timing_boundary != "completed_pageable_copy"))
		{
			return UsageError("tuning requires primary completed-pageable-copy measurements");
		}
		if (expected_samples && run.status == "complete")
		{
			std::uint64_t count = 0;
			if (record["sample_count"].get(count) != simdjson::SUCCESS || count != expected_samples)
			{
				return UsageError("benchmark sample count differs from the request");
			}
		}
		if (expected)
		{
			const auto& want = expected->copy;
			if (run.direction != want.direction || run.bytes != expected->sizes[index] ||
				run.process_round != want.process_round || !SameSettings(run, want) ||
				run.execution != (want.mode == Mode::Driver ? "bare_driver" : "hooked_copy"))
			{
				return Failure(Error::Kind::Verify, "benchmark output differs from the requested fixed copy");
			}
		}
		data.runs.push_back(std::move(run));
		data.raw_runs.push_back(simdjson::to_string(record));
		index++;
	}
	return {};
}

Result<std::string> ChildTimings(Json document, const std::string& source_path)
{
	Json value;
	const auto error = document["timings_seconds"].get(value);
	if (error == simdjson::NO_SUCH_FIELD)
	{
		return std::string("{}");
	}
	simdjson::dom::object timings;
	const auto object_error = error != simdjson::SUCCESS ? error : value.get(timings);
	if (object_error != simdjson::SUCCESS)
	{
		std::fprintf(stderr, "cmt tune: %s: /timings_seconds must be an object\n", source_path.c_str());
		return Failure(Error::Kind::Usage, "invalid child lifecycle timings", static_cast<int>(object_error));
	}
	for (auto field : timings)
	{
		double seconds = 0;
		const auto duration_error = field.value.get(seconds);
		if (duration_error != simdjson::SUCCESS || !std::isfinite(seconds) || seconds < 0)
		{
			std::fprintf(stderr, "cmt tune: %s: timings_seconds[%.*s] = %s\n", source_path.c_str(),
						 static_cast<int>(field.key.size()), field.key.data(),
						 simdjson::to_string(field.value).c_str());
			return Failure(Error::Kind::Usage, "child lifecycle timing must be finite and nonnegative",
						   static_cast<int>(duration_error));
		}
	}
	return simdjson::to_string(value);
}

Result<void> RecordAttempt(TuningData& data, const ProcessCommand& command, const ProcessResult& result,
						   const std::vector<std::string>& artifacts)
{
	std::vector<std::string> arguments;
	for (const auto& argument : command.arguments)
	{
		arguments.push_back(share::ShareablePath(argument, data.out_dir));
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "executable", share::ShareablePath(command.executable, {}), true);
	JsonKey(writer, "arguments");
	WriteJsonStringArray(writer, arguments);
	JsonKey(writer, "environment_overrides");
	WriteJsonStringArray(writer, share::ShareableEnvironment(command.environment, data.out_dir));
	JsonField(writer, "clean_hook_environment", command.clean_hook_environment);
	JsonField(writer, "cancellation_grace_seconds", command.cancellation_grace.count());
	JsonField(writer, "exit_code", result.exit_code);
	JsonField(writer, "signal", result.signal);
	JsonField(writer, "interrupted_by_signal", result.interrupted_by_signal);
	JsonKey(writer, "artifacts");
	writer.start_object();
	bool first = true;
	for (const auto& path : artifacts)
	{
		if (path.empty())
		{
			continue;
		}
		auto hash = HashFile(path);
		if (!hash)
		{
			if (hash.error().GetCode() == ENOENT)
			{
				continue;
			}
			return Failure(hash.error());
		}
		JsonField(writer, share::ShareablePath(path, data.out_dir), *hash, first);
		first = false;
	}
	writer.end_object();
	writer.end_object();
	auto text = FinishJson(writer);
	if (!text)
	{
		return Failure(text.error());
	}
	data.attempts.push_back(*text);
	return {};
}

Result<void> SaveTuning(const std::string& directory, const TuningData& data)
{
	auto text = EncodeTuning(data);
	if (!text)
	{
		return Failure(text.error());
	}
	if (text->size() > tuning::kMaxConfigBytes)
	{
		return UsageError("tune.json exceeds the hook loader limit; reduce the explicit search grid");
	}
	return WriteTextFileAtomic(directory + "/tune.json", *text);
}

bool SameSettings(const CopyRun& a, const CopyRun& b)
{
	if (a.mode != b.mode)
	{
		return false;
	}
	switch (a.mode)
	{
	case Mode::Driver:
		return true;
	case Mode::Chunks:
		return a.settings.workers == b.settings.workers && a.settings.slots == b.settings.slots &&
			   a.settings.slot_bytes == b.settings.slot_bytes && a.settings.loop == b.settings.loop &&
			   a.settings.wc_slots == b.settings.wc_slots;
	case Mode::ProgressivePinning:
		return a.pinning.first_chunk_bytes == b.pinning.first_chunk_bytes &&
			   a.pinning.max_chunk_bytes == b.pinning.max_chunk_bytes &&
			   a.pinning.max_registered_bytes == b.pinning.max_registered_bytes &&
			   a.pinning.read_only == b.pinning.read_only;
	}
	return false;
}

} // namespace tune_detail
} // namespace cmt
