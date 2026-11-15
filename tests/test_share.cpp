// Sharing a tune (share.h, remote.h): what tune.json records of a machine, the resume fingerprint, the selection
// line's copies, `cmt export`, a shared tune's checks, and a server's selection-only tune.
//   test_share                          the cases below
//   test_share --cmt CMT FAKE_LIBDIR    export, check and run (local files, --config URL and --remote_config from a
//                                       local HTTP server) through the cmt binary; FAKE_LIBDIR holds the fake
//                                       libcuda.so.1
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <picosha2.h>

#include "check.h"
#include "config.h"
#include "machine.h"
#include "remote.h"
#include "share.h"
#include "tuner.h"
#include "tuning_fixture.h"

using namespace cmt;
using namespace tuning_fixture;

namespace {

// A machine record as `cmt tune`'s metadata child writes it (cmt-machine 1), with identifying values planted where
// a real machine has them. Each of kPlanted occurs in it, and must not occur in anything cmt records or shares.
constexpr char kPlantedMachine[] = R"({"schema":"cmt-machine 1",
"tool":{"name":"cmt","version":"0.1.0","backend":"cuda","commit":"08f0c7731a494d1e51b008bd20cae030ac774b2c","dirty":false,
 "build_type":"Release","cxx_compiler":"GNU 13.3.0","gpu_compiler":null,"gpu_toolkit":"CUDA 13.1.80"},
"host":{"hostname":"alice-workstation","kernel":{"release":"6.6.36","version":"#1 SMP Fri Oct 23 13:21:06 UTC 2026"},
 "os":"Ubuntu 24.04.3 LTS",
 "cmdline":"BOOT_IMAGE=/boot/vmlinuz root=UUID=822d1c62-d9a6-40cf-963c-92c747100e65 ro intel_iommu=on",
 "virtualization":{"type":"none","evidence":"CPUID hypervisor bit clear and no DMI/sysfs match",
  "cpuid_hypervisor_bit":false,"cpuid_vendor":null,
  "dmi":{"sys_vendor":"Alice Corp","product_name":"SecretBox 9000","board_vendor":"Alice Corp",
   "bios_vendor":"Example Vendor"},"sys_hypervisor_type":null,"device_tree_hypervisor":null},
 "container":{"type":"docker","evidence":"/proc/1/cgroup '0::/docker/c37ebc176c45deadbeef'","pid1_comm":"alice-train"},
 "cpu":{"arch":"x86_64","model":"Test CPU 9000","vendor":"GenuineIntel","threads":16,"sockets":1,"cores":8,"numa_nodes":1,
  "flags":{"avx2":true},"scaling_governor":"powersave","scaling_driver":"intel_pstate"},
 "memory":{"total_bytes":68719476736,"base_page_bytes":4096,"process_available_bytes":34359738368,
  "thp_enabled":"madvise","thp_defrag":"madvise","hugepages":{"2048kB":{"total":0,"free":0}}},
 "iommu":{"cmdline":{"iommu":null,"intel_iommu":"on","amd_iommu":null,"iommu.passthrough":null,"iommu.strict":null},
  "units":["dmar0"],"groups":3}},
"gpus":[{"ordinal":0,"name":"Test GPU","uuid":"GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876","pci_bus_id":"0000:01:00.0",
 "compute_capability":"8.6","gcn_arch":null,"compute_units":82,"memory_bytes":25769803776,"numa_node":0,
 "iommu_group":7,"iommu_group_type":"DMA",
 "pcie_path":[{"bdf":"0000:01:00.0","max_speed":"16.0 GT/s PCIe","max_width":16,"current_speed":"16.0 GT/s PCIe",
   "current_width":16,"max_gbps":31.508},
  {"bdf":"0000:00:01.0","max_speed":"16.0 GT/s PCIe","max_width":16,"current_speed":"16.0 GT/s PCIe",
   "current_width":16,"max_gbps":31.508}],
 "narrowest_link":{"bdf":"0000:01:00.0","link":"16.0 GT/s PCIe x16","gbps":31.508,"current":"16.0 GT/s PCIe x16"},
 "has_pcie_link":true,
 "cpu_affinity":{"nvml_cpus":"0-15","nvml_error":null,"sysfs_numa_node":0,"sysfs_cpus":"0-15",
  "sysfs_source":"sysfs local_cpulist","sysfs_error":null},
 "host_link":{"kind":"pcie","evidence":"integrated=0"},"pageable_memory_access":false}],
"gpu_error":null,
"driver":{"vendor":"nvidia","kernel_module":{"version":"615.71.09","srcversion":"0123456789ABCDEF0123456",
  "banner":"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (alice@workstation)  Wed Jul 29 02:45:17 UTC 2026"},
 "uvm_module":{"version":"615.71.09","srcversion":"FEDCBA9876543210FEDCBA9"},
 "host_registration":{"status":"unknown","large_system_pages_opt_in":null,"mapping_evidence":"not_measured",
  "evidence":"No active cmt patch journal identifies the loaded modules"},
 "libcuda":{"path":"/home/alice/drivers/libcuda.so.615.71.09","version":"615.71.09"},
 "cuda_driver_api":"13.3","cuda_runtime":"13.1"},
"new_section":{"note":"a field added later"}})";

constexpr const char* kPlanted[] = {"alice",		"822d1c62", "Alice Corp",		   "SecretBox",
									"c37ebc176c45", "9d10aa77", "0000:01:00.0",		   "0000:00:01.0",
									"Wed Jul 29",	"/home/",	"a field added later", "LD_LIBRARY_PATH"};

// A request with identifying environment entries, as `cmt tune` wrote them before they were reduced. Open, as
// tuning_fixture's kRequest: Document closes it.
constexpr char kPlantedRequest[] =
	R"({"search":"quick","process_rounds":1,"max_pinned_mib":1024,"samples":2,"grid":"g","environment":)"
	R"(["CMT_WORKER_NUMA=0","LD_LIBRARY_PATH=/home/alice/lib","CUDA_VISIBLE_DEVICES=GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876"])";

constexpr char kSalt[] = "0123456789abcdef0123456789abcdef";

Json Parse(JsonParser& parser, const std::string& text)
{
	Json value;
	CHECK(!parser.parse(text).get(value));
	return value;
}

std::string Leaked(const std::string& text)
{
	std::string leaked;
	for (const char* planted : kPlanted)
	{
		if (text.find(planted) != std::string::npos)
		{
			leaked += std::string(leaked.empty() ? "" : ", ") + planted;
		}
	}
	return leaked;
}

std::string Replaced(std::string text, const std::string& from, const std::string& to)
{
	for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
	{
		text.replace(at, from.size(), to);
	}
	return text;
}

std::string Sha256(const std::string& text)
{
	return picosha2::hash256_hex_string(text.begin(), text.end());
}

// HtoD: the driver wins at 1 MiB, a staged 2 x 2 x 1M layout (20 GB/s) from 64 MiB; DtoH: the driver.
std::vector<CopyRun> ShareRuns()
{
	const StagingSettings staged{2, 2, kMiB, "memcpy", false};
	const auto run = [&](Dir direction, std::uint64_t bytes, Mode mode, double gbps)
	{
		return Run(direction, bytes, mode, gbps, "measured",
				   mode == Mode::Chunks ? staged : StagingSettings{});
	};
	return {run(Dir::HtoD, kMiB, Mode::Driver, 10),		 run(Dir::HtoD, kMiB, Mode::Chunks, 5),
			run(Dir::HtoD, 64 * kMiB, Mode::Driver, 10), run(Dir::HtoD, 64 * kMiB, Mode::Chunks, 20),
			run(Dir::DtoH, kMiB, Mode::Driver, 8),		 run(Dir::DtoH, 64 * kMiB, Mode::Driver, 12)};
}

std::string Tune(const std::string& machine = kPlantedMachine)
{
	return check::Require(Reselect(Document(ShareRuns(), machine, Limit::OldDefault, kPlantedRequest)));
}

// tune.json "host_allocation" (config.h, tuning::HostAllocation) as `cmt tune` writes it: the hook clearly faster
// than the driver, private memory not allowed (THP never).
constexpr char kFasterAllocation[] =
	R"({"state":"measured","bytes":1073741824,"threads":8,"thp":"never",)"
	R"("driver":{"alloc_seconds":[0.39,0.4,0.41],"free_seconds":[0.15,0.15,0.16],"huge_kib":[0,0,0],)"
	R"("alloc_median_seconds":0.4,"free_median_seconds":0.15},)"
	R"("shared":{"alloc_seconds":[0.11,0.12,0.11],"free_seconds":[0.15,0.16,0.15],"huge_kib":[0,0,0],)"
	R"("alloc_median_seconds":0.11,"free_median_seconds":0.15},)"
	R"("private_state":"not_allowed","private":null,"decision":{"shared":true,"private":false}})";
// The same with the hook no faster than the driver.
constexpr char kSlowerAllocation[] =
	R"({"state":"measured","bytes":1073741824,"threads":8,"thp":"never",)"
	R"("driver":{"alloc_seconds":[0.39,0.4,0.41],"free_seconds":[0.15,0.15,0.16],"huge_kib":[0,0,0],)"
	R"("alloc_median_seconds":0.4,"free_median_seconds":0.15},)"
	R"("shared":{"alloc_seconds":[0.39,0.45,0.41],"free_seconds":[0.15,0.16,0.15],"huge_kib":[0,0,0],)"
	R"("alloc_median_seconds":0.41,"free_median_seconds":0.15},)"
	R"("private_state":"not_allowed","private":null,"decision":{"shared":false,"private":false}})";

// Tune() with `section` as its host_allocation.
std::string TuneWith(const std::string& section, const std::string& machine = kPlantedMachine)
{
	const auto document = Document(ShareRuns(), machine, Limit::OldDefault, kPlantedRequest);
	return check::Require(
		Reselect(Replaced(document, R"(,"runs":[)", R"(,"host_allocation":)" + section + R"(,"runs":[)")));
}

LocalCapabilities Everything()
{
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	everything.progressive_pinning = true;
	return everything;
}

// The rows a table holds, as text.
std::string Rows(const Table& table)
{
	std::string rows;
	for (const Row& row : table.rows)
	{
		rows += fmt::format("{} {}..{} {} {}x{}x{} {} {};", DirName(row.dir), row.min_bytes, row.max_bytes,
							ModeName(row.mode), row.settings.workers, row.settings.slots,
							row.settings.chunk_bytes, row.loop_name, row.settings.wc_slots);
	}
	return rows;
}

std::string RowsOf(const std::string& text)
{
	std::uint64_t file_bytes = 0;
	return Rows(check::Require(tuning::ReadSelection(text, Everything(), file_bytes)));
}

// `text` (a share file) with its body edited and file_bytes recomputed, as a careless editor's tool would leave it.
std::string EditedBody(const std::string& text, const std::string& from, const std::string& to)
{
	const auto line = *SelectionLineBytes(text);
	const auto body = Replaced(text.substr(line), from, to);
	return check::Require(RewriteSelection(text, body));
}

// A response of cudamemcpy.com's /api/v1/closest, saved 2026-11-11 with its headers (tests/fixtures/closest): an
// RTX 5090 + Xeon 8481C tune.
std::string ClosestBody()
{
	return check::Require(ReadTextFile(std::string(CMT_TEST_FIXTURES) + "/closest/body.json"));
}

// `text` (a selection-only tune) with `from` replaced by `to` everywhere and file_bytes recomputed.
std::string EditedSelectionOnly(const std::string& text, const std::string& from, const std::string& to)
{
	const auto edited = Replaced(text, from, to);
	const auto recorded = fmt::format("\"file_bytes\":{},", text.size());
	CHECK(edited.find(recorded) != std::string::npos);
	const auto with = [&](std::size_t size)
	{ return Replaced(edited, recorded, fmt::format("\"file_bytes\":{},", size)); };
	auto fixed = with(edited.size());
	while (fixed.size() != with(fixed.size()).size()) // the count gained a digit
	{
		fixed = with(fixed.size());
	}
	return with(fixed.size());
}

} // namespace

// tune.json's source_machine: an allowlist. Every planted identifier is gone, each withheld field is named, a field
// the allowlist does not know (added to the record later) is withheld too, and what selection, the hook and the
// reader need stays. The result is still a complete cmt-machine 1 record and passes the export scan.
TEST(a_machine_record_keeps_only_its_allowlist)
{
	JsonParser parser;
	const auto text = check::Require(share::ShareableMachine(Parse(parser, kPlantedMachine)));
	CHECKF(Leaked(text).empty(), "leaked: %s", Leaked(text).c_str());
	JsonParser kept_parser;
	const Json kept = Parse(kept_parser, text);
	std::vector<std::string> withheld;
	simdjson::dom::array names;
	CHECK(!kept["withheld"].get(names));
	for (Json name : names)
	{
		withheld.emplace_back(std::string_view(name.get_string().value_unsafe()));
	}
	const std::vector<std::string> expected{"host.hostname",
											"host.cmdline",
											"host.virtualization.dmi.sys_vendor",
											"host.virtualization.dmi.product_name",
											"host.virtualization.dmi.board_vendor",
											"host.virtualization.dmi.bios_vendor",
											"host.container.evidence",
											"host.container.pid1_comm",
											"gpus.0.uuid",
											"gpus.0.pci_bus_id",
											"gpus.0.iommu_group",
											"gpus.0.pcie_path.0.bdf",
											"gpus.0.pcie_path.1.bdf",
											"gpus.0.narrowest_link.bdf",
											"new_section.note"};
	CHECK(withheld == expected);
	const auto at = [&](const char* pointer)
	{
		Json value;
		return kept.at_pointer(pointer).get(value) ? std::string("<missing>") : simdjson::to_string(value);
	};
	CHECK_EQ(at("/host/cpu/model"), std::string("\"Test CPU 9000\""));
	CHECK_EQ(at("/gpus/0/name"), std::string("\"Test GPU\""));
	CHECK_EQ(at("/host/iommu/cmdline/intel_iommu"), std::string("\"on\""));
	CHECK_EQ(at("/host/kernel/release"), std::string("\"6.6.36\""));
	CHECK_EQ(at("/gpus/0/narrowest_link/link"), std::string("\"16.0 GT/s PCIe x16\""));
	CHECK_EQ(at("/gpus/0/uuid"), std::string("null"));
	CHECK_EQ(at("/driver/kernel_module/banner"),
			 std::string("\"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09\""));
	CHECK_EQ(at("/driver/libcuda/path"), std::string("\"libcuda.so.615.71.09\""));
	CHECK(MachineMissingKeys(kept).empty());
	CHECK(share::FindSensitive(kept).empty());
}

// A banner kept to its release must not keep a build's user@host: one without a release, or with the build
// details before it, is withheld whole.
TEST(a_banner_that_could_name_a_builder_is_withheld)
{
	for (const char* banner : {"custom module (bob@box) Mon Jan 1", "NVRM version: (bob@box) 1.2.3", ""})
	{
		JsonParser parser;
		const auto machine = fmt::format(R"({{"driver":{{"kernel_module":{{"banner":"{}"}}}}}})", banner);
		const auto text = check::Require(share::ShareableMachine(Parse(parser, machine)));
		CHECKF(text.find("\"banner\":null") != std::string::npos, "%s -> %s", banner, text.c_str());
	}
}

// What the request and each child-process record keep of an environment: ordinals, not device UUIDs; no library
// path; paths relative to the output directory or reduced to their file names.
TEST(recorded_environments_name_no_device_or_directory)
{
	const std::vector<std::string> env{
		"CMT_WORKER_NUMA=0", "CUDA_VISIBLE_DEVICES=GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876,MIG-GPU-x/1/0",
		"LD_LIBRARY_PATH=/home/alice/lib", "CUDA_DEVICE_ORDER=PCI_BUS_ID"};
	CHECK(share::ShareableEnvironment(env) ==
		  (std::vector<std::string>{"CMT_WORKER_NUMA=0", "CUDA_VISIBLE_DEVICES=<2 entries withheld>",
									"CUDA_DEVICE_ORDER=PCI_BUS_ID"}));
	CHECK(share::ShareableEnvironment({"CUDA_VISIBLE_DEVICES=1,3"}) ==
		  std::vector<std::string>{"CUDA_VISIBLE_DEVICES=1,3"});
	CHECK(share::ShareableEnvironment({"CUDA_VISIBLE_DEVICES="}) ==
		  std::vector<std::string>{"CUDA_VISIBLE_DEVICES="});
	const std::vector<std::string> child{"CMT_EXPECTED_RUN=/home/alice/out/artifacts/run-1.request.json",
										 "LD_PRELOAD=/home/alice/cmt/libcmt_hook.so",
										 "CMT_MEASUREMENT_LOCK_PATH=/tmp/cmt-measurements-1000.lock",
										 "STAGING_PIPELINE_WORKERS=2"};
	CHECK(share::ShareableEnvironment(child, "/home/alice/out") ==
		  (std::vector<std::string>{
			  "CMT_EXPECTED_RUN=artifacts/run-1.request.json", "LD_PRELOAD=libcmt_hook.so",
			  "CMT_MEASUREMENT_LOCK_PATH=cmt-measurements-1000.lock", "STAGING_PIPELINE_WORKERS=2"}));
	CHECK_EQ(share::ShareablePath("/home/alice/out/artifacts/x.json", "/home/alice/out/"),
			 std::string("artifacts/x.json"));
	CHECK_EQ(share::ShareablePath("tune/artifacts/x.json", "tune"), std::string("artifacts/x.json"));
	CHECK_EQ(share::ShareablePath("/home/alice/out2/x.json", "/home/alice/out"), std::string("x.json"));
	CHECK_EQ(share::ShareablePath("../../home/alice/x.json", ""), std::string("x.json"));
	CHECK_EQ(share::ShareablePath("--sizes", "/home/alice/out"), std::string("--sizes"));
	// A placement message quoting the PCI address (invocations[].child_placement).
	CHECK_EQ(share::WithoutValue("the CPUs of GPU 0000:01:00.0 are unknown; 0000:01:00.0", "0000:01:00.0"),
			 std::string("the CPUs of GPU <withheld> are unknown; <withheld>"));
	CHECK_EQ(share::WithoutValue("unpinned (no node)", ""), std::string("unpinned (no node)"));
}

// tune.json's request: what the children got, reduced as above.
TEST(the_request_records_no_library_path_or_device_uuid)
{
	TuneArgs options;
	options.grid = check::Require(tune::ParseTuningParams(tune::Search::Quick, ""));
	options.env = {"CMT_WORKER_NUMA=0", "CUDA_VISIBLE_DEVICES=GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876",
				   "LD_LIBRARY_PATH=/home/alice/lib"};
	const auto request = check::Require(RequestJson(options));
	CHECKF(Leaked(request).empty(), "leaked: %s", Leaked(request).c_str());
	CHECK(request.find("\"CMT_WORKER_NUMA=0\"") != std::string::npos);
}

// --resume still refuses another GPU, another PCI slot, another libcuda path: they enter a salted fingerprint.
// The identity names none of them, and the same GPU fingerprints differently under another salt (another file).
TEST(resume_compares_a_salted_fingerprint_not_the_identifiers)
{
	auto self = check::Require(SelfExe());
	TuneArgs options;
	options.grid = check::Require(tune::ParseTuningParams(tune::Search::Quick, ""));
	options.hook = options.benchmark_exe = options.executable = self;
	options.env = {"CUDA_VISIBLE_DEVICES=GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876",
				   "LD_LIBRARY_PATH=/home/alice/lib"};
	options.identity_salt = kSalt;
	options.placement.decision = "unpinned";
	const auto identity = [&](const std::string& machine, const TuneArgs& with)
	{
		JsonParser parser;
		return check::Require(tune::ResumeIdentity(with, Parse(parser, machine)));
	};
	const auto original = identity(kPlantedMachine, options);
	std::string values;
	for (const auto& [key, value] : original)
	{
		values += key + "=" + value + "\n";
	}
	CHECKF(Leaked(values).empty(), "leaked: %s", Leaked(values).c_str());
	CHECK(original.at("fingerprint_salt") == kSalt);
	CHECK(original.at("machine_fingerprint").rfind("sha256:", 0) == 0);
	CHECK(identity(kPlantedMachine, options) == original);
	// Only the fingerprint tells these apart, and it does.
	const auto differs_in = [&](const std::map<std::string, std::string>& other)
	{
		std::set<std::string> keys;
		for (const auto& [key, value] : original)
		{
			if (!other.count(key) || other.at(key) != value)
			{
				keys.insert(key);
			}
		}
		return keys;
	};
	const std::set<std::string> fingerprint{"machine_fingerprint"};
	CHECK(differs_in(identity(Replaced(kPlantedMachine, "9d10aa77", "11111111"), options)) == fingerprint);
	CHECK(differs_in(identity(
			  Replaced(kPlantedMachine, "\"pci_bus_id\":\"0000:01:00.0\"", "\"pci_bus_id\":\"0000:17:00.0\""),
			  options)) == fingerprint);
	CHECK(differs_in(identity(Replaced(kPlantedMachine, "0000:00:01.0", "0000:14:01.0"), options)) ==
		  fingerprint);
	CHECK(differs_in(identity(Replaced(kPlantedMachine, "/home/alice/drivers", "/opt/drivers"), options)) ==
		  fingerprint);
	auto moved = options;
	moved.env.back() = "LD_LIBRARY_PATH=/home/alice/other";
	CHECK(differs_in(identity(kPlantedMachine, moved)) == fingerprint);
	auto other_file = options;
	other_file.identity_salt = "fedcba9876543210fedcba9876543210";
	CHECK(differs_in(identity(kPlantedMachine, other_file)) ==
		  (std::set<std::string>{"fingerprint_salt", "machine_fingerprint"}));
	// A record without the GPU's UUID cannot be fingerprinted, as it could not be identified before.
	JsonParser parser;
	CHECK(!tune::ResumeIdentity(options, Parse(parser, Replaced(kPlantedMachine, "\"uuid\"", "\"uuid_\""))));
	const auto salt = check::Require(share::NewSalt());
	CHECK(salt.size() == 32 && salt.find_first_not_of("0123456789abcdef") == std::string::npos);
	CHECK(check::Require(share::NewSalt()) != salt);
	CHECK(check::Require(
			  tune::SavedSalt(fmt::format(R"({{"identity":{{"fingerprint_salt":"{}"}}}})", kSalt))) == kSalt);
	CHECK(!tune::SavedSalt(R"({"identity":{"/gpus/0/uuid":"x"}})"));
}

// The selection line copies only what selection and the hook read: no hostname, no library path, no request
// field selection ignores; and the table it records is the one every run selects.
TEST(the_selection_line_copies_what_selection_reads)
{
	const auto text = Tune();
	const auto line = text.substr(0, *SelectionLineBytes(text));
	CHECKF(Leaked(line).empty(), "leaked: %s", Leaked(line).c_str());
	CHECK(line.find(R"("source_machine":{"host":{"cpu":{"model":"Test CPU 9000","arch":"x86_64"}},)"
					R"("gpus":[{"name":"Test GPU"}],"driver":{"kernel_module":{"version":"615.71.09"}}})") !=
		  std::string::npos);
	CHECK(line.find(R"("request":{"search":"quick","process_rounds":1,"max_pinned_mib":1024,)"
					R"("environment":["CMT_WORKER_NUMA=0"]})") != std::string::npos);
	CHECK_EQ(RowsOf(text), Rows(check::Require(ParseTuningResult(text, Everything()))));
	CHECK(check::Require(Reselect(text)) == text);
}

// The sensitive-string scan: each kind of identifier is found; the values cmt records are not.
TEST(the_scan_finds_identifiers_and_passes_recorded_values)
{
	const auto finds = [](const char* key, const std::string& value)
	{
		JsonWriter writer;
		writer.start_object();
		JsonField(writer, key, value, true);
		writer.end_object();
		JsonParser parser;
		return !share::FindSensitive(Parse(parser, check::Require(FinishJson(writer)))).empty();
	};
	for (const char* value :
		 {"GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876", "alice@example.com",
		  "Release Build (dvs-builder@U22-I3-AF05)", "192.168.1.10", "at 10.0.0.1:8080", "fe80::1",
		  "2001:0db8:85a3:0000:0000:8a2e:0370:7334", "00:1a:2b:3c:4d:5e", "0000:01:00.0", "/home/alice/x",
		  "/usr/lib/libcuda.so", "~/tunes", "C:\\Users\\bob", "see ../home/bob/x"})
	{
		CHECKF(finds("note", value), "not found: %s", value);
	}
	for (const char* value : {"memcpy", "32.0 GT/s PCIe", "610.57.04", "13.3", "NVIDIA GeForce RTX 3090",
							  "Intel(R) Xeon(R) Platinum 8481C", "AMD Ryzen 7 7800X3D 8-Core Processor",
							  "x86_64", "madvise", "02:45:17", "0-55,112-167", "1.2.3", "full_with_kernel",
							  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "@", "a@"})
	{
		CHECKF(!finds("note", value), "found: %s", value);
	}
	// A copy loop pins its ISA with '@', in its own field only.
	CHECK(!finds("cpu_loop", "nt_store@avx512") && !finds("cpu_loop", "nt_store@sve"));
	CHECK(finds("note", "nt_store@avx512") && finds("cpu_loop", "alice@box"));
	JsonParser parser;
	CHECK(!share::FindSensitive(Parse(parser, R"({"alice@box":1})")).empty()); // keys too
}

// `cmt export`: exactly the allowlist (tunes/README.md), from a tune.json whose body still holds every planted
// identifier, as one written by an older cmt does; the per-size summary has the medians selection compared.
TEST(export_writes_the_allowlist_and_nothing_else)
{
	const auto tune = Tune();
	const auto shared = check::Require(share::Export(tune));
	CHECKF(Leaked(shared).empty(), "leaked: %s", Leaked(shared).c_str());
	JsonParser parser;
	const Json document = Parse(parser, shared);
	std::set<std::string> keys;
	simdjson::dom::object fields;
	CHECK(!document.get(fields));
	for (auto field : fields)
	{
		keys.insert(std::string(field.key));
	}
	CHECK(keys ==
		  (std::set<std::string>{"selection", "schema", "tool", "machine", "summary", "source_sha256"}));
	const auto at = [&](const char* pointer)
	{
		Json value;
		return document.at_pointer(pointer).get(value) ? std::string("<missing>")
													   : simdjson::to_string(value);
	};
	CHECK_EQ(at("/schema"), std::string("\"cmt.tune-share/1\""));
	CHECK_EQ(at("/tool"),
			 std::string(
				 R"({"version":"0.1.0","commit":"08f0c7731a494d1e51b008bd20cae030ac774b2c","dirty":false})"));
	CHECK_EQ(at("/machine"),
			 std::string(
				 R"({"cpu":{"arch":"x86_64","model":"Test CPU 9000","sockets":1,"cores":8,"numa_nodes":1},)"
				 R"("gpu":{"name":"Test GPU","compute_capability":"8.6","memory_bytes":25769803776},)"
				 R"("link":{"kind":"pcie","speed":"16.0 GT/s PCIe","width":16},)"
				 R"("driver":{"version":"615.71.09","cuda_driver_api":"13.3"},"host_registration":"unknown",)"
				 R"("thp":{"enabled":"madvise","defrag":"madvise"}})"));
	CHECK_EQ(at("/source_sha256"), "\"" + Sha256(tune) + "\"");
	// direction, bytes, driver GB/s, selected GB/s, samples behind the selected rate
	const std::vector<std::tuple<std::string, std::uint64_t, double, double, std::uint64_t>> expected{
		{"htod", kMiB, 10, 10, 2},
		{"htod", 64 * kMiB, 10, 20, 2},
		{"dtoh", kMiB, 8, 8, 2},
		{"dtoh", 64 * kMiB, 12, 12, 2}};
	simdjson::dom::array summary;
	CHECK(!document["summary"].get(summary) && summary.size() == expected.size());
	std::size_t i = 0;
	for (Json entry : summary)
	{
		if (i >= expected.size())
		{
			break;
		}
		const auto& [direction, bytes, driver, selected, n] = expected[i++];
		std::string_view got_direction;
		std::uint64_t got_bytes = 0, got_n = 0;
		double got_driver = 0, got_selected = 0;
		CHECK(!entry["direction"].get(got_direction) && got_direction == direction);
		CHECK(!entry["bytes"].get(got_bytes) && got_bytes == bytes);
		CHECK(!entry["driver_gbps"].get(got_driver) && std::abs(got_driver - driver) < 1e-6);
		CHECK(!entry["selected_gbps"].get(got_selected) && std::abs(got_selected - selected) < 1e-6);
		CHECK(!entry["n"].get(got_n) && got_n == n);
	}
	// The hook reads the same rows from the share as from the tune.json.
	CHECK_EQ(RowsOf(shared), RowsOf(tune));
	const auto info = check::Require(share::Validate(shared));
	CHECK(info.key.gpu == "Test GPU" && info.key.cpu == "Test CPU 9000" && info.key.arch == "x86_64" &&
		  info.key.driver == "615.71.09" && info.tier == "quick");
	CHECK_EQ(share::RepositoryPath(info, shared),
			 "test-gpu--test-cpu-9000--quick-" + Sha256(shared).substr(0, 8) + ".json");
}

// Export refuses a tune that is unfinished, whose line differs from its runs, or that carries an identifier in a
// field the allowlist keeps: the scan, not the allowlist, refuses those.
TEST(export_refuses_partial_stale_or_identifying_tunes)
{
	const auto kind = [](const std::string& tune)
	{
		auto shared = share::Export(tune);
		return shared ? -1 : static_cast<int>(shared.error().GetKind());
	};
	const auto usage = static_cast<int>(Error::Kind::Usage);
	const auto verify = static_cast<int>(Error::Kind::Verify);
	CHECK_EQ(kind(Replaced(Tune(), "\"state\":\"complete\"", "\"state\":\"partial\"")), usage);
	CHECK_EQ(kind(Document(ShareRuns(), kPlantedMachine, Limit::OldDefault, kPlantedRequest)), verify);
	for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
			 {"Test CPU 9000", "alice@workstation"},
			 {"\"Test GPU\"", "\"Test GPU GPU-9d10aa77-6fe5-8e5f-9b6e-bf90bf424876\""},
			 {"\"16.0 GT/s PCIe x16\",\"gbps\"", "\"10.1.2.3 x16\",\"gbps\""},
			 {"\"cuda_driver_api\":\"13.3\"", "\"cuda_driver_api\":\"/home/bob/cuda\""},
			 {"\"thp_enabled\":\"madvise\"", "\"thp_enabled\":\"fe80::1\""},
			 {"\"version\":\"0.1.0\"", "\"version\":\"bob@build\""},
			 {"Test CPU 9000", "gpubox-17.lab.internal"},
			 {"\"Test GPU\"", "\"Test GPU on gpubox-17.lab.internal\""}})
	{
		const auto planted = Replaced(kPlantedMachine, from, to);
		CHECKF(planted != kPlantedMachine, "fixture lacks %s", from.c_str());
		CHECKF(kind(Tune(planted)) == verify, "exported with %s", to.c_str());
	}
}

// A share file is accepted only exactly as cmt writes it: an extra field anywhere, a size that does not match, a
// machine key that differs from the line's, or an identifier in an allowlisted field refuses it.
TEST(a_share_file_is_refused_unless_exactly_as_cmt_writes_it)
{
	const auto shared = check::Require(share::Export(Tune()));
	CHECK(share::Validate(shared));
	CHECK(share::IsShare(shared) && !share::IsShare(Tune()));
	const auto line = *SelectionLineBytes(shared);
	const std::vector<std::string> broken{
		shared + " ",
		EditedBody(shared, "\"schema\"", "\"extra\":1,\"schema\""),
		check::Require(RewriteSelection(shared, "\"extra\":1," + shared.substr(line))),
		Replaced(shared, "\"rows\":", "\"note\":\"x\",\"rows\":"),
		EditedBody(shared, "\"gpu\":{\"name\":\"Test GPU\"", "\"gpu\":{\"name\":\"Other GPU\""),
		EditedBody(shared, "\"speed\":\"16.0 GT/s PCIe\"", "\"speed\":\"10.0.0.1\""),
		EditedBody(shared, "\"n\":2", "\"n\":-2"),
		EditedBody(shared, "\"host_registration\":\"unknown\"", "\"host_registration\":\"maybe\""),
		std::string(share::kMaxShareBytes + 1, ' ')};
	for (std::size_t i = 0; i < broken.size(); i++)
	{
		const auto refused = share::Validate(broken[i]);
		CHECKF(!refused && refused.error().GetKind() == Error::Kind::Verify,
			   "variant %zu not refused (exit 6)", i);
	}
	// A field the line does not have, with file_bytes kept right: only the exact-form check refuses it.
	{
		const std::string field = "\"note\":\"x\",";
		auto edited = Replaced(shared, "\"rows\":", field + "\"rows\":");
		const auto old_bytes = fmt::format("\"file_bytes\":{},", shared.size());
		const auto new_bytes = fmt::format("\"file_bytes\":{},", edited.size());
		CHECK(old_bytes.size() == new_bytes.size());
		edited = Replaced(edited, old_bytes, new_bytes);
		std::uint64_t edited_bytes = 0;
		CHECK(!tuning::ReadSelection(edited, Everything(),
									 edited_bytes)); // the hook refuses an unknown field too
		CHECK(!share::Validate(edited));
	}
	// RewriteSelection refuses a line it does not know whole; it never drops a field.
	CHECK(!RewriteSelection(Replaced(shared, "\"rows\":", "\"note\":\"x\",\"rows\":"), "}"));
}

// A tune's pinned-allocation measurement travels with its share, as written (numbers and fixed words: nothing the
// scan or the planted values find); a share without one (exported before it existed) still validates, and one
// whose decision is not its samples' is refused.
TEST(a_share_carries_the_pinned_allocation_measurement)
{
	const auto tune = TuneWith(kFasterAllocation);
	const auto shared = check::Require(share::Export(tune));
	CHECKF(Leaked(shared).empty(), "leaked: %s", Leaked(shared).c_str());
	JsonParser parser;
	const Json document = Parse(parser, shared);
	Json section;
	CHECK(!document["host_allocation"].get(section));
	CHECK_EQ(simdjson::to_string(section), std::string(kFasterAllocation));
	CHECK(share::FindSensitive(section).empty());
	CHECK(share::Validate(shared));
	const auto read = check::Require(tuning::ReadHostAllocation(document));
	CHECK(read && tuning::HostAllocClearlyFaster(read->shared.alloc_seconds, read->driver.alloc_seconds));
	CHECK_EQ(RowsOf(shared), RowsOf(tune));
	// Old form: no section, as before.
	const auto old_share = check::Require(share::Export(Tune()));
	CHECK(old_share.find("host_allocation") == std::string::npos && share::Validate(old_share));
	CHECK(share::Validate(check::Require(share::Export(TuneWith(kSlowerAllocation)))));
	for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
			 {R"("decision":{"shared":true)", R"("decision":{"shared":false)"},
			 {R"("state":"measured")", R"("state":"measured","host":"gpubox")"},
			 {R"("thp":"never")", R"("thp":"/home/alice")"},
			 {R"("host_allocation":{)", R"("host_allocation":{"x":1,)"}})
	{
		const auto refused = share::Validate(EditedBody(shared, from, to));
		CHECKF(!refused && refused.error().GetKind() == Error::Kind::Verify, "accepted with %s", to.c_str());
	}
	// A tune.json whose section is not as cmt writes it is not exported.
	const auto stale = TuneWith(Replaced(kFasterAllocation, R"("shared":true)", R"("shared":false)"));
	CHECK(!share::Export(stale));
}

// CMT_REMOTE_CONFIG, which only `cmt run --remote_config` sets, is a strict 0/1 switch.
TEST(the_remote_allowance_is_a_strict_switch)
{
	const auto config = [&](const char* value)
	{
		return ParseConfig([&](const char* name) -> const char*
						   { return std::string_view(name) == "CMT_REMOTE_CONFIG" ? value : nullptr; },
						   Everything().cpu);
	};
	CHECK(config(nullptr) && !config(nullptr)->remote_config && config("0") && !config("0")->remote_config);
	CHECK(config("1") && config("1")->remote_config);
	CHECK(!config("2") && config("2").error().IsUsage());
}

// A share has no runs to fall back on: when its line is not usable here, Load reports the line's reason, not
// "expected a cmt.tuning result".
TEST(an_unusable_share_reports_its_line)
{
	const auto shared = check::Require(share::Export(Tune()));
	const std::string path = "test_share_policy.json";
	check::Require(WriteTextFile(path, Replaced(shared, "measured_steps_v4", "measured_steps_v0")));
	const auto table = tuning::Load(path.c_str());
	CHECK(!table && std::string_view(table.error().Message()) ==
						"the selection line was selected under another selection policy");
	std::remove(path.c_str());
}

// The path scheme of tunes/ (tunes/README.md).
TEST(slugs_name_the_repository_paths)
{
	CHECK_EQ(share::Slug("NVIDIA GeForce RTX 3090"), std::string("nvidia-geforce-rtx-3090"));
	CHECK_EQ(share::Slug("Intel(R) Xeon(R) Platinum 8481C"), std::string("intel-r-xeon-r-platinum-8481c"));
	CHECK_EQ(share::Slug("  AMD EPYC 7763 64-Core Processor  "),
			 std::string("amd-epyc-7763-64-core-processor"));
}

// A CPU or GPU model name in a share must be a vendor's: the names of the first six contributed tunes and other
// real ones pass; a DNS-like word or another form does not, in "model" and "name" fields only.
TEST(the_scan_refuses_a_host_name_in_a_model_name)
{
	const auto finds = [](const char* key, const std::string& value)
	{
		JsonWriter writer;
		writer.start_object();
		JsonField(writer, key, value, true);
		writer.end_object();
		JsonParser parser;
		return !share::FindSensitive(Parse(parser, check::Require(FinishJson(writer)))).empty();
	};
	for (const char* model :
		 {"AMD EPYC-Milan Processor", "AMD Ryzen 7 7800X3D 8-Core Processor",
		  "Intel(R) Core(TM) Ultra 9 285K", "Intel(R) Xeon(R) 6787P", "Intel(R) Xeon(R) Platinum 8481C",
		  "NVIDIA A100-SXM4-40GB", "NVIDIA B300 SXM6 AC", "NVIDIA GeForce RTX 3090",
		  "NVIDIA GeForce RTX 5090", "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
		  "AMD EPYC 7763 64-Core Processor", "Tesla V100-SXM2-16GB", "NVIDIA GH200 480GB", "Neoverse-V2",
		  "CMT fake GPU", "Test CPU 9000"})
	{
		CHECKF(!finds("model", model) && !finds("name", model), "refused: %s", model);
	}
	for (const char* model : {"gpubox-17.lab.internal", "Xeon on gpubox-17.lab.internal",
							  "NVIDIA RTX 3090 (host: x)", "AMD  EPYC", " AMD EPYC", "AMD EPYC ", "AMD\tEPYC",
							  "Intel\xc2\xae Xeon", "", "box.example.com", "GPU \"7\""})
	{
		CHECKF(finds("model", model) && finds("name", model), "passed: %s", model);
	}
	CHECK(finds("model", std::string(129, 'A')) && !finds("model", std::string(128, 'A')));
	CHECK(!finds("note", "gpubox-17.lab.internal") && !finds("hostname_like", "AMD  EPYC"));
}

// A server's selection-only tune (cudamemcpy.com /api/v1/closest): accepted as served, its PCI addresses and the
// rest of the site's leftovers included (it is read, never sent); refused when anything selection relies on is off.
TEST(a_selection_only_tune_is_its_line)
{
	const auto body = ClosestBody();
	CHECK(share::IsSelectionOnly(body) && !share::IsShare(body));
	CHECK(!share::IsSelectionOnly(Tune()) && !share::IsSelectionOnly(check::Require(share::Export(Tune()))));
	const auto info = check::Require(share::ValidateSelectionOnly(body));
	CHECK(info.source == "cudamemcpy.com" && info.key.arch == "x86_64" &&
		  info.key.gpu == "NVIDIA GeForce RTX 5090" && info.key.cpu == "Intel(R) Xeon(R) Platinum 8481C");
	const std::string path = "test_share_closest.json";
	check::Require(WriteTextFile(path, body));
	const auto table = tuning::Load(path.c_str()); // the selection line's fast path: there are no runs
	CHECK(table && !(*table)->rows.empty() && (*table)->source_gpu_name == "NVIDIA GeForce RTX 5090");
	std::remove(path.c_str());
	for (const auto& broken :
		 {body + " ", EditedSelectionOnly(body, "\"state\":\"complete\"", "\"state\":\"partial\""),
		  EditedSelectionOnly(body, "\"schema\":\"cmt.tuning\"", "\"schema\":\"cmt.tune-share/1\""),
		  EditedSelectionOnly(body, "\"source\":\"cudamemcpy.com\"", "\"source\":\"x\\u001b[2J\""),
		  EditedSelectionOnly(body, "\"source\":\"cudamemcpy.com\"", "\"origin\":\"cudamemcpy.com\""),
		  EditedSelectionOnly(body, "measured_steps_v4", "measured_steps_v0"),
		  EditedSelectionOnly(body, "\"rows\":[", "\"rows\":[1,"),
		  EditedSelectionOnly(body, "\"version\":1}", "\"version\":2}"),
		  EditedSelectionOnly(body, "\"runs\":[]", "\"runs\":[{}]"),
		  std::string(share::kMaxShareBytes + 1, ' '),
		  // printed by `cmt check` and `cmt run --remote-config`: no terminal escapes, no endless names
		  EditedSelectionOnly(body, "NVIDIA GeForce RTX 5090", "NVIDIA\\u001b[2J"),
		  EditedSelectionOnly(body, "Intel(R) Xeon(R) Platinum 8481C", std::string(129, 'x'))})
	{
		CHECKF(!share::ValidateSelectionOnly(broken), "accepted: %.120s", broken.c_str());
	}
	// Runs make it an ordinary tune.json again: checked against them, not as a server's line.
	CHECK(!share::IsSelectionOnly(EditedSelectionOnly(body, "\"runs\":[]", "\"runs\":[{}]")));
}

// The selection line is read as strictly as the runs: each field once, only the fields cmt writes, and rows that
// are nonempty, ascending and disjoint per direction ([min_bytes, max_bytes)). The fixture's rows are x86_64's.
TEST(the_selection_line_refuses_repeated_unknown_or_overlapping_entries)
{
#if defined(__x86_64__)
	const auto body = ClosestBody();
	std::uint64_t file_bytes = 0;
	CHECK(tuning::ReadSelection(body, Everything(), file_bytes));
	for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
			 {"\"policy\":\"measured_steps_v4\"",
			  "\"policy\":\"measured_steps_v4\",\"policy\":\"measured_steps_v4\""},
			 {"\"rows\":[", "\"note\":1,\"rows\":["},
			 {"{\"direction\":\"htod\"", "{\"note\":1,\"direction\":\"htod\""},
			 {"{\"method\":\"driver\"}", "{\"method\":\"driver\",\"x\":1}"},
			 {"\"method\":\"staged\"", "\"method\":\"staged\",\"x\":1"},
			 {"\"method\":\"staged\"", "\"method\":\"staged\",\"method\":\"staged\""},
			 {"\"max_bytes\":4096,\"min_bytes\":0", "\"max_bytes\":4096,\"min_bytes\":4096"},
			 {"\"max_bytes\":16384,\"min_bytes\":4096", "\"max_bytes\":16384,\"min_bytes\":0"}})
	{
		const auto edited = EditedSelectionOnly(body, from, to);
		CHECKF(!tuning::ReadSelection(edited, Everything(), file_bytes), "accepted: %s", to.c_str());
	}
#endif
}

TEST(url_encoding_keeps_only_unreserved_characters)
{
	CHECK_EQ(UrlEncode("Intel(R) Xeon(R) Platinum 8481C"),
			 std::string("Intel%28R%29%20Xeon%28R%29%20Platinum%208481C"));
	CHECK_EQ(UrlEncode("a&b=c/d?e#f+g"), std::string("a%26b%3Dc%2Fd%3Fe%23f%2Bg"));
	CHECK_EQ(UrlEncode("x86_64 580.95-1~"), std::string("x86_64%20580.95-1~"));
	CHECK_EQ(UrlEncode("\xc3\xa9"), std::string("%C3%A9"));
}

// ---- through the cmt binary ----

namespace {

struct Ran
{
	int status = -1;
	std::string output; ///< stdout and stderr
};

// `argv` run with `env` added, nothing through a shell.
Ran Exec(const std::vector<std::string>& argv, const std::vector<std::string>& env = {})
{
	const std::string log = "test_share_exec.log";
	const pid_t child = fork();
	if (child == 0)
	{
		for (const auto& entry : env)
		{
			const auto equals = entry.find('=');
			setenv(entry.substr(0, equals).c_str(), entry.substr(equals + 1).c_str(), 1);
		}
		const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
		if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0)
		{
			_exit(126);
		}
		std::vector<char*> words;
		for (const auto& word : argv)
		{
			words.push_back(const_cast<char*>(word.c_str()));
		}
		words.push_back(nullptr);
		execvp(words[0], words.data());
		_exit(127);
	}
	Ran ran;
	int status = 0;
	if (child > 0 && waitpid(child, &status, 0) == child)
	{
		ran.status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
	}
	auto text = ReadTextFile(log);
	ran.output = text ? *text : "";
	std::remove(log.c_str());
	return ran;
}

#define CHECK_RAN(ran, expected)                                                                             \
	CHECKF((ran).status == (expected), "exit %d, %d expected:\n%s", (ran).status, (expected),                \
		   (ran).output.c_str())

struct Reply
{
	Reply(int status_code = 200, std::string type = "application/json", std::string text = {},
		  std::string more = {})
		: status(status_code), content_type(std::move(type)), body(std::move(text)), headers(std::move(more))
	{
	}

	int status;
	std::string content_type;
	std::string body;
	std::string headers; ///< more header lines, each ending in \r\n
};

// A local HTTP/1.1 server for one scenario: `routes` by path (the query is ignored), 404 otherwise; each request's
// head appended to `log`. It runs in a child until Stop.
class Server
{
public:
	Server(std::map<std::string, Reply> routes, std::string log) : log_(std::move(log))
	{
		const int listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
		sockaddr_in address{};
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
		socklen_t size = sizeof(address);
		if (listener < 0 || bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
			listen(listener, 16) != 0 ||
			getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0)
		{
			std::perror("test server");
			std::abort();
		}
		port_ = ntohs(address.sin_port);
		std::remove(log_.c_str());
		pid_ = fork();
		if (pid_ == 0)
		{
			prctl(PR_SET_PDEATHSIG,
				  SIGKILL); // a failed check exits without destructors: never outlive the test
			// A client that stops reading early (curl --max-filesize on /huge.json) must not kill the server with
			// SIGPIPE: every later request would then be refused, not answered.
			signal(SIGPIPE, SIG_IGN);
			for (;;)
			{
				const int connection = accept(listener, nullptr, nullptr);
				if (connection < 0)
				{
					continue;
				}
				std::string head;
				char buffer[4096];
				while (head.find("\r\n\r\n") == std::string::npos && head.size() < 65536)
				{
					const ssize_t got = read(connection, buffer, sizeof(buffer));
					if (got <= 0)
					{
						break;
					}
					head.append(buffer, static_cast<std::size_t>(got));
				}
				if (FILE* file = std::fopen(log_.c_str(), "a"))
				{
					std::fputs(head.c_str(), file);
					std::fclose(file);
				}
				const auto target = head.substr(4, head.find(' ', 4) - 4);
				const auto found = routes.find(target.substr(0, target.find('?')));
				const Reply reply =
					found == routes.end() ? Reply{404, "text/plain", "not found"} : found->second;
				const auto response =
					fmt::format("HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}"
								"Connection: close\r\n\r\n",
								reply.status, reply.content_type, reply.body.size(), reply.headers) +
					reply.body;
				for (std::size_t sent = 0; sent < response.size();)
				{
					const ssize_t wrote = write(connection, response.data() + sent, response.size() - sent);
					if (wrote <= 0)
					{
						break;
					}
					sent += static_cast<std::size_t>(wrote);
				}
				close(connection);
			}
		}
		close(listener);
	}

	Server(const Server&) = delete;
	Server& operator=(const Server&) = delete;

	~Server()
	{
		kill(pid_, SIGKILL);
		waitpid(pid_, nullptr, 0);
	}

	std::string Url() const
	{
		return fmt::format("http://127.0.0.1:{}", port_);
	}

	std::string Requests() const
	{
		auto text = ReadTextFile(log_);
		return text ? *text : "";
	}

private:
	std::string log_;
	int port_ = 0;
	pid_t pid_ = -1;
};

bool HasCurl()
{
	return Exec({"curl", "--version"}).status == 0;
}

int CliCommands(const std::string& cmt, const std::string& fake_libdir)
{
	// The children get no proxy: a request to 127.0.0.1 must reach the test server.
	for (const char* name : {"http_proxy", "https_proxy", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"})
	{
		unsetenv(name);
	}
	const std::string dir = "test_share_cli";
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir + "/repo/tunes");
	const auto tune_path = dir + "/tune.json";
	const auto share_path = dir + "/share.json";
	check::Require(WriteTextFile(tune_path, Tune()));

	// export, then check
	CHECK_RAN(Exec({cmt, "export", tune_path, "--out", share_path}), 0);
	const auto shared = check::Require(ReadTextFile(share_path));
	CHECKF(Leaked(shared).empty(), "leaked: %s", Leaked(shared).c_str());
	CHECK_RAN(Exec({cmt, "export", tune_path, "--out", share_path}), 2); // never replaced
	const auto stale_path = dir + "/stale.json";
	check::Require(WriteTextFile(stale_path,
								 Document(ShareRuns(), kPlantedMachine, Limit::OldDefault, kPlantedRequest)));
	CHECK_RAN(Exec({cmt, "export", stale_path, "--out", dir + "/stale-share.json"}), 6);
	const auto checked = Exec({cmt, "check", share_path});
	CHECK_RAN(checked, 0);
	CHECK(checked.output.find("OK: a shared tune") != std::string::npos &&
		  checked.output.find("cannot be re-derived") != std::string::npos);
	CHECK_RAN(Exec({cmt, "check", "--write", share_path}), 2);
	const auto tampered_path = dir + "/tampered.json";
	check::Require(WriteTextFile(tampered_path, EditedBody(shared, "\"schema\"", "\"extra\":1,\"schema\"")));
	CHECK_RAN(Exec({cmt, "check", tampered_path}), 6); // malformed: a failed check, not a usage error
	CHECK_RAN(Exec({cmt, "check", tune_path}), 0);	   // a tune.json still checks against its runs

	// --resume of an output whose identity has no fingerprint salt (an older cmt): refused before any child runs.
	// (The fake libcuda shows one GPU: without any, `cmt tune` stops before this.)
	const auto resumed = Exec({cmt, "tune", "--out", dir, "--resume", "--progress_bar", "off"},
							  {"LD_LIBRARY_PATH=" + fake_libdir});
	CHECK_RAN(resumed, 2);
	CHECK(resumed.output.find("written by an older cmt") != std::string::npos);
	CHECK(!std::filesystem::exists(dir + "/artifacts"));

	// The machine `cmt run` finds under the fake libcuda: this CPU, the fake's GPU name.
	const auto here_model = check::Require(CpuModelName());
#if defined(__aarch64__)
	const std::string here_arch = "aarch64";
#else
	const std::string here_arch = "x86_64";
#endif
	const auto here_machine =
		Replaced(Replaced(Replaced(kPlantedMachine, "\"Test CPU 9000\"", "\"" + here_model + "\""),
						  "\"Test GPU\"", "\"CMT fake GPU\""),
				 "\"arch\":\"x86_64\"", "\"arch\":\"" + here_arch + "\"");
	const auto here_tune = Tune(here_machine);
	const auto here_share = check::Require(share::Export(here_tune));
	const auto here_path = dir + "/here.json";
	check::Require(WriteTextFile(here_path, here_share));
	// Downloads go to this cache, not the user's (an XDG directory must be absolute, or it is ignored).
	const auto cache_home = std::filesystem::absolute(dir + "/cache").string();
	const std::vector<std::string> fake{"LD_LIBRARY_PATH=" + fake_libdir, "XDG_CACHE_HOME=" + cache_home};
	const auto with = [&](std::vector<std::string> env, const std::string& entry)
	{
		env.push_back(entry);
		return env;
	};
	const std::vector<std::string> sees_allowance{"sh", "-c", "test \"$CMT_REMOTE_CONFIG\" = 1"};
	const std::vector<std::string> sees_none{"sh", "-c", "test -z \"$CMT_REMOTE_CONFIG\""};
	const auto run = [&](std::vector<std::string> flags, const std::vector<std::string>& app,
						 const std::vector<std::string>& env)
	{
		flags.insert(flags.begin(), {cmt, "run"});
		flags.push_back("--");
		flags.insert(flags.end(), app.begin(), app.end());
		return Exec(flags, env);
	};

	// cmt run --config FILE: runs only where the hook would use it, and never passes the remote allowance on.
	CHECK_RAN(run({"--config", here_path}, sees_none, fake), 0);
	const std::vector<std::string> sees_debug{"sh", "-c", "test \"$MEMCPY_HOOK_DEBUG\" = 1"};
	CHECK_RAN(run({"--config", here_path, "--debug"}, sees_debug, fake), 0);

	CHECK_RAN(run({"--config", here_path}, sees_none, with(fake, "CMT_REMOTE_CONFIG=1")), 0);
	// --cuda_host_alloc_hook on sets CMT_CUDA_HOST_ALLOC_HOOK=1; off or absent leaves it unset, also when the shell
	// sets it; any other value is a usage error and starts nothing.
	const std::vector<std::string> sees_host_alloc{"sh", "-c", "test \"$CMT_CUDA_HOST_ALLOC_HOOK\" = 1"};
	const std::vector<std::string> sees_no_host_alloc{"sh", "-c",
													  "test -z \"${CMT_CUDA_HOST_ALLOC_HOOK+set}\""};
	CHECK_RAN(run({"--config", here_path, "--cuda_host_alloc_hook", "on"}, sees_host_alloc, fake), 0);
	CHECK_RAN(run({"--config", here_path, "--cuda_host_alloc_hook", "off"}, sees_no_host_alloc,
				  with(fake, "CMT_CUDA_HOST_ALLOC_HOOK=1")),
			  0);
	CHECK_RAN(run({"--config", here_path}, sees_no_host_alloc, with(fake, "CMT_CUDA_HOST_ALLOC_HOOK=1")), 0);
	CHECK_RAN(run({"--config", here_path, "--cuda_host_alloc_hook", "yes"}, {"true"}, fake), 2);
	// A tune that measured the hook clearly faster turns it on (shared memory only); an explicit
	// --cuda_host_alloc_hook wins over it; private memory only when asked; a tune that measured it slower, or
	// whose decision is not its samples', does not.
	const std::vector<std::string> sees_shared_only{
		"sh", "-c",
		"test \"$CMT_CUDA_HOST_ALLOC_HOOK\" = 1 && test -z \"${CMT_CUDA_HOST_ALLOC_PRIVATE+set}\""};
	const std::vector<std::string> sees_private{
		"sh", "-c", "test \"$CMT_CUDA_HOST_ALLOC_HOOK\" = 1 && test \"$CMT_CUDA_HOST_ALLOC_PRIVATE\" = 1"};
	const std::vector<std::string> sees_neither{
		"sh", "-c", "test -z \"${CMT_CUDA_HOST_ALLOC_HOOK+set}${CMT_CUDA_HOST_ALLOC_PRIVATE+set}\""};
	const auto faster_path = dir + "/here-faster.json";
	const auto faster_share_path = dir + "/here-faster-share.json";
	const auto slower_path = dir + "/here-slower.json";
	const auto stale_decision_path = dir + "/here-stale.json";
	check::Require(WriteTextFile(faster_path, TuneWith(kFasterAllocation, here_machine)));
	check::Require(WriteTextFile(faster_share_path,
								 check::Require(share::Export(TuneWith(kFasterAllocation, here_machine)))));
	check::Require(WriteTextFile(slower_path, TuneWith(kSlowerAllocation, here_machine)));
	const auto stale_decision = Replaced(kFasterAllocation, R"("shared":true)", R"("shared":false)");
	check::Require(WriteTextFile(stale_decision_path, TuneWith(stale_decision, here_machine)));
	const auto auto_on = run({"--config", faster_path}, sees_shared_only, fake);
	CHECK_RAN(auto_on, 0);
	CHECKF(auto_on.output.find("pinned allocations use the hook") == std::string::npos &&
			   auto_on.output.find("private huge-page memory") == std::string::npos, "%s",
		   auto_on.output.c_str());
	const auto verbose_on = run({"--config", faster_path, "--verbose"}, sees_shared_only, fake);
	CHECK_RAN(verbose_on, 0);
	CHECKF(verbose_on.output.find("pinned allocations use the hook") != std::string::npos &&
			   verbose_on.output.find("private huge-page memory") == std::string::npos, "%s",
		   verbose_on.output.c_str());
	CHECK_RAN(run({"--config", faster_share_path}, sees_shared_only, fake), 0);
	CHECK_RAN(run({"--config", faster_path, "--cuda_host_alloc_hook", "off"}, sees_neither, fake), 0);
	CHECK_RAN(run({"--config", faster_path, "--cuda_host_alloc_hook", "on"}, sees_shared_only, fake), 0);
	CHECK_RAN(run({"--config", faster_path, "--cuda_host_alloc_hook", "private"}, sees_private, fake), 0);
	CHECK_RAN(run({"--config", here_path, "--cuda_host_alloc_hook", "private"}, sees_private, fake), 0);
	CHECK_RAN(run({"--config", slower_path}, sees_neither, fake), 0);
	CHECK_RAN(run({"--config", here_path}, sees_neither, with(fake, "CMT_CUDA_HOST_ALLOC_PRIVATE=1")), 0);
	CHECK_RAN(run({"--config", stale_decision_path}, {"true"}, fake), 2);
	CHECK_RAN(run({"--config", stale_decision_path, "--cuda_host_alloc_hook", "off"}, sees_neither, fake), 0);
	CHECK_RAN(Exec({cmt, "check", faster_path}), 0);
	CHECK_RAN(Exec({cmt, "check", faster_share_path}), 0);
	CHECK_RAN(Exec({cmt, "check", stale_decision_path}), 6);
	const auto elsewhere = run({"--config", share_path}, {"true"}, fake);
	CHECK_RAN(elsewhere, 3);
	CHECKF(elsewhere.output.find(
			   "was measured on Test GPU / Test CPU 9000 (x86_64), not on this CMT fake GPU / " +
			   here_model) != std::string::npos &&
			   elsewhere.output.find("cmt tune --search quick") != std::string::npos,
		   "%s", elsewhere.output.c_str());
	CHECK_RAN(run({"--config", share_path}, {"true"}, with(fake, "CMT_REMOTE_CONFIG=1")), 3);
	// No GPU (the real driver, or none, with every device hidden): exit 3, nothing started.
	const std::vector<std::string> no_gpu{"CUDA_VISIBLE_DEVICES=", "XDG_CACHE_HOME=" + cache_home};
	const auto hidden = run({"--config", here_path}, {"true"}, no_gpu);
	CHECK_RAN(hidden, 3);
	CHECK(hidden.output.find("no CUDA device visible") != std::string::npos);

	// The combined command loads a completed local tune without measuring, keeps
	// the application's arguments/status, and leaves the saved measurements intact.
	const auto saved_dir = dir + "/saved-tune";
	std::filesystem::create_directories(saved_dir);
	const auto saved_path = saved_dir + "/tune.json";
	check::Require(WriteTextFile(saved_path, here_tune));
	const auto combined = Exec({cmt, "tune", "--out", saved_dir, "--", "sh", "-c",
		"test -n \"$CMT_CONFIG\" && test -n \"$LD_PRELOAD\" && test \"$1\" = --help || exit 99; exit 17",
		"app", "--help"}, fake);
	CHECK_RAN(combined, 17);
	CHECK(combined.output.find("Loaded tuned file:") != std::string::npos);
	CHECK(combined.output.find("Tuning") == std::string::npos);
	CHECK_EQ(check::Require(ReadTextFile(saved_path)), here_tune);
	// The default is shared across working directories, with XDG overriding ~/.cache.
	const auto cached_dir = cache_home + "/cmt";
	std::filesystem::create_directories(cached_dir);
	check::Require(WriteTextFile(cached_dir + "/tune.json", here_tune));
	for (const auto& flags : {std::vector<std::string>{}, std::vector<std::string>{"--verbose", "--debug"}})
	{
		std::vector<std::string> command{cmt, "tune"};
		command.insert(command.end(), flags.begin(), flags.end());
		command.insert(command.end(), {"--", "sh", "-c", "test \"$CMT_CONFIG\" = \"$1\" || exit 99; exit 17",
									 "app", cached_dir + "/tune.json"});
		const auto cached = Exec(command, fake);
		CHECK_RAN(cached, 17);
		CHECK(cached.output.find("Loaded tuned file:") != std::string::npos);
		CHECK(cached.output.find("Tuning") == std::string::npos);
	}
	CHECK_RAN(Exec({cmt, "tune"}, fake), 0); // tune-only also reuses a complete cache
	CHECK_EQ(check::Require(ReadTextFile(cached_dir + "/tune.json")), here_tune);
	CHECK(!std::filesystem::exists(cached_dir + "/artifacts"));
	const auto home = std::filesystem::absolute(dir + "/home").string();
	std::filesystem::create_directories(home + "/.cache/cmt");
	check::Require(WriteTextFile(home + "/.cache/cmt/tune.json", here_tune));
	const std::vector<std::string> home_env{"LD_LIBRARY_PATH=" + fake_libdir, "HOME=" + home,
										 "XDG_CACHE_HOME=relative-is-ignored"};
	const auto from_home = Exec({cmt, "tune", "--", "true"}, home_env);
	CHECK_RAN(from_home, 0);
	CHECK(from_home.output.find(home + "/.cache/cmt/tune.json") != std::string::npos);
	check::Require(WriteTextFile(cached_dir + "/tune.json", Tune(kPlantedMachine)));
	CHECK_RAN(Exec({cmt, "tune", "--", "true"}, fake), 3);
	CHECK_RAN(Exec({cmt, "tune"}, fake), 3);
	check::Require(WriteTextFile(cached_dir + "/tune.json", here_tune));
	// Explicit measurement controls take the resume path instead of silently
	// applying unrelated saved settings; no GPU means no application launch.
	const auto explicit_search = Exec({cmt, "tune", "--out", saved_dir, "--search", "full", "--", "true"}, no_gpu);
	CHECK_RAN(explicit_search, 3);
	CHECK(explicit_search.output.find("Loaded tuned file:") == std::string::npos);
	check::Require(WriteTextFile(saved_path, Tune(kPlantedMachine)));
	CHECK_RAN(Exec({cmt, "tune", "--out", saved_dir, "--", "true"}, fake), 3);
	check::Require(WriteTextFile(saved_path, "{}"));
	CHECK_RAN(Exec({cmt, "tune", "--out", saved_dir, "--", "true"}, fake), 6);

	// A server's selection-only tune, saved locally, is a --config like any other: another machine's is refused.
	const auto closest = ClosestBody();
	const auto closest_path = dir + "/closest.json";
	check::Require(WriteTextFile(closest_path, closest));
	const auto checked_closest = Exec({cmt, "check", closest_path});
	CHECK_RAN(checked_closest, 0);
	CHECK(checked_closest.output.find("OK: selection only (from cudamemcpy.com)") != std::string::npos &&
		  checked_closest.output.find("can't be re-derived") != std::string::npos);
	CHECK_RAN(Exec({cmt, "check", "--write", closest_path}), 2);
	CHECK_RAN(run({"--config", closest_path}, {"true"}, with(fake, "CMT_REMOTE_CONFIG=1")), 3);

	if (!HasCurl())
	{
		std::printf("SKIP: the HTTP cases need curl on PATH\n");
		std::filesystem::remove_all(dir);
		return check::exit_code() != 0 ? check::exit_code() : 77; // ctest: SKIP_RETURN_CODE 77
	}
	// --config URL: one GET, printed first, with nothing sent but three headers; the file lands in the cache.
	const auto here_url = "/tunes/cmt-fake-gpu--test-cpu--quick.json";
	const auto identifying = EditedBody(here_share, "\"speed\":\"16.0 GT/s PCIe\"", "\"speed\":\"10.0.0.1\"");
	const std::string huge(share::kMaxShareBytes + 4096, ' ');
	{
		Server server({{here_url, {200, "application/json", here_share}},
					   {"/elsewhere.json", {200, "application/json", shared}},
					   {"/identifying.json", {200, "application/json", identifying}},
					   {"/tune.json", {200, "application/json", check::Require(ReadTextFile(tune_path))}},
					   {"/huge.json", {200, "application/json", huge}},
					   {"/broken.json", {500, "text/plain", "x"}}},
					  dir + "/requests.log");
		const auto url = server.Url() + here_url;
		const auto ran = run({"--config", url}, sees_none, fake);
		CHECK_RAN(ran, 0);
		CHECK(ran.output.find("cmt: GET " + url + "\n") != std::string::npos);
		const auto requests = server.Requests();
		CHECKF(requests.rfind(std::string("GET ") + here_url + " HTTP/1.1\r\n", 0) == 0, "%s",
			   requests.c_str());
		std::set<std::string> headers;
		for (std::size_t at = 0; (at = requests.find("\r\n", at)) != std::string::npos; at += 2)
		{
			const auto colon = requests.find(':', at);
			if (colon != std::string::npos && colon < requests.find("\r\n", at + 2))
			{
				headers.insert(requests.substr(at + 2, colon - at - 2));
			}
		}
		CHECKF(headers == (std::set<std::string>{"Host", "User-Agent", "Accept"}), "%s", requests.c_str());
		CHECK(requests.find("User-Agent: cmt\r\n") != std::string::npos);
		const auto cached = dir + "/cache/cmt/" + Sha256(url).substr(0, 16) + ".json";
		CHECK(ReadTextFile(cached).value_or("") == here_share);
		const auto checked_url = Exec({cmt, "check", url}, fake);
		CHECK_RAN(checked_url, 0);
		CHECK(checked_url.output.find("OK: a shared tune") != std::string::npos);
		CHECK_RAN(Exec({cmt, "check", "--write", url}, fake), 2);
		// Another machine's: exit 3 naming both; an identifier in it, or not a share: 6; too big, 404, 500: 4.
		const auto other = run({"--config", server.Url() + "/elsewhere.json"}, {"true"}, fake);
		CHECK_RAN(other, 3);
		CHECK(other.output.find("was measured on Test GPU / Test CPU 9000") != std::string::npos);
		CHECK_RAN(run({"--config", server.Url() + "/identifying.json"}, {"true"}, fake), 6);
		CHECK_RAN(run({"--config", server.Url() + "/tune.json"}, {"true"}, fake), 6);
		CHECK_RAN(run({"--config", server.Url() + "/huge.json"}, {"true"}, fake), 4);
		const auto missing = run({"--config", server.Url() + "/missing.json"}, {"true"}, fake);
		CHECK_RAN(missing, 4);
		CHECK(missing.output.find("answered HTTP 404") != std::string::npos);
		CHECK_RAN(run({"--config", server.Url() + "/broken.json"}, {"true"}, fake), 4);
		CHECK(ReadTextFile(cached).value_or("") == here_share); // a failed download leaves the cache alone
		// No GPU: nothing is requested.
		const auto before = server.Requests();
		CHECK_RAN(run({"--config", url}, {"true"}, no_gpu), 3);
		CHECK(server.Requests() == before);
	}
	// http to another host, or any other scheme, is refused before a request.
	CHECK_RAN(run({"--config", "http://example.invalid/x.json"}, {"true"}, fake), 2);
	CHECK_RAN(run({"--config", "http://localhost.example.invalid/x.json"}, {"true"}, fake), 2);
	CHECK_RAN(run({"--remote_config=http://example.invalid"}, {"true"}, fake), 2);
	{
		// The authority is parsed: a user name (`localhost:1@` is user "localhost", password "1") never makes another
		// host local, nor does a trailing dot (a DNS name) or a port that is not a number. Nothing is requested.
		Server server({{"/x.json", {200, "application/json", here_share}},
					   {"/hop.json", {302, "text/plain", "", "Location: http://127.0.0.1:1/x.json\r\n"}}},
					  dir + "/requests.log");
		const auto port = server.Url().substr(server.Url().rfind(':'));
		for (const std::string& url : std::vector<std::string>{
				 "http://localhost:1@127.0.0.1" + port + "/x.json",
				 "http://127.0.0.1" + port + "@example.invalid/x.json",
				 "http://user@127.0.0.1" + port + "/x.json", "http://localhost." + port + "/x.json",
				 "http://127.0.0.1" + port + "x/x.json", "http://[::1]x/x.json", "http:///x.json"})
		{
			CHECK_RAN(run({"--config", url}, {"true"}, fake), 2);
			CHECK_RAN(Exec({cmt, "check", url}, fake), 2);
		}
		CHECK(server.Requests().empty());
		// The scheme is case-insensitive, and so is localhost.
		CHECK_RAN(run({"--config", "HTTP://LOCALHOST" + port + "/x.json"}, sees_none, fake), 0);
		CHECK(!server.Requests().empty());
		// A redirect is followed to https only: http to http, even on this host, is refused by curl.
		const auto hopped = run({"--config", server.Url() + "/hop.json"}, {"true"}, fake);
		CHECK_RAN(hopped, 4);
		CHECK(hopped.output.find("curl exited") != std::string::npos);
	}
	{
		// A user name and password in an https URL are sent (to that host) but never printed.
		const auto ran = run({"--config", "https://someone:s3cret@example.invalid/x.json"}, {"true"}, fake);
		CHECK(ran.status == 4 && ran.output.find("s3cret") == std::string::npos &&
			  ran.output.find("cmt: GET https://***@example.invalid/x.json") != std::string::npos);
		const auto remote = run({"--remote-config=https://someone:s3cret@example.invalid"}, {"true"}, fake);
		CHECK(remote.status == 4 && remote.output.find("s3cret") == std::string::npos);
	}

	// Static remote lookup reuses the website summary and fetches the newest exact model match.
	const auto exact =
		EditedSelectionOnly(EditedSelectionOnly(closest, "Intel(R) Xeon(R) Platinum 8481C", here_model),
							"NVIDIA GeForce RTX 5090", "CMT fake GPU");
	const auto index = [&](const std::string& id, const std::string& cpu = "")
	{
		JsonWriter writer;
		writer.start_object();
		JsonKey(writer, "runs", true);
		writer.start_array();
		for (const auto& run_id : {std::string("older"), id})
		{
			if (run_id != "older")
			{
				writer.append_comma();
			}
			writer.start_object();
			JsonField(writer, "id", run_id, true);
			JsonField(writer, "cpu", cpu.empty() ? here_model : cpu);
			JsonField(writer, "gpu", "CMT fake GPU");
			JsonField(writer, "has_selection", true);
			writer.end_object();
		}
		writer.end_array();
		writer.end_object();
		return check::Require(FinishJson(writer));
	};
	const std::string tune_url = "/data/z-newest/tune.json";
	{
		Server server({{"/data/summary.json", {200, "application/json", index("z-newest")}},
					   {tune_url, {200, "application/json", exact}}},
					  dir + "/requests.log");
		for (const auto& form : {std::vector<std::string>{"--remote-config=" + server.Url()},
								 std::vector<std::string>{"--remote-config", server.Url() + "/"}})
		{
			const auto ran = run(form, sees_allowance, fake);
			CHECK_RAN(ran, 0);
			CHECK(ran.output.find("(exact match)") != std::string::npos);
		}
		CHECK(server.Requests().find("GET /data/summary.json HTTP/1.1") != std::string::npos);
		CHECK(server.Requests().find("GET " + tune_url + " HTTP/1.1") != std::string::npos);
		CHECK(server.Requests().find("?cpu=") == std::string::npos);
		const auto cached = cache_home + "/cmt/" + Sha256(server.Url() + tune_url).substr(0, 16) + ".json";
		CHECK_EQ(check::Require(ReadTextFile(cached)), exact);
		CHECK_RAN(run({"--config", server.Url() + tune_url}, sees_none, fake), 0);
		const auto before = server.Requests();
		CHECK_RAN(run({"--remote-config=" + server.Url()}, {"true"}, no_gpu), 3);
		CHECK(server.Requests() == before);
	}
	for (const auto& body :
		 {closest, EditedSelectionOnly(exact, "\"arch\":\"x86_64\"", "\"arch\":\"riscv64\"")})
	{
		Server server({{"/data/summary.json", {200, "application/json", index("z-newest")}},
					   {tune_url, {200, "application/json", body}}},
					  dir + "/requests.log");
		const auto ran = run({"--remote-config=" + server.Url()}, {"true"}, fake);
		CHECK_RAN(ran, body == closest && here_arch == "x86_64" ? 6 : 3);
	}
	for (const auto& body : {index("z-newest", "another CPU"), std::string("{\"runs\":[]}")})
	{
		Server server({{"/data/summary.json", {200, "application/json", body}}}, dir + "/requests.log");
		CHECK_RAN(run({"--remote-config=" + server.Url()}, {"true"}, fake), 3);
	}
	for (const auto& body : {std::string("<!doctype html>"), index("../escape")})
	{
		Server server({{"/data/summary.json", {200, "application/json", body}}}, dir + "/requests.log");
		CHECK_RAN(run({"--remote-config=" + server.Url()}, {"true"}, fake), 6);
	}
	{
		Server server({{"/data/summary.json", {503, "text/plain", "busy"}}}, dir + "/requests.log");
		CHECK_RAN(run({"--remote-config=" + server.Url()}, {"true"}, fake), 4);
	}
	// Nothing listens: a network error, exit 4.
	std::string closed;
	{
		Server server({}, dir + "/requests.log");
		closed = server.Url();
	}
	CHECK_RAN(run({"--remote_config=" + closed}, {"true"}, fake), 4);
	std::filesystem::remove_all(dir);
	return check::exit_code();
}

} // namespace

int main(int argc, char** argv)
{
	if (argc == 4 && std::string_view(argv[1]) == "--cmt")
	{
		return CliCommands(argv[2], argv[3]);
	}
	return check::run_all();
}
