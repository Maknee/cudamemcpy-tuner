#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include "common/error.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "machine.h"
#include "tests/check.h"

namespace {
struct Sysfs
{
	std::filesystem::path root;

	Sysfs()
	{
		char name[] = "/tmp/cmt-sysfs-test-XXXXXX";
		const char* p = mkdtemp(name);
		if (!p)
		{
			return;
		}
		root = p;
	}

	~Sysfs()
	{
		std::error_code ec;
		std::filesystem::remove_all(root, ec);
	}

	void link_files(const std::filesystem::path& device, const std::string& width)
	{
		std::error_code error;
		std::filesystem::create_directories(device, error);
		CHECK(!error);
		if (error)
		{
			return;
		}
		for (const auto& prefix : {"max", "current"})
		{
			std::ofstream(device / (std::string(prefix) + "_link_speed")) << "16.0 GT/s PCIe\n";
			std::ofstream(device / (std::string(prefix) + "_link_width")) << width << '\n';
		}
	}
};
} // namespace

TEST(pcie_path_follows_the_gpu_and_its_upstream_bridge)
{
	Sysfs fs;
	CHECK(!fs.root.empty());
	if (fs.root.empty())
	{
		return;
	}
	const auto bridge = fs.root / "devices/pci0000:00/0000:00:01.1";
	const auto gpu = bridge / "0000:01:00.0";
	const auto bus = fs.root / "bus/pci/devices";
	fs.link_files(bridge, "8");
	fs.link_files(gpu, "16");
	std::error_code error;
	std::filesystem::create_directories(bus, error);
	CHECK(!error);
	if (error)
	{
		return;
	}
	std::filesystem::create_directory_symlink(gpu, bus / "0000:01:00.0", error);
	CHECK(!error);
	if (error)
	{
		return;
	}
	const auto path = cmt::host::PciePath(bus.string(), "0000:01:00.0");
	CHECK_EQ(std::size(path), std::size_t{2});
	if (std::size(path) != 2)
	{
		return;
	}
	CHECK_EQ(path[0].bdf, std::string("0000:01:00.0"));
	CHECK_EQ(path[1].bdf, std::string("0000:00:01.1"));
	CHECK_EQ(path[0].link_max, std::string("16.0 GT/s PCIe x16"));
	CHECK(cmt::host::LinkGbps(path[0].link_max) > cmt::host::LinkGbps(path[1].link_max));
	CHECK(std::empty(cmt::host::PciePath(bus.string(), "0000:02:00.0")));
}

TEST(thp_policy_and_cpu_list_parsers_preserve_unknown_values)
{
	CHECK(cmt::host::Bracketed("always [madvise] never") == "madvise");
	CHECK(!cmt::host::Bracketed("unreadable").has_value());
	const auto cpus = cmt::host::ParseIndexList("0-3,8,10-11");
	CHECK(cpus.has_value());
	if (cpus)
	{
		CHECK_EQ(std::size(*cpus), std::size_t{7});
	}
	CHECK(!cmt::host::ParseIndexList("3-1").has_value());
}

TEST(machine_json_preserves_schema_and_unknown_facts)
{
	Sysfs fs;
	CHECK(!fs.root.empty());
	if (fs.root.empty())
	{
		return;
	}
	JsonWriter writer;
	writer.start_object();
	cmt::JsonField(writer, "schema", cmt::kMachineSchema, true);
	cmt::JsonKey(writer, "tool");
	cmt::WriteToolJson(writer);
	cmt::JsonKey(writer, "host");
	cmt::host::WriteHostJson(writer, fs.root.string(), cmt::host::Cpuid{false, std::nullopt});
	cmt::JsonKey(writer, "gpus");
	writer.start_array();
	writer.end_array();
	cmt::JsonField(writer, "gpu_error", nullptr);
	cmt::JsonKey(writer, "driver");
	cmt::nv::WriteDriverJson(writer, fs.root.string(), "", std::nullopt, std::nullopt);
	writer.end_object();
	const auto text = check::Require(cmt::FinishJson(writer));
	CHECK(!text.empty() && text.back() == '\n');
	JsonParser parser;
	simdjson::dom::element doc;
	const auto error = parser.parse(text).get(doc);
	CHECK(!error);
	if (error)
	{
		return;
	}
	CHECK(cmt::MachineMissingKeys(doc).empty());
	for (const char* path : {"/host/hostname", "/host/cpu/model", "/host/cpu/threads", "/host/cpu/flags",
							 "/host/memory/total_bytes", "/host/memory/hugepages",
							 "/driver/kernel_module/version", "/driver/cuda_driver_api"})
	{
		simdjson::dom::element value;
		CHECK(!doc.at_pointer(path).get(value) && value.is_null());
	}
	simdjson::dom::array gpus;
	CHECK(!doc["gpus"].get(gpus) && gpus.size() == 0);
	simdjson::dom::element absent;
	CHECK(doc.at_pointer("/driver/patch_status").get(absent) == simdjson::NO_SUCH_FIELD);
}

TEST(machine_json_round_trips_escaped_identity_bytes)
{
	// Control bytes, embedded NUL, quotes, backslashes and UTF-8 must survive.
	std::string identity = "node \"café\"\\rack\n\t";
	identity.append(1, '\0');
	identity += "tail";
	JsonWriter writer;
	writer.start_object();
	cmt::JsonField(writer, "identity", identity, true);
	cmt::JsonField(writer, "false_flag", std::optional<bool>(false));
	cmt::JsonField(writer, "zero", std::optional<std::uint64_t>(0));
	cmt::JsonField(writer, "unknown", std::optional<bool>{});
	writer.end_object();
	const auto text = check::Require(cmt::FinishJson(writer));
	JsonParser parser;
	simdjson::dom::element doc;
	const auto error = parser.parse(text).get(doc);
	CHECK(!error);
	if (error)
	{
		return;
	}
	std::string_view actual;
	CHECK(!doc["identity"].get(actual) && actual == identity);
	bool flag = true;
	CHECK(!doc["false_flag"].get(flag) && !flag);
	std::uint64_t zero = 1;
	CHECK(!doc["zero"].get(zero) && zero == 0);
	simdjson::dom::element unknown;
	CHECK(!doc["unknown"].get(unknown) && unknown.is_null());
}

TEST(machine_json_rejects_invalid_utf8_instead_of_rewriting_identity)
{
	JsonWriter writer;
	writer.start_object();
	cmt::JsonField(writer, "hostname", std::string("node\xff", 5), true);
	writer.end_object();
	CHECK_ERROR_KIND(cmt::FinishJson(writer), Verify);
}

TEST(machine_driver_json_keeps_optional_patch_status_and_version_types)
{
	Sysfs fs;
	CHECK(!fs.root.empty());
	if (fs.root.empty())
	{
		return;
	}
	JsonWriter writer;
	cmt::nv::WriteDriverJson(writer, fs.root.string(), "0000-1000 r-xp 0 00:00 0 /opt/libcuda.so.615.71.09\n",
							 13010, 12080, std::string{});
	const auto text = check::Require(cmt::FinishJson(writer));
	JsonParser parser;
	simdjson::dom::element doc;
	const auto error = parser.parse(text).get(doc);
	CHECK(!error);
	if (error)
	{
		return;
	}
	std::string_view version;
	CHECK(!doc["cuda_driver_api"].get(version) && version == "13.1");
	CHECK(!doc["cuda_runtime"].get(version) && version == "12.8");
	CHECK(!doc.at_pointer("/libcuda/version").get(version) && version == "615.71.09");
	for (const char* field : {"open_kernel", "patched", "set", "patches"})
	{
		simdjson::dom::element value;
		CHECK(!doc["patch_status"][field].get(value) && value.is_null());
	}
	simdjson::dom::array matches;
	CHECK(!doc.at_pointer("/patch_status/matching_sets").get(matches) && matches.size() == 0);
	std::string_view evidence;
	CHECK(!doc.at_pointer("/patch_status/evidence").get(evidence) && !evidence.empty());
}

TEST(machine_pci_json_distinguishes_missing_address_from_absent_link)
{
	Sysfs fs;
	CHECK(!fs.root.empty());
	if (fs.root.empty())
	{
		return;
	}
	for (const bool have_address : {false, true})
	{
		JsonWriter writer;
		writer.start_object();
		const bool linked = cmt::host::WritePciDeviceFields(
			writer, fs.root.string(),
			have_address ? std::optional<std::string>("0000:01:00.0") : std::nullopt, true);
		writer.end_object();
		CHECK(!linked);
		const auto text = check::Require(cmt::FinishJson(writer));
		JsonParser parser;
		simdjson::dom::element doc;
		const auto error = parser.parse(text).get(doc);
		CHECK(!error);
		if (error)
		{
			continue;
		}
		simdjson::dom::element link;
		const auto link_error = doc["has_pcie_link"].get(link);
		CHECK(!link_error);
		if (link_error)
		{
			continue;
		}
		if (have_address)
		{
			bool value = true;
			CHECK(!link.get(value) && !value);
		}
		else
		{
			CHECK(link.is_null());
		}
		simdjson::dom::array path;
		CHECK(!doc["pcie_path"].get(path) && path.size() == 0);
	}
}

TEST(machine_schema_validation_reports_wrong_sections_without_exceptions)
{
	JsonParser parser;
	simdjson::dom::element doc;
	const std::string text = R"({"schema":"other","tool":[],"host":{},"driver":null,"gpus":[1]})";
	const auto error = parser.parse(text).get(doc);
	CHECK(!error);
	if (error)
	{
		return;
	}
	const auto missing = cmt::MachineMissingKeys(doc);
	for (const std::string field : {"schema", "tool", "driver", "gpu_error", "gpus.0", "host.cpu"})
	{
		CHECK(std::find(missing.begin(), missing.end(), field) != missing.end());
	}
}

TEST(machine_pci_json_marks_nonfinite_and_overflowed_rates_unknown)
{
	Sysfs fs;
	CHECK(!fs.root.empty());
	if (fs.root.empty())
	{
		return;
	}
	const auto gpu = fs.root / "sys/bus/pci/devices/0000:01:00.0";
	fs.link_files(gpu, "16");
	// Explicit infinity, NaN, parse overflow, arithmetic overflow, and rounding
	// overflow must all stay out of simdjson's finite-number conversion path.
	for (const std::string speed :
		 {"inf GT/s PCIe", "nan GT/s PCIe", "1e309 GT/s PCIe", "1e308 GT/s PCIe", "1e307 GT/s PCIe"})
	{
		CHECK(cmt::host::LinkGbps(speed + " x16") == 0);
		{
			std::ofstream file(gpu / "max_link_speed");
			file << speed << '\n';
			CHECK(file.good());
		}
		JsonWriter writer;
		cmt::host::WritePciDeviceJson(writer, fs.root.string(), "0000:01:00.0");
		const auto text = check::Require(cmt::FinishJson(writer));
		JsonParser parser;
		simdjson::dom::element doc;
		const auto error = parser.parse(text).get(doc);
		CHECK(!error);
		if (error)
		{
			continue;
		}
		simdjson::dom::element rate;
		CHECK(!doc.at_pointer("/pcie_path/0/max_gbps").get(rate) && rate.is_null());
		simdjson::dom::element narrowest;
		CHECK(!doc["narrowest_link"].get(narrowest) && narrowest.is_null());
		bool linked = true;
		CHECK(!doc["has_pcie_link"].get(linked) && !linked);
		std::string_view recorded_speed;
		CHECK(!doc.at_pointer("/pcie_path/0/max_speed").get(recorded_speed) && recorded_speed == speed);
	}
	const double normal = cmt::host::LinkGbps("16.0 GT/s PCIe x16");
	CHECK(std::isfinite(normal) && std::abs(normal - 31.50769230769231) < 0.000001);
}

using namespace cmt;

namespace {
std::string thp_mode()
{
	std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
	std::string s;
	std::getline(f, s);
	return s; // e.g. "always [madvise] never"
}
} // namespace

TEST(names_round_trip)
{
	for (Pages p : {Pages::Small, Pages::Thp, Pages::Hugetlb})
	{
		CHECK(ParsePages(PagesName(p)) == p);
	}
	CHECK(ParsePages("small") == Pages::Small);
	CHECK_ERROR_KIND(ParsePages("1g"), Usage);
}

TEST(buffers_are_2mib_aligned_and_writable)
{
	for (Pages p : {Pages::Small, Pages::Thp})
	{
		auto b_result = HostBuffer::Create(5u << 20, p);
		CHECK(b_result);
		if (!b_result)
		{
			return;
		}
		auto b = std::move(*b_result);
		CHECK(std::data(b) != nullptr);
		CHECK_EQ(std::size(b), std::size_t{5u << 20});
		CHECK_EQ(reinterpret_cast<std::uintptr_t>(std::data(b)) % ThpBytes(), std::uintptr_t{0});
		std::data(b)[0] = 1;
		std::data(b)[std::size(b) - 1] = 2; // the whole range is mapped
	}
}

TEST(small_pages_are_never_huge)
{
	auto b_result = HostBuffer::Create(8u << 20, Pages::Small);
	CHECK(b_result);
	if (!b_result)
	{
		return;
	}
	auto b = std::move(*b_result);
	CHECK(b.HugeKiB().has_value());
	CHECK_EQ(b.HugeKiB().value_or(-1), 0L);
}

TEST(thp_pages_are_huge_when_the_kernel_allows)
{
	// THP "never" (or a fragmented host) gives 4 KiB pages; huge_kib then reports 0 and the CSV says so.
	// Only when THP is enabled and the first touch got huge pages do we check the count adds up.
	if (thp_mode().find("[never]") != std::string::npos)
	{
		std::printf("  (THP disabled on this host)\n");
		return;
	}
	auto b_result = HostBuffer::Create(8u << 20, Pages::Thp);
	CHECK(b_result);
	if (!b_result)
	{
		return;
	}
	auto b = std::move(*b_result);
	const long k = b.HugeKiB().value_or(-1);
	CHECK(k >= 0);
	CHECK(k <= 8 * 1024);
	CHECK(k % static_cast<long>(ThpBytes() >> 10) == 0);
	std::printf("  (thp: %ld of 8192 KiB on huge pages)\n", k);
}

TEST(hugetlb_without_reserved_pages_is_unsupported_not_a_crash)
{
	// asks for more than any test host reserves; must fail as Unsupported with a hint, never abort
	auto result = HostBuffer::Create(std::size_t{1} << 40, Pages::Hugetlb);
	CHECK(!result);
	if (!result)
	{
		CHECK(result.error().IsUnsupported());
		CHECK(std::string(result.error().Message()).find("nr_hugepages") != std::string::npos);
	}
}

// Lifetime tracking (registry pattern): every mapping made is unmapped exactly once, through moves too.
TEST(no_mapping_leaks_through_moves)
{
	const long base = HostBuffer::Live();
	{
		auto a_result = HostBuffer::Create(1u << 20, Pages::Small);
		CHECK(a_result);
		if (!a_result)
		{
			return;
		}
		auto a = std::move(*a_result);
		CHECK_EQ(HostBuffer::Live(), base + 1);
		HostBuffer b(std::move(a));
		CHECK_EQ(HostBuffer::Live(), base + 1); // moved, not copied
		CHECK(std::data(a) == nullptr);
		auto c_result = HostBuffer::Create(2u << 20, Pages::Small);
		CHECK(c_result);
		if (!c_result)
		{
			return;
		}
		auto c = std::move(*c_result);
		CHECK_EQ(HostBuffer::Live(), base + 2);
		c = std::move(b); // c's old mapping released, b's taken over
		CHECK_EQ(HostBuffer::Live(), base + 1);
		CHECK_EQ(std::size(c), std::size_t{1u << 20});
		HostBuffer d;
		HostBuffer* same =
			&d; // through a pointer: a literal `d = std::move(d)` is (rightly) a -Wself-move error
		d = std::move(*same); // self-move keeps the (empty) state
		CHECK_EQ(HostBuffer::Live(), base + 1);
	}
	CHECK_EQ(HostBuffer::Live(), base);
}

TEST(zero_bytes_is_a_bug_and_huge_is_refused)
{
	CHECK_PANICS((void)HostBuffer::Create(0, Pages::Small));
	CHECK_ERROR_KIND(HostBuffer::Create(std::size_t{1} << 41, Pages::Small),
					 Usage); // above kMaxBytes, before any mmap
}

TEST(page_sizes_come_from_the_kernel)
{
	CHECK(BasePageBytes() >= 4096);
	CHECK((BasePageBytes() & (BasePageBytes() - 1)) == 0);
	CHECK(ThpBytes() >= BasePageBytes());
	const auto avail = HostMemoryAvailable(Pages::Small);
	CHECK(avail.has_value() && *avail > 0);
}

TEST_MAIN()
