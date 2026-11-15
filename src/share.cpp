#include "share.h"
#include <charconv>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <initializer_list>
#include <map>
#include <set>

#include <sys/random.h>

#include <fmt/format.h>
#include <picosha2.h>

#include "arch/copy.h"
#include "common/helpers.h"
#include "config.h"
#include "machine.h"

namespace cmt {
namespace share {
namespace {

enum class Keep : std::uint8_t
{
	Whole,
	Banner,
	Basename
};

struct Rule
{
	std::string_view pattern;
	Keep keep;
};

constexpr Rule kMachineRules[] = {
	{"/schema", Keep::Whole},
	{"/tool", Keep::Whole},
	{"/host/kernel", Keep::Whole},
	{"/host/os", Keep::Whole},
	{"/host/virtualization/type", Keep::Whole},
	{"/host/virtualization/evidence", Keep::Whole},
	{"/host/virtualization/cpuid_hypervisor_bit", Keep::Whole},
	{"/host/virtualization/cpuid_vendor", Keep::Whole},
	{"/host/virtualization/sys_hypervisor_type", Keep::Whole},
	{"/host/virtualization/device_tree_hypervisor", Keep::Whole},
	{"/host/container/type", Keep::Whole},
	{"/host/cpu", Keep::Whole},
	{"/host/memory", Keep::Whole},
	{"/host/iommu", Keep::Whole},
	{"/gpus/*/ordinal", Keep::Whole},
	{"/gpus/*/name", Keep::Whole},
	{"/gpus/*/compute_capability", Keep::Whole},
	{"/gpus/*/gcn_arch", Keep::Whole},
	{"/gpus/*/compute_units", Keep::Whole},
	{"/gpus/*/memory_bytes", Keep::Whole},
	{"/gpus/*/numa_node", Keep::Whole},
	{"/gpus/*/iommu_group_type", Keep::Whole},
	{"/gpus/*/pcie_path/*/max_speed", Keep::Whole},
	{"/gpus/*/pcie_path/*/max_width", Keep::Whole},
	{"/gpus/*/pcie_path/*/current_speed", Keep::Whole},
	{"/gpus/*/pcie_path/*/current_width", Keep::Whole},
	{"/gpus/*/pcie_path/*/max_gbps", Keep::Whole},
	{"/gpus/*/narrowest_link/link", Keep::Whole},
	{"/gpus/*/narrowest_link/gbps", Keep::Whole},
	{"/gpus/*/narrowest_link/current", Keep::Whole},
	{"/gpus/*/has_pcie_link", Keep::Whole},
	{"/gpus/*/cpu_affinity", Keep::Whole},
	{"/gpus/*/host_link", Keep::Whole},
	{"/gpus/*/pageable_memory_access", Keep::Whole},
	{"/gpu_error", Keep::Whole},
	{"/driver/vendor", Keep::Whole},
	{"/driver/kernel_module/version", Keep::Whole},
	{"/driver/kernel_module/srcversion", Keep::Whole},
	{"/driver/kernel_module/banner", Keep::Banner},
	{"/driver/uvm_module", Keep::Whole},
	{"/driver/host_registration", Keep::Whole},
	{"/driver/libcuda/path", Keep::Basename},
	{"/driver/libcuda/version", Keep::Whole},
	{"/driver/cuda_driver_api", Keep::Whole},
	{"/driver/cuda_runtime", Keep::Whole},
	{"/driver/patch_status", Keep::Whole},
};

std::vector<std::string_view> Segments(std::string_view pointer)
{
	std::vector<std::string_view> segments;
	for (std::size_t at = 1; at <= std::size(pointer);)
	{
		const auto end = std::min(pointer.find('/', at), std::size(pointer));
		segments.push_back(pointer.substr(at, end - at));
		at = end + 1;
	}
	return segments;
}

std::optional<Keep> RuleFor(const std::vector<std::string>& path)
{
	for (const Rule& rule : kMachineRules)
	{
		const auto pattern = Segments(rule.pattern);
		bool matches = std::size(pattern) == std::size(path);
		for (std::size_t i = 0; matches && i < std::size(path); i++)
		{
			matches = pattern[i] == "*" || pattern[i] == path[i];
		}
		if (matches)
		{
			return rule.keep;
		}
	}
	return std::nullopt;
}

std::string Pointer(const std::vector<std::string>& path)
{
	std::string pointer;
	for (const auto& segment : path)
	{
		pointer += '/';
		for (char c : segment)
		{
			pointer += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
		}
	}
	return pointer;
}

std::string Dotted(const std::vector<std::string>& path)
{
	std::string dotted;
	for (const auto& segment : path)
	{
		dotted += (dotted.empty() ? "" : ".") + segment;
	}
	return dotted;
}

std::string_view Basename(std::string_view path)
{
	const auto slash = path.rfind('/');
	return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

std::optional<std::string> ShortBanner(std::string_view banner)
{
	const auto release = nv::ReleaseInBanner(std::string(banner));
	const auto at = release ? banner.find(*release) : std::string_view::npos;
	if (at == std::string_view::npos)
	{
		return std::nullopt;
	}
	const auto kept = banner.substr(0, at + std::size(*release));
	if (kept.find_first_of("(@") != std::string_view::npos)
	{
		return std::nullopt;
	}
	return std::string(kept);
}

void WriteMachine(JsonWriter& writer, Json value, std::vector<std::string>& path, std::vector<std::string>& withheld)
{
	if (const auto keep = RuleFor(path))
	{
		std::string_view text;
		if (*keep == Keep::Whole || value.get(text))
		{
			writer.append_raw(*keep == Keep::Whole ? simdjson::to_string(value) : std::string("null"));
			return;
		}
		const auto reduced = *keep == Keep::Banner ? ShortBanner(text) : std::optional<std::string>(Basename(text));
		if (reduced)
		{
			writer.escape_and_append_with_quotes(*reduced);
		}
		else
		{
			writer.append_null();
		}
		return;
	}
	simdjson::dom::object object;
	simdjson::dom::array array;
	if (!value.get(object))
	{
		writer.start_object();
		bool first = true;
		for (auto field : object)
		{
			JsonKey(writer, field.key, first);
			first = false;
			path.emplace_back(field.key);
			WriteMachine(writer, field.value, path, withheld);
			path.pop_back();
		}
		writer.end_object();
	}
	else if (!value.get(array))
	{
		writer.start_array();
		std::size_t index = 0;
		for (Json element : array)
		{
			writer.append_raw(index == 0 ? "" : ",");
			path.push_back(std::to_string(index++));
			WriteMachine(writer, element, path, withheld);
			path.pop_back();
		}
		writer.end_array();
	}
	else
	{
		if (!value.is_null())
		{
			withheld.push_back(Dotted(path));
		}
		writer.append_null();
	}
}

Result<std::string> Text(const JsonWriter& writer)
{
	auto text = FinishJson(writer);
	if (text)
	{
		text->pop_back();
	}
	return text;
}

std::string Sha256(std::string_view bytes)
{
	return picosha2::hash256_hex_string(std::begin(bytes), std::end(bytes));
}

bool IsHex(char c)
{
	return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

bool IsAlpha(char c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool IsDigit(char c)
{
	return c >= '0' && c <= '9';
}

bool UuidAt(std::string_view s, std::size_t at)
{
	constexpr std::size_t kGroups[] = {8, 4, 4, 4, 12};
	for (std::size_t group : kGroups)
	{
		if (at > std::size(s) || std::size(s) - at < group ||
			!std::all_of(std::begin(s) + static_cast<std::ptrdiff_t>(at),
						 std::begin(s) + static_cast<std::ptrdiff_t>(at + group), IsHex))
		{
			return false;
		}
		at += group;
		if (group != 12)
		{
			if (at >= std::size(s) || s[at] != '-')
			{
				return false;
			}
			at++;
		}
	}
	return true;
}

bool Ipv4At(std::string_view s, std::size_t at)
{
	if (at > 0 && (IsDigit(s[at - 1]) || s[at - 1] == '.'))
	{
		return false;
	}
	for (int group = 0; group < 4; group++)
	{
		std::size_t digits = 0;
		unsigned value = 0;
		while (at < std::size(s) && IsDigit(s[at]) && digits < 4)
		{
			value = value * 10 + static_cast<unsigned>(s[at++] - '0');
			digits++;
		}
		if (digits == 0 || digits > 3 || value > 255)
		{
			return false;
		}
		if (group < 3)
		{
			if (at >= std::size(s) || s[at] != '.')
			{
				return false;
			}
			at++;
		}
	}
	return at == std::size(s) || !IsDigit(s[at]);
}

bool HexColonsAt(std::string_view s, std::size_t at)
{
	if (at > 0 && IsHex(s[at - 1]))
	{
		return false;
	}
	std::size_t groups = 0;
	while (at < std::size(s))
	{
		std::size_t digits = 0;
		while (at < std::size(s) && IsHex(s[at]) && digits < 5)
		{
			at++;
			digits++;
		}
		if (digits == 0 || digits > 4)
		{
			break;
		}
		groups++;
		if (at >= std::size(s) || s[at] != ':')
		{
			break;
		}
		at++;
	}
	return groups >= 6;
}

bool CompressedIpv6At(std::string_view s, std::size_t at)
{
	return s.compare(at, 2, "::") == 0 &&
		   ((at > 0 && IsHex(s[at - 1])) || (at + 2 < std::size(s) && IsHex(s[at + 2])));
}

bool PciAt(std::string_view s, std::size_t at)
{
	constexpr std::string_view kShape = "xxxx:xx:xx.d";
	if (std::size(s) - at < std::size(kShape))
	{
		return false;
	}
	for (std::size_t i = 0; i < std::size(kShape); i++)
	{
		const char c = s[at + i];
		if ((kShape[i] == 'x' && !IsHex(c)) || (kShape[i] == 'd' && (c < '0' || c > '7')) ||
			(kShape[i] != 'x' && kShape[i] != 'd' && c != kShape[i]))
		{
			return false;
		}
	}
	return true;
}

bool WordChar(char c)
{
	return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '.' || c == '_' || c == '-' || c == '+' ||
		   c == '%';
}

bool LoopName(std::string_view key, std::string_view value)
{
	return key == "cpu_loop" && (ParseLoopName(Arch::X86_64, value) || ParseLoopName(Arch::Aarch64, value));
}

bool DnsLike(std::string_view word)
{
	const auto last = word.rfind('.');
	if (last == std::string_view::npos || last + 2 >= std::size(word))
	{
		return false;
	}
	for (std::size_t start = 0; start <= last;)
	{
		const auto end = std::min(word.find('.', start), std::size(word));
		const auto label = word.substr(start, end - start);
		if (std::empty(label) || !std::all_of(std::begin(label), std::end(label), [](char c)
											  { return (c >= '0' && c <= '9') || c == '-' || IsAlpha(c); }))
		{
			return false;
		}
		start = end + 1;
	}
	const auto tld = word.substr(last + 1);
	return std::all_of(std::begin(tld), std::end(tld), IsAlpha);
}

const char* ModelKind(std::string_view s)
{
	constexpr std::string_view kPunctuation = "()+,./@_-";
	const bool plain =
		!std::empty(s) && std::size(s) <= 128 && s.front() != ' ' && s.back() != ' ' &&
		s.find("  ") == std::string_view::npos &&
		std::all_of(std::begin(s), std::end(s), [&](char c)
					{ return IsAlpha(c) || IsDigit(c) || c == ' ' || kPunctuation.find(c) != std::string_view::npos; });
	if (!plain)
	{
		return "not a plain CPU or GPU model name";
	}
	for (std::size_t start = 0; start < std::size(s);)
	{
		const auto end = std::min(s.find(' ', start), std::size(s));
		if (DnsLike(s.substr(start, end - start)))
		{
			return "a host name";
		}
		start = end + 1;
	}
	return nullptr;
}

const char* SensitiveKind(std::string_view s, std::string_view key)
{
	if (!std::empty(s) && (s.front() == '/' || s.front() == '~'))
	{
		return "an absolute or home path";
	}
	for (std::string_view home : {"/home/", "/Users/", "/root/", "\\Users\\", "~/"})
	{
		if (s.find(home) != std::string_view::npos)
		{
			return "a home path";
		}
	}
	for (std::size_t at = 0; at < std::size(s); at++)
	{
		if (UuidAt(s, at))
		{
			return "a UUID";
		}
		if (s[at] == '@' && at > 0 && at + 1 < std::size(s) && WordChar(s[at - 1]) && WordChar(s[at + 1]) &&
			!LoopName(key, s))
		{
			return "an e-mail address or user@host";
		}
		if (IsDigit(s[at]) && Ipv4At(s, at))
		{
			return "an IPv4 address";
		}
		if (HexColonsAt(s, at) || CompressedIpv6At(s, at))
		{
			return "an IPv6 or MAC address";
		}
		if (PciAt(s, at))
		{
			return "a PCI address";
		}
	}
	return key == "model" || key == "name" ? ModelKind(s) : nullptr;
}

void Scan(Json value, std::vector<std::string>& path, std::string_view key, std::vector<Finding>& findings)
{
	simdjson::dom::object object;
	simdjson::dom::array array;
	std::string_view text;
	if (!value.get(object))
	{
		for (auto field : object)
		{
			if (const char* kind = SensitiveKind(field.key, {}))
			{
				findings.push_back({Pointer(path), kind, std::string(field.key)});
			}
			path.emplace_back(field.key);
			Scan(field.value, path, field.key, findings);
			path.pop_back();
		}
	}
	else if (!value.get(array))
	{
		std::size_t index = 0;
		for (Json element : array)
		{
			path.push_back(std::to_string(index++));
			Scan(element, path, key, findings);
			path.pop_back();
		}
	}
	else if (!value.get(text))
	{
		if (const char* kind = SensitiveKind(text, key))
		{
			findings.push_back({Pointer(path), kind, std::string(text)});
		}
	}
}

bool PrintFindings(const std::vector<Finding>& findings)
{
	for (const auto& finding : findings)
	{
		fmt::print(stderr, "cmt: {} looks like {}: {:?}\n", std::empty(finding.pointer) ? "/" : finding.pointer,
				   finding.kind, finding.value);
	}
	return !std::empty(findings);
}

std::string RawAt(Json document, std::string_view pointer)
{
	Json value;
	return document.at_pointer(pointer).get(value) ? std::string("null") : simdjson::to_string(value);
}

Result<std::string> MachineKeyJson(Json machine)
{
	std::string_view link;
	std::optional<std::string> speed;
	std::optional<std::uint64_t> width;
	if (!machine.at_pointer("/gpus/0/narrowest_link/link").get(link))
	{
		const auto x = link.rfind(" x");
		std::uint64_t parsed = 0;
		const auto digits = x == std::string_view::npos ? std::string_view{} : link.substr(x + 2);
		const auto end = std::data(digits) + std::size(digits);
		if (!std::empty(digits) && std::from_chars(std::data(digits), end, parsed).ptr == end)
		{
			speed = std::string(link.substr(0, x));
			width = parsed;
		}
	}
	std::string_view registration;
	const bool active = !machine.at_pointer("/driver/host_registration/status").get(registration) &&
						registration == "active";
	auto key = KeyOf(machine);
	if (!key)
	{
		return Failure(key.error());
	}
	JsonWriter writer;
	writer.start_object();
	JsonKey(writer, "cpu", true);
	writer.start_object();
	JsonField(writer, "arch", key->arch, true);
	JsonField(writer, "model", key->cpu);
	for (const char* field : {"sockets", "cores", "numa_nodes"})
	{
		JsonKey(writer, field);
		writer.append_raw(RawAt(machine, std::string("/host/cpu/") + field));
	}
	writer.end_object();
	JsonKey(writer, "gpu");
	writer.start_object();
	JsonField(writer, "name", key->gpu, true);
	JsonKey(writer, "compute_capability");
	writer.append_raw(RawAt(machine, "/gpus/0/compute_capability"));
	JsonKey(writer, "memory_bytes");
	writer.append_raw(RawAt(machine, "/gpus/0/memory_bytes"));
	writer.end_object();
	JsonKey(writer, "link");
	writer.start_object();
	JsonKey(writer, "kind", true);
	writer.append_raw(RawAt(machine, "/gpus/0/host_link/kind"));
	JsonField(writer, "speed", speed);
	JsonField(writer, "width", width);
	writer.end_object();
	JsonKey(writer, "driver");
	writer.start_object();
	JsonField(writer, "version", key->driver, true);
	JsonKey(writer, "cuda_driver_api");
	writer.append_raw(RawAt(machine, "/driver/cuda_driver_api"));
	writer.end_object();
	JsonField(writer, "host_registration", active ? "active" : "unknown");
	JsonKey(writer, "thp");
	writer.start_object();
	JsonKey(writer, "enabled", true);
	writer.append_raw(RawAt(machine, "/host/memory/thp_enabled"));
	JsonKey(writer, "defrag");
	writer.append_raw(RawAt(machine, "/host/memory/thp_defrag"));
	writer.end_object();
	writer.end_object();
	return Text(writer);
}

bool Selects(const Row& row, const CopyRun& run)
{
	if (run.mode != row.mode)
	{
		return false;
	}
	switch (row.mode)
	{
	case Mode::Driver:
		return true;
	case Mode::Chunks:
		return run.settings.workers == static_cast<int>(row.settings.workers) &&
			   run.settings.slots == static_cast<int>(row.settings.slots) &&
			   run.settings.slot_bytes == row.settings.chunk_bytes && run.settings.loop == row.loop_name &&
			   run.settings.wc_slots == row.settings.wc_slots;
	case Mode::ProgressivePinning:
		return run.pinning.first_chunk_bytes == row.pinning.first_chunk_bytes &&
			   run.pinning.max_chunk_bytes == row.pinning.max_chunk_bytes &&
			   run.pinning.max_registered_bytes == row.pinning.max_registered_bytes &&
			   run.pinning.read_only == row.pinning.read_only;
	}
	return false;
}

Result<std::string> SummaryJson(const std::vector<CopyRun>& runs, const Table& table)
{
	struct Samples
	{
		std::vector<double> driver;
		std::vector<double> selected;
	};
	std::map<std::pair<unsigned, std::uint64_t>, Samples> sizes;
	for (const CopyRun& run : runs)
	{
		if (run.status != "complete" || !run.verified)
		{
			continue;
		}
		auto& samples = sizes[{static_cast<unsigned>(run.direction), run.bytes}];
		const std::vector<double> rates = std::empty(run.sample_gbps) ? std::vector<double>{run.gbps} : run.sample_gbps;
		const Row* row = table.Find(run.direction, run.bytes);
		if (run.mode == Mode::Driver)
		{
			samples.driver.insert(std::end(samples.driver), std::begin(rates), std::end(rates));
		}
		if (row ? Selects(*row, run) : run.mode == Mode::Driver)
		{
			samples.selected.insert(std::end(samples.selected), std::begin(rates), std::end(rates));
		}
	}
	JsonWriter writer;
	writer.start_array();
	bool first = true;
	for (const auto& [key, samples] : sizes)
	{
		const auto median = [](const std::vector<double>& values)
		{ return std::empty(values) ? std::nullopt : std::optional<double>(Median(values)); };
		writer.append_raw(first ? "" : ",");
		first = false;
		writer.start_object();
		JsonField(writer, "direction", DirName(static_cast<Dir>(key.first)), true);
		JsonField(writer, "bytes", key.second);
		JsonField(writer, "driver_gbps", median(samples.driver));
		JsonField(writer, "selected_gbps", median(samples.selected));
		JsonField(writer, "n", std::size(samples.selected));
		writer.end_object();
	}
	writer.end_array();
	return Text(writer);
}

LocalCapabilities Everything()
{
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	everything.progressive_pinning = true;
	return everything;
}

bool HasExactly(simdjson::dom::object object, std::initializer_list<std::string_view> keys)
{
	std::set<std::string_view> seen;
	for (auto field : object)
	{
		if (std::find(std::begin(keys), std::end(keys), field.key) == std::end(keys) || !seen.insert(field.key).second)
		{
			return false;
		}
	}
	return std::size(seen) == std::size(keys);
}

using Field = simdjson::simdjson_result<Json>;

bool StringOrNull(Field field)
{
	Json value;
	return !field.get(value) && (value.is_string() || value.is_null());
}

bool UnsignedOrNull(Field field)
{
	Json value;
	std::uint64_t number = 0;
	return !field.get(value) && (value.is_null() || !value.get(number));
}

bool NonEmptyString(Field field)
{
	std::string_view text;
	return !field.get(text) && !std::empty(text);
}

bool RateOrNull(Field field)
{
	Json value;
	double rate = 0;
	return !field.get(value) && (value.is_null() || (!value.get(rate) && std::isfinite(rate) && rate >= 0));
}

bool ValidMachine(Field machine)
{
	simdjson::dom::object object, cpu, gpu, link, driver, thp;
	std::string_view registration;
	return !machine.get(object) &&
		   HasExactly(object, {"cpu", "gpu", "link", "driver", "host_registration", "thp"}) &&
		   !object["cpu"].get(cpu) && HasExactly(cpu, {"arch", "model", "sockets", "cores", "numa_nodes"}) &&
		   NonEmptyString(cpu["arch"]) && NonEmptyString(cpu["model"]) && UnsignedOrNull(cpu["sockets"]) &&
		   UnsignedOrNull(cpu["cores"]) && UnsignedOrNull(cpu["numa_nodes"]) && !object["gpu"].get(gpu) &&
		   HasExactly(gpu, {"name", "compute_capability", "memory_bytes"}) && NonEmptyString(gpu["name"]) &&
		   StringOrNull(gpu["compute_capability"]) && UnsignedOrNull(gpu["memory_bytes"]) &&
		   !object["link"].get(link) && HasExactly(link, {"kind", "speed", "width"}) && StringOrNull(link["kind"]) &&
		   StringOrNull(link["speed"]) && UnsignedOrNull(link["width"]) && !object["driver"].get(driver) &&
		   HasExactly(driver, {"version", "cuda_driver_api"}) && NonEmptyString(driver["version"]) &&
		   StringOrNull(driver["cuda_driver_api"]) && !object["host_registration"].get(registration) &&
		   (registration == "active" || registration == "unknown") && !object["thp"].get(thp) &&
		   HasExactly(thp, {"enabled", "defrag"}) && StringOrNull(thp["enabled"]) && StringOrNull(thp["defrag"]);
}

bool ValidTool(Field tool)
{
	simdjson::dom::object object;
	Json dirty;
	return !tool.get(object) && HasExactly(object, {"version", "commit", "dirty"}) && StringOrNull(object["version"]) &&
		   StringOrNull(object["commit"]) && !object["dirty"].get(dirty) && (dirty.is_bool() || dirty.is_null());
}

bool ValidSummary(Field summary)
{
	simdjson::dom::array entries;
	if (summary.get(entries))
	{
		return false;
	}
	for (Json entry : entries)
	{
		simdjson::dom::object object;
		std::string_view direction;
		std::uint64_t number = 0;
		if (entry.get(object) || !HasExactly(object, {"direction", "bytes", "driver_gbps", "selected_gbps", "n"}) ||
			object["direction"].get(direction) || (direction != "htod" && direction != "dtoh") ||
			object["bytes"].get(number) || !RateOrNull(object["driver_gbps"]) || !RateOrNull(object["selected_gbps"]) ||
			object["n"].get(number))
		{
			return false;
		}
	}
	return true;
}

bool Sha256Hex(std::string_view text)
{
	return std::size(text) == 64 &&
		   std::all_of(std::begin(text), std::end(text), [](char c) { return IsDigit(c) || (c >= 'a' && c <= 'f'); });
}

} // namespace

Result<std::string> ShareableMachine(Json machine)
{
	JsonWriter writer;
	std::vector<std::string> path;
	std::vector<std::string> withheld;
	simdjson::dom::object object;
	if (machine.get(object))
	{
		return UsageError("a machine record must be a JSON object");
	}
	WriteMachine(writer, machine, path, withheld);
	auto text = Text(writer);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonWriter list;
	WriteJsonStringArray(list, withheld);
	auto names = Text(list);
	if (!names)
	{
		return Failure(names.error());
	}
	text->pop_back();
	return *text + (std::size(*text) > 1 ? "," : "") + "\"withheld\":" + *names + "}";
}

std::vector<std::string> ShareableEnvironment(const std::vector<std::string>& env, std::string_view out_dir)
{
	std::vector<std::string> kept;
	for (const auto& entry : env)
	{
		const auto equals = entry.find('=');
		const std::string_view name = std::string_view(entry).substr(0, equals);
		const std::string_view value =
			equals == std::string::npos ? std::string_view{} : std::string_view(entry).substr(equals + 1);
		if (name == "LD_LIBRARY_PATH")
		{
			continue;
		}
		if (name == "CUDA_VISIBLE_DEVICES")
		{
			const bool ordinals =
				std::all_of(std::begin(value), std::end(value), [](char c) { return IsDigit(c) || c == ','; });
			kept.push_back(ordinals ? entry
									: fmt::format("CUDA_VISIBLE_DEVICES=<{} entries withheld>",
												  std::count(std::begin(value), std::end(value), ',') + 1));
			continue;
		}
		kept.push_back(std::string(name) + (equals == std::string::npos ? "" : "=") + ShareablePath(value, out_dir));
	}
	return kept;
}

std::string ShareablePath(std::string_view path, std::string_view out_dir)
{
	while (std::size(out_dir) > 1 && out_dir.back() == '/')
	{
		out_dir.remove_suffix(1);
	}
	if (!std::empty(out_dir) && path.rfind(out_dir, 0) == 0 && std::size(path) > std::size(out_dir) &&
		path[std::size(out_dir)] == '/')
	{
		return std::string(path.substr(std::size(out_dir) + 1));
	}
	return std::string(path.find('/') == std::string_view::npos ? path : Basename(path));
}

std::string WithoutValue(std::string text, std::string_view value)
{
	constexpr std::string_view kWithheld = "<withheld>";
	for (auto at = std::empty(value) ? std::string::npos : text.find(value); at != std::string::npos;
		 at = text.find(value, at + std::size(kWithheld)))
	{
		text.replace(at, std::size(value), kWithheld);
	}
	return text;
}

Result<std::string> MachineFingerprint(Json machine, const std::vector<std::string>& env, std::string_view salt)
{
	std::string payload = std::string(salt) + "\n";
	const auto add = [&](const std::string& pointer, bool required) -> Result<void>
	{
		Json value;
		if (machine.at_pointer(pointer).get(value))
		{
			if (required)
			{
				fmt::print(stderr, "cmt tune: missing identity field /source_machine{}\n", pointer);
				return UsageError("missing machine identity field");
			}
			payload += pointer + " absent\n";
			return {};
		}
		payload += pointer + "=" + simdjson::to_string(value) + "\n";
		return {};
	};
	for (const char* pointer : {"/gpus/0/uuid", "/gpus/0/pci_bus_id", "/gpus/0/iommu_group"})
	{
		CMT_RETURN_IF_ERROR(add(pointer, true));
	}
	for (const char* pointer : {"/driver/kernel_module/banner", "/driver/libcuda/path"})
	{
		CMT_RETURN_IF_ERROR(add(pointer, false));
	}
	simdjson::dom::array hops;
	if (machine.at_pointer("/gpus/0/pcie_path").get(hops))
	{
		return UsageError("machine identity requires a PCIe path array");
	}
	for (std::size_t index = 0; index < std::size(hops); index++)
	{
		CMT_RETURN_IF_ERROR(add("/gpus/0/pcie_path/" + std::to_string(index) + "/bdf", true));
	}
	for (const auto& entry : env)
	{
		payload += "env " + entry + "\n";
	}
	return "sha256:" + Sha256(payload);
}

Result<std::string> NewSalt()
{
	unsigned char bytes[16];
	for (std::size_t filled = 0; filled < sizeof(bytes);)
	{
		const ssize_t got = getrandom(bytes + filled, sizeof(bytes) - filled, 0);
		if (got < 0 && errno == EINTR)
		{
			continue;
		}
		if (got <= 0)
		{
			return OsError("getrandom for the identity salt");
		}
		filled += static_cast<std::size_t>(got);
	}
	std::string hex;
	for (unsigned char byte : bytes)
	{
		hex += fmt::format("{:02x}", byte);
	}
	return hex;
}

std::vector<Finding> FindSensitive(Json document)
{
	std::vector<Finding> findings;
	std::vector<std::string> path;
	Scan(document, path, {}, findings);
	return findings;
}

Result<MachineKey> KeyOf(Json machine)
{
	MachineKey key;
	const auto read = [&](std::initializer_list<std::string_view> pointers, std::string& into)
	{
		for (auto pointer : pointers)
		{
			std::string_view text;
			if (!machine.at_pointer(pointer).get(text) && !std::empty(text))
			{
				into = text;
				return true;
			}
		}
		return false;
	};
	if (!read({"/host/cpu/arch"}, key.arch) || !read({"/host/cpu/model"}, key.cpu) || !read({"/gpus/0/name"}, key.gpu) ||
		!read({"/driver/kernel_module/version", "/driver/libcuda/version"}, key.driver))
	{
		return UsageError("the machine record lacks the CPU arch and model, the GPU name or the driver version");
	}
	return key;
}

std::string Slug(std::string_view text)
{
	std::string slug;
	for (char c : text)
	{
		const auto byte = static_cast<unsigned char>(c);
		if (byte < 0x80 && std::isalnum(byte))
		{
			slug += static_cast<char>(std::tolower(byte));
		}
		else if (!std::empty(slug) && slug.back() != '-')
		{
			slug += '-';
		}
	}
	while (!std::empty(slug) && slug.back() == '-')
	{
		slug.pop_back();
	}
	return slug;
}

Result<std::string> Export(const std::string& tune_json)
{
	JsonParser parser;
	Json document;
	std::string_view schema, state;
	std::uint64_t version = 0;
	Json machine;
	if (parser.parse(tune_json).get(document) || document["schema"].get(schema) || schema != "cmt.tuning" ||
		document["version"].get(version) || version != 1 || document["state"].get(state) ||
		document["source_machine"].get(machine))
	{
		return UsageError("expected a published cmt.tuning version 1 result (tune.json)");
	}
	if (state != "complete")
	{
		return UsageError("tune.json is partial; finish it with `cmt tune --resume` before sharing it");
	}
	auto expected = Reselect(tune_json);
	if (!expected)
	{
		return Failure(expected.error());
	}
	if (*expected != tune_json)
	{
		return Failure(Error::Kind::Verify,
					   "line 1 differs from the runs (or was written by an older cmt); run `cmt check --write` first");
	}
	if (!SelectionLineBytes(tune_json))
	{
		return Unsupported("this tune.json has no selection line (its runs select no table, or it would exceed "
						   "1 MiB), so there is nothing to share");
	}
	std::uint64_t file_bytes = 0;
	auto table = tuning::ReadSelection(tune_json, Everything(), file_bytes);
	auto runs = ReadCopyRuns(document);
	auto key = MachineKeyJson(machine);
	if (!table || !runs || !key)
	{
		return Failure(!table ? table.error() : !runs ? runs.error() : key.error());
	}
	auto summary = SummaryJson(*runs, *table);
	if (!summary)
	{
		return Failure(summary.error());
	}
	auto host_allocation = tuning::ReadHostAllocation(document);
	if (!host_allocation)
	{
		return Failure(host_allocation.error());
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", kShareSchema, true);
	JsonKey(writer, "tool");
	writer.start_object();
	JsonKey(writer, "version", true);
	writer.append_raw(RawAt(machine, "/tool/version"));
	JsonKey(writer, "commit");
	writer.append_raw(RawAt(machine, "/tool/commit"));
	JsonKey(writer, "dirty");
	writer.append_raw(RawAt(machine, "/tool/dirty"));
	writer.end_object();
	JsonKey(writer, "machine");
	writer.append_raw(*key);
	JsonKey(writer, "summary");
	writer.append_raw(*summary);
	JsonField(writer, "source_sha256", Sha256(tune_json));
	if (*host_allocation)
	{
		JsonKey(writer, "host_allocation");
		tuning::WriteHostAllocation(writer, **host_allocation);
	}
	writer.end_object();
	auto body = FinishJson(writer);
	if (!body)
	{
		return Failure(body.error());
	}
	auto share = RewriteSelection(tune_json, std::string_view(*body).substr(1));
	if (!share)
	{
		return Failure(share.error());
	}
	auto checked = Validate(*share);
	if (!checked)
	{
		return Failure(checked.error());
	}
	return share;
}

std::string RepositoryPath(const ShareInfo& info, std::string_view text)
{
	return fmt::format("{}--{}--{}-{}.json", Slug(info.key.gpu).substr(0, 96), Slug(info.key.cpu).substr(0, 96),
					   Slug(info.tier).substr(0, 32), Sha256(text).substr(0, 8));
}

Result<ShareInfo> Validate(const std::string& text)
{
	const auto malformed = [](const auto& what) { return Failure(Error::Kind::Verify, what); };
	if (std::size(text) > kMaxShareBytes)
	{
		return malformed("a shared tune is at most 2 MiB");
	}
	JsonParser parser;
	Json document;
	simdjson::dom::object object;
	std::string_view schema, source_sha256;
	const auto parsed = parser.parse(text).get(document);
	if (parsed == simdjson::MEMALLOC)
	{
		Panic("shared tune allocation failed");
	}
	if (parsed || document.get(object) ||
		(!HasExactly(object, {"selection", "schema", "tool", "machine", "summary", "source_sha256"}) &&
		 !HasExactly(object,
					 {"selection", "schema", "tool", "machine", "summary", "source_sha256", "host_allocation"})) ||
		object["schema"].get(schema) || schema != kShareSchema)
	{
		return malformed("not a cmt.tune-share/1 document");
	}
	const auto line = SelectionLineBytes(text);
	std::uint64_t file_bytes = 0;
	if (!line || document.at_pointer("/selection/file_bytes").get(file_bytes) || file_bytes != std::size(text))
	{
		return malformed("the shared tune's selection line is missing or for another file size");
	}
	auto canonical = RewriteSelection(text, std::string_view(text).substr(*line));
	if (!canonical || *canonical != text)
	{
		return malformed("the shared tune's selection line is not as cmt writes it");
	}
	if (!ValidTool(object["tool"]) || !ValidMachine(object["machine"]) || !ValidSummary(object["summary"]) ||
		object["source_sha256"].get(source_sha256) || !Sha256Hex(source_sha256) ||
		!tuning::ReadHostAllocation(document))
	{
		return malformed("the shared tune has an invalid tool, machine, summary, source_sha256 or host_allocation");
	}
	ShareInfo info;
	const auto at = [&](const char* pointer)
	{
		std::string_view value;
		return document.at_pointer(pointer).get(value) ? std::string() : std::string(value);
	};
	info.key = {at("/machine/cpu/arch"), at("/machine/gpu/name"), at("/machine/cpu/model"),
				at("/machine/driver/version")};
	info.tier = at("/selection/request/search");
	if (!PrintableField(info.key.arch) || !PrintableField(info.key.gpu) || !PrintableField(info.key.cpu) ||
		!PrintableField(info.key.driver) || !PrintableField(info.tier))
	{
		return malformed("the shared tune's machine names or tier are not printable ASCII of at most 128 bytes");
	}
	if (info.key.arch != at("/selection/source_machine/host/cpu/arch") ||
		info.key.cpu != at("/selection/source_machine/host/cpu/model") ||
		info.key.gpu != at("/selection/source_machine/gpus/0/name"))
	{
		return malformed("the shared tune's machine key differs from its selection line's");
	}
	if (info.key.arch == ArchName(NativeArch()))
	{
		auto rows = tuning::ReadSelection(text, Everything(), file_bytes);
		if (!rows)
		{
			return Failure(rows.error());
		}
	}
	if (PrintFindings(FindSensitive(document)))
	{
		return Failure(Error::Kind::Verify, "the shared tune holds data that looks personal or host-identifying");
	}
	return info;
}

bool IsSelectionOnly(const std::string& text)
{
	JsonParser parser;
	simdjson::dom::object document;
	simdjson::dom::array runs;
	Json remote;
	return !parser.parse(text).get(document) && !document["remote"].get(remote) && !document["runs"].get(runs) &&
		   runs.size() == 0;
}

Result<SelectionOnlyInfo> ValidateSelectionOnly(const std::string& text)
{
	const auto malformed = [](const auto& what) { return Failure(Error::Kind::Verify, what); };
	if (std::size(text) > kMaxShareBytes)
	{
		return malformed("a selection-only tune is at most 2 MiB");
	}
	JsonParser parser;
	Json document;
	simdjson::dom::array runs;
	std::string_view schema, state, source;
	std::uint64_t version = 0, file_bytes = 0;
	const auto parsed = parser.parse(text).get(document);
	if (parsed == simdjson::MEMALLOC)
	{
		Panic("selection-only tune allocation failed");
	}
	if (parsed || document["schema"].get(schema) || schema != "cmt.tuning" || document["version"].get(version) ||
		version != 1 || document["state"].get(state) || state != "complete" || document["runs"].get(runs) ||
		runs.size() != 0 || document.at_pointer("/remote/source").get(source))
	{
		return malformed("not a selection-only cmt.tuning document (complete, no runs, a remote source)");
	}
	if (std::empty(source) || std::size(source) > 253 ||
		!std::all_of(std::begin(source), std::end(source),
					 [](char c) { return IsAlpha(c) || IsDigit(c) || c == '.' || c == '-'; }))
	{
		return malformed("the remote source is not a host name");
	}
	if (!SelectionLineBytes(text) || document.at_pointer("/selection/file_bytes").get(file_bytes) ||
		file_bytes != std::size(text))
	{
		return malformed("the selection line is missing or for another file size");
	}
	SelectionOnlyInfo info;
	info.source = source;
	const auto at = [&](const char* pointer)
	{
		std::string_view value;
		return document.at_pointer(pointer).get(value) ? std::string() : std::string(value);
	};
	info.key = {at("/selection/source_machine/host/cpu/arch"), at("/selection/source_machine/gpus/0/name"),
				at("/selection/source_machine/host/cpu/model"),
				at("/selection/source_machine/driver/kernel_module/version")};
	if (std::empty(info.key.arch) || std::empty(info.key.gpu) || std::empty(info.key.cpu))
	{
		return malformed("the selection line names no CPU arch and model or GPU");
	}
	if (!PrintableField(info.key.arch) || !PrintableField(info.key.gpu) || !PrintableField(info.key.cpu) ||
		!PrintableField(info.key.driver))
	{
		return malformed("the selection line's machine names are not printable ASCII of at most 128 bytes");
	}
	if (info.key.arch == ArchName(NativeArch()))
	{
		auto rows = tuning::ReadSelection(text, Everything(), file_bytes);
		if (!rows)
		{
			return Failure(rows.error());
		}
	}
	return info;
}

} // namespace share
} // namespace cmt
