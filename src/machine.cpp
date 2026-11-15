#include "cuda/kernel_status.h"
#include "machine.h"
#include "common/helpers.h"
#include "cmt_build_info.h"

namespace cmt {

namespace {

std::string Under(const std::string& root, const char* path)
{
	if (std::empty(root) || root == "/")
	{
		return path;
	}
	return (root.back() == '/' ? root.substr(0, std::size(root) - 1) : root) + path;
}

std::string TrimMetadata(const std::string& s)
{
	const std::size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos)
	{
		return {};
	}
	const std::size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

std::optional<std::string> ReadMetadataText(const std::string& path)
{
	auto text = ReadTextFile(path);
	return text ? std::optional<std::string>(std::move(*text)) : std::nullopt;
}

std::optional<std::string> ReadMetadataLine(const std::string& path)
{
	auto text = ReadMetadataText(path);
	if (!text)
	{
		return std::nullopt;
	}
	auto line = TrimMetadata(text->substr(0, text->find('\n')));
	return line.empty() ? std::nullopt : std::optional<std::string>(std::move(line));
}

std::optional<std::vector<std::string>> ListMetadataDirectory(const std::string& path)
{
	auto names = ListDirectory(path);
	return names ? std::optional<std::vector<std::string>>(std::move(*names)) : std::nullopt;
}

} // namespace

void WriteToolJson(JsonWriter& writer)
{
	writer.start_object();
	JsonField(writer, "name", "cmt", true);
	JsonField(writer, "version", CMT_BUILD_VERSION);
	JsonField(writer, "backend", CMT_BUILD_BACKEND);
	JsonField(writer, "commit", CMT_BUILD_COMMIT);
	JsonField(writer, "dirty",
			  CMT_BUILD_DIRTY < 0 ? std::nullopt : std::optional<bool>(CMT_BUILD_DIRTY == 1));
	for (const auto& [key, value] :
		 {std::pair{"build_type", CMT_BUILD_TYPE}, std::pair{"cxx_compiler", CMT_BUILD_CXX_COMPILER},
		  std::pair{"gpu_compiler", CMT_BUILD_GPU_COMPILER}, std::pair{"gpu_toolkit", CMT_BUILD_GPU_TOOLKIT}})
	{
		JsonField(writer, key, *value ? std::optional<std::string_view>(value) : std::nullopt);
	}
	writer.end_object();
}

std::vector<std::string> MachineMissingKeys(const Json& doc)
{
	std::vector<std::string> bad;
	if (!doc.is_object())
	{
		return {"(not an object)"};
	}
	std::string_view schema;
	if (doc["schema"].get(schema) || schema != kMachineSchema)
	{
		bad.push_back("schema");
	}
	const std::pair<const char*, std::vector<const char*>> sections[] = {
		{"tool",
		 {"name", "version", "backend", "commit", "dirty", "build_type", "cxx_compiler", "gpu_toolkit"}},
		{"host",
		 {"hostname", "kernel", "os", "cmdline", "virtualization", "container", "cpu", "memory", "iommu"}},
		{"driver", {"vendor", "kernel_module"}}};
	for (const auto& [name, keys] : sections)
	{
		simdjson::dom::object section;
		if (doc[name].get(section))
		{
			bad.push_back(name);
			continue;
		}
		for (const char* key : keys)
		{
			Json value;
			if (section[key].get(value))
			{
				bad.push_back(std::string(name) + "." + key);
			}
		}
	}
	simdjson::dom::object host;
	if (!doc["host"].get(host))
	{
		for (const char* key : {"virtualization", "container", "cpu", "memory", "iommu", "kernel"})
		{
			Json value;
			if (!host[key].get(value) && !value.is_object())
			{
				bad.push_back(std::string("host.") + key);
			}
		}
	}
	Json gpu_error;
	if (doc["gpu_error"].get(gpu_error))
	{
		bad.push_back("gpu_error");
	}
	simdjson::dom::array gpus;
	if (doc["gpus"].get(gpus))
	{
		bad.push_back("gpus");
		return bad;
	}
	std::size_t i = 0;
	for (const auto gpu : gpus)
	{
		if (!gpu.is_object())
		{
			bad.push_back(fmt::format("gpus.{}", i));
		}
		else
		{
			for (const char* key : {"ordinal", "name", "uuid", "pci_bus_id", "compute_capability",
									"compute_units", "memory_bytes", "numa_node", "iommu_group", "pcie_path",
									"narrowest_link", "has_pcie_link", "host_link"})
			{
				Json value;
				if (gpu[key].get(value))
				{
					bad.push_back(fmt::format("gpus.{}.{}", i, key));
				}
			}
		}
		++i;
	}
	return bad;
}

} // namespace cmt

namespace cmt {
namespace host {

namespace fs = std::filesystem;

namespace {

const char* BuildArch()
{
#if defined(__x86_64__)
	return "x86_64";
#elif defined(__aarch64__)
	return "aarch64";
#elif defined(__powerpc64__)
	return "ppc64le";
#else
	return "unknown";
#endif
}

std::optional<std::uint64_t> ToU64(const std::string& s)
{
	if (std::empty(s) || s.find_first_not_of("0123456789") != std::string::npos)
	{
		return std::nullopt;
	}
	errno = 0;
	char* end = nullptr;
	const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
	if (errno == ERANGE || *end)
	{
		return std::nullopt;
	}
	return static_cast<std::uint64_t>(v);
}

std::optional<long long> ToI64(const std::string& s)
{
	if (std::empty(s))
	{
		return std::nullopt;
	}
	errno = 0;
	char* end = nullptr;
	const long long v = std::strtoll(s.c_str(), &end, 10);
	if (errno == ERANGE || *end || end == s.c_str())
	{
		return std::nullopt;
	}
	return v;
}

bool IsBdf(const std::string& s)
{
	const std::size_t n = std::size(s);
	if (n < 12 || s[n - 8] != ':' || s[n - 5] != ':' || s[n - 2] != '.')
	{
		return false;
	}
	for (std::size_t i = 0; i < n - 1; i++)
	{
		if (i == n - 8 || i == n - 5 || i == n - 2)
		{
			continue;
		}
		const char c = s[i];
		if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
		{
			return false;
		}
	}
	return s[n - 1] >= '0' && s[n - 1] <= '7';
}

std::string JoinLink(const std::string& speed, const std::string& width)
{
	if (std::empty(speed))
	{
		return "";
	}
	return fmt::format("{} x{}", speed, width);
}

const char* const kX86Flags[] = {"sse2", "avx2", "avx512f",	   "erms",
								 "fsrm", "clwb", "clflushopt", "movdir64b"};
const char* const kArmFlags[] = {"asimd", "sve", "sve2"};

void WriteCpuJson(JsonWriter& writer, const std::string& root)
{
	writer.start_object();
	JsonField(writer, "arch", BuildArch(), true);
	const auto text = ReadMetadataText(Under(root, "/proc/cpuinfo"));
	const CpuInfo ci = text ? ParseCpuInfo(*text) : CpuInfo{};
	JsonField(writer, "model", ci.model);
	JsonField(writer, "vendor", ci.vendor);
	const auto online = ReadMetadataLine(Under(root, "/sys/devices/system/cpu/online"));
	const auto cpus = online ? ParseIndexList(*online) : std::nullopt;
	JsonField(writer, "threads", cpus ? std::optional<std::size_t>(cpus->size()) : std::nullopt);
	std::optional<std::size_t> sockets, cores;
	if (cpus && !cpus->empty())
	{
		std::set<std::string> packages;
		std::set<std::pair<std::string, std::string>> core_ids;
		bool complete = true;
		for (std::size_t c : *cpus)
		{
			const std::string topo =
				fmt::format("{}/cpu{}/topology/", Under(root, "/sys/devices/system/cpu"), c);
			const auto pkg = ReadMetadataLine(topo + "physical_package_id"),
					   core = ReadMetadataLine(topo + "core_id");
			if (!pkg || !core)
			{
				complete = false;
				break;
			}
			packages.insert(*pkg);
			core_ids.insert({*pkg, *core});
		}
		if (complete)
		{
			sockets = packages.size();
			cores = core_ids.size();
		}
	}
	JsonField(writer, "sockets", sockets);
	JsonField(writer, "cores", cores);
	const auto nodes_line = ReadMetadataLine(Under(root, "/sys/devices/system/node/online"));
	const auto nodes = nodes_line ? ParseIndexList(*nodes_line) : std::nullopt;
	JsonField(writer, "numa_nodes", nodes ? std::optional<std::size_t>(nodes->size()) : std::nullopt);
	JsonKey(writer, "flags");
	if (ci.flags)
	{
		writer.start_object();
		const std::set<std::string> have(ci.flags->begin(), ci.flags->end());
		bool first = true;
		if (ci.arm_features)
		{
			for (const char* name : kArmFlags)
			{
				JsonField(writer, name, have.count(name) > 0, first);
				first = false;
			}
		}
		else
		{
			for (const char* name : kX86Flags)
			{
				JsonField(writer, name, have.count(name) > 0, first);
				first = false;
			}
		}
		writer.end_object();
	}
	else
	{
		writer.append_null();
	}
	const std::string freq =
		cpus && !cpus->empty()
			? fmt::format("{}/cpu{}/cpufreq/", Under(root, "/sys/devices/system/cpu"), cpus->front())
			: "";
	JsonField(writer, "scaling_governor",
			  freq.empty() ? std::nullopt : ReadMetadataLine(freq + "scaling_governor"));
	JsonField(writer, "scaling_driver",
			  freq.empty() ? std::nullopt : ReadMetadataLine(freq + "scaling_driver"));
	writer.end_object();
}

void WriteMemoryJson(JsonWriter& writer, const std::string& root)
{
	writer.start_object();
	const auto mi = ReadMetadataText(Under(root, "/proc/meminfo"));
	JsonField(writer, "total_bytes", mi ? ParseMemInfoValue(*mi, "MemTotal") : std::nullopt, true);
	const bool live_host = root.empty() || root == "/";
	JsonField(writer, "base_page_bytes",
			  live_host ? std::optional<std::size_t>(BasePageBytes()) : std::nullopt);
	JsonField(writer, "process_available_bytes",
			  live_host ? HostMemoryAvailable(Pages::Small) : std::nullopt);
	JsonField(writer, "process_available_basis",
			  "MemAvailable capped by applicable visible cgroup v2 limits");
	JsonField(writer, "process_available_limitation",
			  "estimate only; concurrent use and hidden ancestors can reduce it; null means unknown");
	const auto thp = ReadMetadataLine(Under(root, "/sys/kernel/mm/transparent_hugepage/enabled"));
	const auto defrag = ReadMetadataLine(Under(root, "/sys/kernel/mm/transparent_hugepage/defrag"));
	JsonField(writer, "thp_enabled", thp ? Bracketed(*thp) : std::nullopt);
	JsonField(writer, "thp_defrag", defrag ? Bracketed(*defrag) : std::nullopt);
	const std::string hp = Under(root, "/sys/kernel/mm/hugepages");
	JsonKey(writer, "hugepages");
	if (const auto sizes = ListMetadataDirectory(hp))
	{
		writer.start_object();
		bool first = true;
		for (const auto& d : *sizes)
		{
			if (d.rfind("hugepages-", 0) != 0)
			{
				continue;
			}
			const auto nr = ReadMetadataLine(hp + "/" + d + "/nr_hugepages");
			const auto fr = ReadMetadataLine(hp + "/" + d + "/free_hugepages");
			JsonKey(writer, d.substr(std::size("hugepages-") - 1), first);
			writer.start_object();
			JsonField(writer, "total", nr ? ToU64(*nr) : std::nullopt, true);
			JsonField(writer, "free", fr ? ToU64(*fr) : std::nullopt);
			writer.end_object();
			first = false;
		}
		writer.end_object();
	}
	else
	{
		writer.append_null();
	}
	writer.end_object();
}

void WriteIommuJson(JsonWriter& writer, const std::string& root, const std::optional<std::string>& cmdline)
{
	writer.start_object();
	JsonKey(writer, "cmdline", true);
	writer.start_object();
	bool first = true;
	for (const char* key : {"iommu", "intel_iommu", "amd_iommu", "iommu.passthrough", "iommu.strict"})
	{
		JsonField(writer, key, cmdline ? CommandLineParam(*cmdline, key) : std::nullopt, first);
		first = false;
	}
	writer.end_object();
	JsonKey(writer, "units");
	if (const auto units = ListMetadataDirectory(Under(root, "/sys/class/iommu")))
	{
		WriteJsonStringArray(writer, *units);
	}
	else
	{
		writer.append_null();
	}
	const auto groups = ListMetadataDirectory(Under(root, "/sys/kernel/iommu_groups"));
	JsonField(writer, "groups", groups ? std::optional<std::size_t>(groups->size()) : std::nullopt);
	writer.end_object();
}

// From systemd v255, src/basic/virt.c; omit Oracle Corporation (also used by bare-metal servers).
const std::pair<const char*, const char*> kDmiVendors[] = {{"KVM", "kvm"},
														   {"OpenStack", "kvm"},
														   {"KubeVirt", "kvm"},
														   {"Amazon EC2", "amazon"},
														   {"QEMU", "qemu"},
														   {"VMware", "vmware"},
														   {"VMW", "vmware"},
														   {"innotek GmbH", "oracle"},
														   {"VirtualBox", "oracle"},
														   {"Xen", "xen"},
														   {"Bochs", "bochs"},
														   {"Parallels", "parallels"},
														   {"BHYVE", "bhyve"},
														   {"Hyper-V", "microsoft"},
														   {"Apple Virtualization", "apple"},
														   {"Google Compute Engine", "google"}};

// From systemd's cpuid_vendor_table.
const std::pair<const char*, const char*> kCpuidVendors[] = {
	{"XenVMMXenVMM", "xen"},   {"KVMKVMKVM", "kvm"},	   {"Linux KVM Hv", "kvm"},
	{"TCGTCGTCGTCG", "qemu"},  {"VMwareVMware", "vmware"}, {"Microsoft Hv", "microsoft"},
	{"bhyve bhyve ", "bhyve"}, {"QNXQVMBSQG", "qnx"},	   {"ACRNACRNACRN", "acrn"},
	{"SRESRESRESRE", "sre"},   {"VBoxVBoxVBox", "oracle"}};

} // namespace

std::optional<std::vector<std::size_t>> ParseIndexList(const std::string& list)
{
	constexpr std::size_t kMaxMetadataIndices = 1u << 20;
	std::vector<std::size_t> out;
	const std::string t = TrimMetadata(list);
	if (std::empty(t))
	{
		return out;
	}
	std::istringstream in(t);
	for (std::string part; std::getline(in, part, ',');)
	{
		const std::size_t dash = part.find('-');
		const auto lo = ToU64(dash == std::string::npos ? part : part.substr(0, dash));
		const auto hi = dash == std::string::npos ? lo : ToU64(part.substr(dash + 1));
		if (!lo || !hi || *hi < *lo || *hi - *lo >= kMaxMetadataIndices ||
			*hi - *lo + 1 > kMaxMetadataIndices - out.size())
		{
			return std::nullopt;
		}
		const auto count = *hi - *lo + 1;
		for (std::uint64_t offset = 0; offset < count; ++offset)
		{
			out.push_back(static_cast<std::size_t>(*lo + offset));
		}
	}
	return out;
}

std::optional<std::string> Bracketed(const std::string& line)
{
	const std::size_t b = line.find('['), e = line.find(']');
	if (b == std::string::npos || e == std::string::npos || e <= b + 1)
	{
		return std::nullopt;
	}
	return line.substr(b + 1, e - b - 1);
}

std::optional<std::string> CommandLineParam(const std::string& cmdline, const std::string& key)
{
	std::optional<std::string> v;
	std::istringstream in(cmdline);
	for (std::string w; in >> w;)
	{
		if (w == key)
		{
			v = std::string();
		}
		else if (w.rfind(key + "=", 0) == 0)
		{
			v = w.substr(std::size(key) + 1);
		}
	}
	return v;
}

std::optional<std::string> OsReleaseValue(const std::string& os_release, const std::string& key)
{
	std::istringstream in(os_release);
	for (std::string l; std::getline(in, l);)
	{
		if (l.rfind(key + "=", 0) != 0)
		{
			continue;
		}
		std::string v = TrimMetadata(l.substr(std::size(key) + 1));
		if (std::size(v) >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front())
		{
			v = v.substr(1, std::size(v) - 2);
		}
		if (std::empty(v))
		{
			return std::nullopt;
		}
		return v;
	}
	return std::nullopt;
}

std::optional<std::string> MappedLibrary(const std::string& maps, const std::string& stem)
{
	std::istringstream in(maps);
	for (std::string l; std::getline(in, l);)
	{
		const std::size_t slash = l.find('/');
		if (slash == std::string::npos)
		{
			continue;
		}
		std::string path = TrimMetadata(l.substr(slash));
		if (const std::size_t del = path.rfind(" (deleted)");
			del != std::string::npos && del + 10 == std::size(path))
		{
			path.erase(del);
		}
		const std::string name = path.substr(path.find_last_of('/') + 1);
		if (name.rfind(stem, 0) == 0)
		{
			return path;
		}
	}
	return std::nullopt;
}

std::optional<std::string> SonameVersion(const std::string& path, const std::string& stem)
{
	const std::string name = path.substr(path.find_last_of('/') + 1);
	if (name.rfind(stem + ".", 0) != 0)
	{
		return std::nullopt;
	}
	const std::string v = name.substr(std::size(stem) + 1);
	if (v.find('.') == std::string::npos || v.find_first_not_of("0123456789.") != std::string::npos)
	{
		return std::nullopt;
	}
	return v;
}

KernelModule ReadKernelModule(const std::string& root, const std::string& name)
{
	KernelModule m;
	const std::string dir = Under(root, "/sys/module/") + name;
	std::error_code ec;
	m.loaded = fs::is_directory(dir, ec);
	m.version = ReadMetadataLine(dir + "/version");
	m.srcversion = ReadMetadataLine(dir + "/srcversion");
	return m;
}

Cpuid ReadCpuid()
{
	Cpuid c;
#if defined(__x86_64__) || defined(__i386__)
	unsigned a = 0, b = 0, cx = 0, d = 0;
	if (!__get_cpuid(1, &a, &b, &cx, &d))
	{
		return c;
	}
	c.hypervisor_bit = (cx >> 31) & 1u;
	if (*c.hypervisor_bit)
	{
		__cpuid(0x40000000, a, b, cx, d);
		char v[13] = {0};
		std::memcpy(v, &b, 4);
		std::memcpy(v + 4, &cx, 4);
		std::memcpy(v + 8, &d, 4);
		c.vendor = std::string(v);
	}
#endif
	return c;
}

VmEvidence ReadVmEvidence(const std::string& root, const Cpuid& cpuid)
{
	VmEvidence e;
	e.cpuid = cpuid;
	const std::string dmi = Under(root, "/sys/class/dmi/id/");
	e.sys_vendor = ReadMetadataLine(dmi + "sys_vendor");
	e.product_name = ReadMetadataLine(dmi + "product_name");
	e.board_vendor = ReadMetadataLine(dmi + "board_vendor");
	e.bios_vendor = ReadMetadataLine(dmi + "bios_vendor");
	e.sys_hypervisor_type = ReadMetadataLine(Under(root, "/sys/hypervisor/type"));
	if (const auto dt = ReadMetadataText(Under(root, "/proc/device-tree/hypervisor/compatible")))
	{
		const std::string first =
			dt->substr(0, dt->find('\0'));
		if (!std::empty(first))
		{
			e.device_tree_hypervisor = first;
		}
	}
	return e;
}

Verdict ClassifyVm(const VmEvidence& e)
{
	const bool hv = e.cpuid.hypervisor_bit.value_or(false);
	if (hv && e.cpuid.vendor)
	{
		for (const auto& [sig, type] : kCpuidVendors)
		{
			if (*e.cpuid.vendor == sig)
			{
				return {std::string(type), fmt::format("CPUID hypervisor vendor '{}'", *e.cpuid.vendor)};
			}
		}
	}
	if (e.sys_hypervisor_type)
	{
		return {*e.sys_hypervisor_type, fmt::format("/sys/hypervisor/type '{}'", *e.sys_hypervisor_type)};
	}
	const std::pair<const char*, const std::optional<std::string>*> dmi[] = {
		{"product_name", &e.product_name},
		{"sys_vendor", &e.sys_vendor},
		{"board_vendor", &e.board_vendor},
		{"bios_vendor", &e.bios_vendor}};
	for (const auto& [field, value] : dmi)
	{
		if (!*value)
		{
			continue;
		}
		for (const auto& [prefix, type] : kDmiVendors)
		{
			if ((*value)->rfind(prefix, 0) != 0)
			{
				continue;
			}
			if (std::string(type) == "amazon" && e.cpuid.hypervisor_bit == false)
			{
				return {std::string("none"),
						fmt::format("DMI {} '{}' but CPUID hypervisor bit clear: bare-metal instance", field,
									**value)};
			}
			return {std::string(type), fmt::format("DMI {} '{}'", field, **value)};
		}
	}
	if (e.sys_vendor && *e.sys_vendor == "Microsoft Corporation" && e.product_name &&
		*e.product_name == "Virtual Machine")
	{
		return {std::string("microsoft"), "DMI 'Microsoft Corporation' / 'Virtual Machine' (Hyper-V, Azure)"};
	}
	if (e.device_tree_hypervisor)
	{
		const std::string& c = *e.device_tree_hypervisor;
		const char* type = c.find("kvm") != std::string::npos	   ? "kvm"
						   : c.find("xen") != std::string::npos	   ? "xen"
						   : c.find("vmware") != std::string::npos ? "vmware"
																   : "vm-other";
		return {std::string(type), fmt::format("device-tree hypervisor '{}'", c)};
	}
	if (hv)
	{
		return {std::string("vm-other"), fmt::format("CPUID hypervisor bit set, vendor '{}' not recognized",
													 e.cpuid.vendor.value_or(""))};
	}
	if (e.cpuid.hypervisor_bit == false)
	{
		return {std::string("none"), "CPUID hypervisor bit clear and no DMI/sysfs match"};
	}
	return {std::nullopt, "no CPUID on this architecture and no DMI/sysfs/device-tree match"};
}

Verdict DetectContainer(const std::string& root)
{
	std::error_code ec;
	if (fs::exists(Under(root, "/.dockerenv"), ec))
	{
		return {std::string("docker"), "/.dockerenv exists"};
	}
	if (fs::exists(Under(root, "/run/.containerenv"), ec))
	{
		return {std::string("podman"), "/run/.containerenv exists"};
	}
	if (const auto env = ReadMetadataText(Under(root, "/proc/1/environ")))
	{
		for (std::size_t b = 0; b < std::size(*env);)
		{
			const std::size_t e = env->find('\0', b);
			const std::string kv = env->substr(b, e == std::string::npos ? std::string::npos : e - b);
			if (kv.rfind("container=", 0) == 0 && std::size(kv) > 10)
			{
				return {kv.substr(10), "PID 1's environment has " + kv};
			}
			if (e == std::string::npos)
			{
				break;
			}
			b = e + 1;
		}
	}
	if (const auto cg = ReadMetadataText(Under(root, "/proc/1/cgroup")))
	{
		const std::pair<const char*, const char*> marks[] = {{"kubepods", "kubernetes"},
															 {"docker", "docker"},
															 {"libpod", "podman"},
															 {"/lxc", "lxc"},
															 {"containerd", "containerd"}};
		std::istringstream in(*cg);
		for (std::string l; std::getline(in, l);)
		{
			for (const auto& [mark, type] : marks)
			{
				if (l.find(mark) != std::string::npos)
				{
					return {std::string(type), fmt::format("/proc/1/cgroup '{}'", TrimMetadata(l))};
				}
			}
		}
	}
	return {std::nullopt, "no marker file, no container= for PID 1, no container cgroup path (a cgroup "
						  "namespace hides it)"};
}

std::vector<PcieHop> PciePath(const std::string& pci_devices, const std::string& bdf)
{
	std::vector<PcieHop> out;
	std::error_code ec;
	const fs::path resolved = fs::canonical(pci_devices + "/" + bdf, ec);
	if (ec)
	{
		return out;
	}
	for (fs::path dev = resolved; !dev.empty() && IsBdf(dev.filename().string()); dev = dev.parent_path())
	{
		PcieHop h;
		h.bdf = dev.filename().string();
		const std::string d = dev.string() + "/";
		h.max_speed = ReadMetadataLine(d + "max_link_speed").value_or("");
		h.max_width = ReadMetadataLine(d + "max_link_width").value_or("");
		h.current_speed = ReadMetadataLine(d + "current_link_speed").value_or("");
		h.current_width = ReadMetadataLine(d + "current_link_width").value_or("");
		h.link_max = JoinLink(h.max_speed, h.max_width);
		h.link_current = JoinLink(h.current_speed, h.current_width);
		out.push_back(std::move(h));
		if (dev == dev.parent_path())
		{
			break;
		}
	}
	return out;
}

double LinkGbps(const std::string& l)
{
	double gts = 0;
	if (std::sscanf(l.c_str(), "%lf GT/s", &gts) != 1 || !std::isfinite(gts) || !(gts > 0))
	{
		return 0;
	}
	const std::size_t x = l.rfind(" x");
	const auto width = x == std::string::npos ? std::nullopt : ToU64(l.substr(x + 2));
	if (!width || *width == 0)
	{
		return 0;
	}
	const double enc = gts >= 64 ? 1.0 : gts >= 8 ? 128.0 / 130 : 0.8;
	const double gbps = gts * enc * static_cast<double>(*width) / 8;
	if (!std::isfinite(gbps) || gbps > std::numeric_limits<double>::max() / 1000)
	{
		return 0;
	}
	return gbps;
}

void WriteHostJson(JsonWriter& writer, const std::string& root, const Cpuid& cpuid)
{
	writer.start_object();
	JsonField(writer, "hostname", ReadMetadataLine(Under(root, "/proc/sys/kernel/hostname")), true);
	JsonKey(writer, "kernel");
	writer.start_object();
	JsonField(writer, "release", ReadMetadataLine(Under(root, "/proc/sys/kernel/osrelease")), true);
	JsonField(writer, "version", ReadMetadataLine(Under(root, "/proc/sys/kernel/version")));
	writer.end_object();
	auto os = ReadMetadataText(Under(root, "/etc/os-release"));
	if (!os)
	{
		os = ReadMetadataText(Under(root, "/usr/lib/os-release"));
	}
	JsonField(writer, "os", os ? OsReleaseValue(*os, "PRETTY_NAME") : std::nullopt);
	const auto cmdline = ReadMetadataLine(Under(root, "/proc/cmdline"));
	JsonField(writer, "cmdline", cmdline);
	const VmEvidence e = ReadVmEvidence(root, cpuid);
	const Verdict vm = ClassifyVm(e);
	JsonKey(writer, "virtualization");
	writer.start_object();
	JsonField(writer, "type", vm.type, true);
	JsonField(writer, "evidence", vm.evidence);
	JsonField(writer, "cpuid_hypervisor_bit", e.cpuid.hypervisor_bit);
	JsonField(writer, "cpuid_vendor", e.cpuid.vendor);
	JsonKey(writer, "dmi");
	writer.start_object();
	JsonField(writer, "sys_vendor", e.sys_vendor, true);
	JsonField(writer, "product_name", e.product_name);
	JsonField(writer, "board_vendor", e.board_vendor);
	JsonField(writer, "bios_vendor", e.bios_vendor);
	writer.end_object();
	JsonField(writer, "sys_hypervisor_type", e.sys_hypervisor_type);
	JsonField(writer, "device_tree_hypervisor", e.device_tree_hypervisor);
	writer.end_object();
	const Verdict ct = DetectContainer(root);
	JsonKey(writer, "container");
	writer.start_object();
	JsonField(writer, "type", ct.type, true);
	JsonField(writer, "evidence", ct.evidence);
	JsonField(writer, "pid1_comm", ReadMetadataLine(Under(root, "/proc/1/comm")));
	writer.end_object();
	JsonKey(writer, "cpu");
	WriteCpuJson(writer, root);
	JsonKey(writer, "memory");
	WriteMemoryJson(writer, root);
	JsonKey(writer, "iommu");
	WriteIommuJson(writer, root, cmdline);
	writer.end_object();
}

bool WritePciDeviceFields(JsonWriter& writer, const std::string& root, const std::optional<std::string>& bdf,
						  bool first)
{
	if (!bdf)
	{
		JsonField(writer, "numa_node", nullptr, first);
		JsonField(writer, "iommu_group", nullptr);
		JsonField(writer, "iommu_group_type", nullptr);
		JsonKey(writer, "pcie_path");
		writer.start_array();
		writer.end_array();
		JsonField(writer, "narrowest_link", nullptr);
		JsonField(writer, "has_pcie_link", nullptr);
		return false;
	}
	const std::string devices = Under(root, "/sys/bus/pci/devices");
	const std::string dev = devices + "/" + *bdf;
	const auto numa = ReadMetadataLine(dev + "/numa_node");
	JsonField(writer, "numa_node", numa ? ToI64(*numa) : std::nullopt, first);
	std::error_code ec;
	const fs::path group = fs::read_symlink(dev + "/iommu_group", ec);
	if (!ec && !group.filename().empty())
	{
		const std::string n = group.filename().string();
		JsonField(writer, "iommu_group", n);
		JsonField(writer, "iommu_group_type",
				  ReadMetadataLine(Under(root, "/sys/kernel/iommu_groups/") + n + "/type"));
	}
	else
	{
		JsonField(writer, "iommu_group", nullptr);
		JsonField(writer, "iommu_group_type", nullptr);
	}
	const auto path = PciePath(devices, *bdf);
	const auto nonempty = [](const std::string& text) -> std::optional<std::string_view>
	{ return text.empty() ? std::nullopt : std::optional<std::string_view>(text); };
	JsonKey(writer, "pcie_path");
	writer.start_array();
	std::optional<std::size_t> narrowest;
	double slow = 0;
	for (std::size_t i = 0; i < path.size(); ++i)
	{
		const PcieHop& hop = path[i];
		const double gbps = LinkGbps(hop.link_max);
		if (gbps > 0 && (!narrowest || gbps < slow))
		{
			narrowest = i;
			slow = gbps;
		}
		if (i != 0)
		{
			writer.append_comma();
		}
		writer.start_object();
		JsonField(writer, "bdf", hop.bdf, true);
		JsonField(writer, "max_speed", nonempty(hop.max_speed));
		JsonField(writer, "max_width", ToU64(hop.max_width));
		JsonField(writer, "current_speed", nonempty(hop.current_speed));
		JsonField(writer, "current_width", ToU64(hop.current_width));
		JsonField(writer, "max_gbps",
				  gbps > 0 ? std::optional<double>(std::round(gbps * 1000) / 1000) : std::nullopt);
		writer.end_object();
	}
	writer.end_array();
	JsonKey(writer, "narrowest_link");
	if (narrowest)
	{
		writer.start_object();
		JsonField(writer, "bdf", path[*narrowest].bdf, true);
		JsonField(writer, "link", path[*narrowest].link_max);
		JsonField(writer, "gbps", std::round(slow * 1000) / 1000);
		JsonField(writer, "current", nonempty(path[*narrowest].link_current));
		writer.end_object();
	}
	else
	{
		writer.append_null();
	}
	const bool has_pcie_link = !path.empty() && LinkGbps(path.front().link_max) > 0;
	JsonField(writer, "has_pcie_link", has_pcie_link);
	return has_pcie_link;
}

void WritePciDeviceJson(JsonWriter& writer, const std::string& root, const std::string& bdf)
{
	writer.start_object();
	WritePciDeviceFields(writer, root, bdf, true);
	writer.end_object();
}

} // namespace host
} // namespace cmt

namespace cmt {
namespace nv {

namespace {

std::string Show(std::optional<bool> b)
{
	return b ? (*b ? "1" : "0") : "unreadable";
}

} // namespace

std::optional<std::string> ReleaseInBanner(const std::string& banner)
{
	std::istringstream w(banner);
	for (std::string x; w >> x;)
	{
		if (std::count(std::begin(x), std::end(x), '.') >= 1 &&
			std::isdigit(static_cast<unsigned char>(x.front())) &&
			std::all_of(std::begin(x), std::end(x),
						[](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) || ch == '.'; }))
		{
			return x;
		}
	}
	return std::nullopt;
}

Module NvidiaModule(const std::string& root)
{
	Module m;
	const host::KernelModule k = host::ReadKernelModule(root, "nvidia");
	const std::string proc =
		(std::empty(root) || root == "/" ? std::string() : root) + "/proc/driver/nvidia/version";
	m.banner = ReadMetadataLine(proc);
	m.version = k.version ? k.version : (m.banner ? ReleaseInBanner(*m.banner) : std::nullopt);
	m.srcversion = k.srcversion;
	return m;
}

const char* HostLinkName(HostLink k)
{
	switch (k)
	{
	case HostLink::Pcie:
		return "pcie";
	case HostLink::NvlinkC2c:
		return "nvlink-c2c";
	case HostLink::Nvlink:
		return "nvlink";
	case HostLink::Integrated:
		return "integrated";
	case HostLink::Unknown:
		return "unknown";
	}
	return "unknown";
}

HostLinkVerdict DecideHostLink(std::optional<bool> integrated, std::optional<bool> native_atomics,
							   std::optional<bool> uses_host_page_tables, bool has_pcie_link, bool aarch64)
{
	const std::string ev =
		fmt::format("integrated={} hostNativeAtomicSupported={} pageableMemoryAccessUsesHostPageTables={} "
					"sysfs_pcie_link={} host_arch={}",
					Show(integrated), Show(native_atomics), Show(uses_host_page_tables),
					has_pcie_link ? "yes" : "no", aarch64 ? "aarch64" : "other");
	if (integrated == true)
	{
		return {HostLink::Integrated, ev};
	}
	if (native_atomics == true && uses_host_page_tables == true)
	{
		return {aarch64 ? HostLink::NvlinkC2c : HostLink::Nvlink, ev};
	}
	if (native_atomics == false && uses_host_page_tables == false && has_pcie_link)
	{
		return {HostLink::Pcie, ev};
	}
	return {HostLink::Unknown, ev};
}

std::optional<std::string> CudaVersionString(int packed)
{
	if (packed <= 0)
	{
		return std::nullopt;
	}
	return fmt::format("{}.{}", packed / 1000, (packed % 1000) / 10);
}

void WriteDriverJson(JsonWriter& writer, const std::string& root, const std::string& maps,
					 std::optional<int> driver_api, std::optional<int> runtime,
					 const std::optional<std::string>& module_sets)
{
	const Module m = NvidiaModule(root);
	const auto lib = host::MappedLibrary(maps, "libcuda.so");
	writer.start_object();
	JsonField(writer, "vendor", "nvidia", true);
	JsonKey(writer, "kernel_module");
	writer.start_object();
	JsonField(writer, "version", m.version, true);
	JsonField(writer, "srcversion", m.srcversion);
	JsonField(writer, "banner", m.banner);
	writer.end_object();
	const auto uvm = host::ReadKernelModule(root, "nvidia_uvm");
	JsonKey(writer, "uvm_module");
	writer.start_object();
	JsonField(writer, "version", uvm.version, true);
	JsonField(writer, "srcversion", uvm.srcversion);
	writer.end_object();
	const auto registration = ReadHostRegistrationStatus(root);
	JsonKey(writer, "host_registration");
	writer.start_object();
	JsonField(writer, "status", registration.active ? "active" : "unknown", true);
	JsonField(writer, "large_system_pages_opt_in", registration.large_system_pages);
	JsonField(writer, "mapping_evidence", "not_measured");
	JsonField(writer, "evidence", registration.evidence);
	writer.end_object();
	JsonKey(writer, "libcuda");
	writer.start_object();
	JsonField(writer, "path", lib, true);
	JsonField(writer, "version", lib ? host::SonameVersion(*lib, "libcuda.so") : std::nullopt);
	writer.end_object();
	JsonField(writer, "cuda_driver_api", driver_api ? CudaVersionString(*driver_api) : std::nullopt);
	JsonField(writer, "cuda_runtime", runtime ? CudaVersionString(*runtime) : std::nullopt);
	if (module_sets)
	{
		const auto status = cmt::nv::InspectKernelPatches(root, *module_sets);
		JsonKey(writer, "patch_status");
		writer.start_object();
		JsonField(writer, "open_kernel", status.open_kernel, true);
		JsonField(writer, "patched", status.patched);
		JsonField(writer, "set", status.set);
		JsonField(writer, "patches", status.patches);
		JsonKey(writer, "matching_sets");
		WriteJsonStringArray(writer, status.matching_sets);
		JsonField(writer, "evidence", status.evidence);
		writer.end_object();
	}
	writer.end_object();
}

} // namespace nv
} // namespace cmt

namespace cmt {
namespace nv {
KernelPatchStatus InspectKernelPatches(const std::string& root, const std::string& sets_directory)
{
	KernelPatchStatus result;
	const auto module = cmt::nv::NvidiaModule(root);
	const auto uvm = cmt::host::ReadKernelModule(root, "nvidia_uvm");
	if (module.banner && module.banner->find("NVIDIA UNIX Open Kernel Module") != std::string::npos)
	{
		result.open_kernel = true;
	}
	else if (module.banner && module.banner->find("NVIDIA UNIX") != std::string::npos &&
			 module.banner->find("Kernel Module") != std::string::npos)
	{
		result.open_kernel = false;
	}
	if (!module.srcversion || !uvm.srcversion)
	{
		result.evidence = "both loaded nvidia and nvidia_uvm srcversion values are required";
		return result;
	}
	const auto sets = ListMetadataDirectory(sets_directory);
	if (!sets)
	{
		result.evidence =
			"module-set directory is unreadable; pass --module-sets with kmod.sh's sets directory";
		return result;
	}
	for (const auto& name : *sets)
	{
		const auto text = ReadMetadataText(sets_directory + "/" + name + "/SRCVERSIONS");
		if (!text)
		{
			continue;
		}
		std::optional<std::string> nv_hash, uvm_hash;
		bool valid = true;
		std::istringstream lines(*text);
		for (std::string line; std::getline(lines, line);)
		{
			if (line.find_first_not_of(" \t\r") == std::string::npos)
			{
				continue;
			}
			std::istringstream words(line);
			std::string file, hash, extra;
			if (!(words >> file >> hash) || (words >> extra))
			{
				valid = false;
				break;
			}
			auto* slot = file == "nvidia.ko" ? &nv_hash : file == "nvidia-uvm.ko" ? &uvm_hash : nullptr;
			if (!slot || *slot || hash.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
			{
				valid = false;
				break;
			}
			*slot = hash;
		}
		if (valid && nv_hash == module.srcversion && uvm_hash == uvm.srcversion)
		{
			result.matching_sets.push_back(name);
		}
	}
	if (std::size(result.matching_sets) != 1)
	{
		result.evidence = std::empty(result.matching_sets)
							  ? "no manifest matches both loaded module source hashes"
							  : "multiple manifests match; patch provenance is ambiguous";
		return result;
	}
	result.set = result.matching_sets.front();
	result.patches = ReadMetadataLine(sets_directory + "/" + *result.set + "/PATCHES");
	if (!result.patches || std::empty(*result.patches))
	{
		result.evidence = "source hashes match, but the set's PATCHES description is missing";
		return result;
	}
	result.patched = *result.patches != "none";
	result.evidence =
		"both loaded module source hashes match this local set; PATCHES supplies its patch description";
	return result;
}
} // namespace nv
} // namespace cmt
