#include "common/helpers.h"
#include "machine.h"
#include "cuda/cuda.h"
#include "common/aliases.h"
#include "common/error.h"

namespace cmt {

namespace {

std::string ReadLine(const std::string& path)
{
	const auto text = ReadTextFile(path).value_or("");
	return text.substr(0, text.find('\n'));
}

std::string Link(const std::string& dev, const char* kind)
{
	const std::string sp = ReadLine(dev + "/" + kind + "_link_speed"),
					  w = ReadLine(dev + "/" + kind + "_link_width");
	if (std::empty(sp))
	{
		return "";
	}
	return sp + " x" + w;
}

std::string Lower(std::string s)
{
	for (auto& c : s)
	{
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return s;
}

} // namespace

Result<DeviceInfo> QueryDevice(int ordinal)
{
	DeviceInfo d;
	d.ordinal = ordinal;
	CMT_CUDA(cudaSetDevice(ordinal));
	cudaDeviceProp p{};
	CMT_CUDA(cudaGetDeviceProperties(&p, ordinal));
	d.name = p.name;
	d.cc_major = p.major, d.cc_minor = p.minor;
	d.sms = p.multiProcessorCount;
	d.memory_bytes = p.totalGlobalMem;
	char bus[32] = {0};
	CMT_CUDA(cudaDeviceGetPCIBusId(bus, sizeof bus, ordinal));
	d.pci_bus_id =
		Lower(bus);
	const std::string dev = "/sys/bus/pci/devices/" + d.pci_bus_id;
	d.link_max = Link(dev, "max");
	d.link_current = Link(dev, "current");
	CMT_CUDA(cudaDriverGetVersion(&d.driver_version));
	CMT_CUDA(cudaRuntimeGetVersion(&d.runtime_version));
	d.host_cpu = CpuModelName().value_or("");
	utsname u{};
	if (!uname(&u))
	{
		d.kernel = u.release;
	}
	return d;
}

std::string Describe(const DeviceInfo& d)
{
	std::ostringstream o;
	o << "device: " << d.ordinal << "\n"
	  << "name: " << d.name << "\n"
	  << "compute_capability: " << d.cc_major << "." << d.cc_minor << "\n"
	  << "sms: " << d.sms << "\n"
	  << "memory_gib: " << (d.memory_bytes >> 30) << "\n"
	  << "link_max: " << d.link_max << "\n"
	  << "link_current: " << d.link_current << "\n"
	  << "cuda_driver: " << d.driver_version / 1000 << "." << (d.driver_version % 1000) / 10 << "\n"
	  << "cuda_runtime: " << d.runtime_version / 1000 << "." << (d.runtime_version % 1000) / 10 << "\n"
	  << "host_cpu: " << d.host_cpu << "\n"
	  << "kernel: " << d.kernel << "\n";
	return o.str();
}

} // namespace cmt

namespace cmt {

namespace {

constexpr const char* kRoot = "/";

std::optional<int> Attribute(cudaDeviceAttr a, int ordinal)
{
	int v = 0;
	if (cudaDeviceGetAttribute(&v, a, ordinal) != cudaSuccess)
	{
		(void)cudaGetLastError();
		return std::nullopt;
	}
	return v;
}

std::optional<bool> Flag(cudaDeviceAttr a, int ordinal)
{
	const auto v = Attribute(a, ordinal);
	return v ? std::optional<bool>(*v != 0) : std::nullopt;
}

std::string UuidText(const cudaUUID_t& u)
{
	std::string hex;
	for (std::size_t i = 0; i < sizeof u.bytes; i++)
	{
		hex += fmt::format("{:02x}", static_cast<unsigned char>(u.bytes[i]));
	}
	return fmt::format("GPU-{}-{}-{}-{}-{}", hex.substr(0, 8), hex.substr(8, 4), hex.substr(12, 4),
					   hex.substr(16, 4), hex.substr(20));
}

void WriteGpuJson(JsonWriter& writer, int ordinal, bool nvml_query)
{
	writer.start_object();
	JsonField(writer, "ordinal", ordinal, true);
	cudaDeviceProp p{};
	const bool props = cudaGetDeviceProperties(&p, ordinal) == cudaSuccess;
	if (!props)
	{
		(void)cudaGetLastError();
	}
	JsonField(writer, "name", props ? std::optional<std::string_view>(p.name) : std::nullopt);
	JsonField(writer, "uuid", props ? std::optional<std::string>(UuidText(p.uuid)) : std::nullopt);
	char bus[32] = {0};
	std::optional<std::string> bdf;
	if (cudaDeviceGetPCIBusId(bus, sizeof bus, ordinal) == cudaSuccess)
	{
		bdf = Lower(bus);
	}
	else
	{
		(void)cudaGetLastError();
	}
	JsonField(writer, "pci_bus_id", bdf);
	JsonField(writer, "compute_capability",
			  props ? std::optional<std::string>(fmt::format("{}.{}", p.major, p.minor)) : std::nullopt);
	JsonField(writer, "gcn_arch", nullptr);
	JsonField(writer, "compute_units", props ? std::optional<int>(p.multiProcessorCount) : std::nullopt);
	JsonField(writer, "memory_bytes", props ? std::optional<std::size_t>(p.totalGlobalMem) : std::nullopt);
	const bool has_pcie_link = host::WritePciDeviceFields(writer, kRoot, bdf);
	JsonKey(writer, "cpu_affinity");
	writer.start_object();
	const auto nvml = !nvml_query
						  ? Result<std::vector<unsigned>>(Failure(Error::Kind::Unsupported, "not queried"))
					  : bdf ? NvmlGpuCpus(*bdf)
							: Result<std::vector<unsigned>>(UsageError("no PCI address"));
	const auto sysfs =
		bdf ? GpuLocalCpus(*bdf, {}) : GpuCpuSet{{}, std::nullopt, CpuSource::None, {}, "no PCI address"};
	const auto list = [](const std::vector<unsigned>& cpus)
	{ return std::empty(cpus) ? std::nullopt : std::optional<std::string>(FormatCpuList(cpus)); };
	JsonField(writer, "nvml_cpus", nvml ? list(*nvml) : std::nullopt, true);
	JsonField(writer, "nvml_error", nvml ? std::nullopt : std::optional<std::string>(nvml.error().Message()));
	JsonField(writer, "sysfs_numa_node", sysfs.node);
	JsonField(writer, "sysfs_cpus", list(sysfs.cpus));
	JsonField(writer, "sysfs_source",
			  sysfs.source == CpuSource::None ? std::nullopt
											  : std::optional<std::string>(CpuSourceName(sysfs.source)));
	JsonField(writer, "sysfs_error",
			  std::empty(sysfs.sysfs_error) ? std::nullopt : std::optional<std::string>(sysfs.sysfs_error));
	writer.end_object();
	const auto atomics = Flag(cudaDevAttrHostNativeAtomicSupported, ordinal);
	const auto ats = Flag(cudaDevAttrPageableMemoryAccessUsesHostPageTables, ordinal);
	const auto integrated = Flag(cudaDevAttrIntegrated, ordinal);
#if defined(__aarch64__)
	constexpr bool aarch64 = true;
#else
	constexpr bool aarch64 = false;
#endif
	const nv::HostLinkVerdict link = nv::DecideHostLink(integrated, atomics, ats, has_pcie_link, aarch64);
	JsonKey(writer, "host_link");
	writer.start_object();
	JsonField(writer, "kind", nv::HostLinkName(link.kind), true);
	JsonField(writer, "evidence", link.evidence);
	writer.end_object();
	JsonField(writer, "pageable_memory_access", Flag(cudaDevAttrPageableMemoryAccess, ordinal));
	writer.end_object();
}

} // namespace

Result<MachineReport> MachineReportFor(int device, bool all_devices,
									   const std::optional<std::string>& module_sets, bool nvml)
{
	MachineReport report;
	std::optional<std::string> gpu_error;
	int n = 0;
	const cudaError_t error = cudaGetDeviceCount(&n);
	if (error == cudaErrorNoDevice || (error == cudaSuccess && n == 0))
	{
		(void)cudaGetLastError();
		report.exit_code = 77;
		n = 0;
	}
	else if (error != cudaSuccess)
	{
		(void)cudaGetLastError();
		gpu_error =
			fmt::format("cudaGetDeviceCount: {} ({})", cudaGetErrorString(error), static_cast<int>(error));
		report.exit_code = 5;
		n = 0;
	}
	else if (device >= n)
	{
		return UsageError("--device selects an ordinal outside the visible GPU range");
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", kMachineSchema, true);
	JsonKey(writer, "tool");
	WriteToolJson(writer);
	JsonKey(writer, "host");
	host::WriteHostJson(writer, kRoot, host::ReadCpuid());
	JsonKey(writer, "gpus");
	writer.start_array();
	bool first = true;
	for (int i = 0; i < n; ++i)
	{
		if (all_devices || i == device)
		{
			if (!first)
			{
				writer.append_comma();
			}
			WriteGpuJson(writer, i, nvml);
			first = false;
		}
	}
	writer.end_array();
	JsonField(writer, "gpu_error", gpu_error);
	int drv = 0, rt = 0;
	const bool have_drv = cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0;
	const bool have_rt = cudaRuntimeGetVersion(&rt) == cudaSuccess && rt > 0;
	(void)cudaGetLastError();
	JsonKey(writer, "driver");
	nv::WriteDriverJson(writer, kRoot, ReadTextFile("/proc/self/maps").value_or(""),
						have_drv ? std::optional<int>(drv) : std::nullopt,
						have_rt ? std::optional<int>(rt) : std::nullopt, module_sets);
	writer.end_object();
	auto text = FinishJson(writer);
	if (!text)
	{
		return cmt::Failure(text.error());
	}
	report.text = std::move(*text);
	return report;
}

} // namespace cmt
