#pragma once

#include "common/aliases.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/error.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/utsname.h>
#include <utility>
#include <vector>

namespace cmt {

struct DeviceInfo
{
	int ordinal = 0;
	std::string name;
	int cc_major = 0;
	int cc_minor = 0;
	int sms = 0;
	std::size_t memory_bytes = 0;
	std::string pci_bus_id;
	std::string link_max;
	std::string link_current;
	int driver_version = 0;
	int runtime_version = 0;
	std::string host_cpu;
	std::string kernel;
};

Result<DeviceInfo> QueryDevice(int ordinal);

std::string Describe(const DeviceInfo& d);

} // namespace cmt

namespace cmt {

inline constexpr const char* kMachineSchema = "cmt-machine 1";

void WriteToolJson(JsonWriter& writer);

std::vector<std::string> MachineMissingKeys(const Json& doc);

struct MachineReport
{
	std::string text;
	int exit_code = 0;
};

Result<MachineReport> MachineReportFor(int device, bool all_devices,
									   const std::optional<std::string>& module_sets = std::nullopt,
									   bool nvml = false);

} // namespace cmt

namespace cmt {
namespace host {

std::optional<std::vector<std::size_t>> ParseIndexList(const std::string& list);
std::optional<std::string> Bracketed(const std::string& line);
std::optional<std::string> CommandLineParam(const std::string& cmdline, const std::string& key);
std::optional<std::string> OsReleaseValue(const std::string& os_release, const std::string& key);

std::optional<std::string> MappedLibrary(const std::string& maps, const std::string& stem);
std::optional<std::string> SonameVersion(const std::string& path, const std::string& stem);

struct KernelModule
{
	std::optional<std::string> version;
	std::optional<std::string> srcversion;
	bool loaded = false;
};

KernelModule ReadKernelModule(const std::string& root, const std::string& name);

struct Cpuid
{
	std::optional<bool> hypervisor_bit;
	std::optional<std::string> vendor;
};

Cpuid ReadCpuid();

struct VmEvidence
{
	Cpuid cpuid;
	std::optional<std::string> sys_vendor;
	std::optional<std::string> product_name;
	std::optional<std::string> board_vendor;
	std::optional<std::string> bios_vendor;
	std::optional<std::string> sys_hypervisor_type;
	std::optional<std::string> device_tree_hypervisor;
};

VmEvidence ReadVmEvidence(const std::string& root, const Cpuid& cpuid);

struct Verdict
{
	std::optional<std::string> type;
	std::string evidence;
};

Verdict ClassifyVm(const VmEvidence& e);
Verdict DetectContainer(const std::string& root);

struct PcieHop
{
	std::string bdf;
	std::string max_speed;
	std::string max_width;
	std::string current_speed;
	std::string current_width;
	std::string link_max;
	std::string link_current;
};

std::vector<PcieHop> PciePath(const std::string& pci_devices, const std::string& bdf);
double LinkGbps(const std::string& link);

void WriteHostJson(JsonWriter& writer, const std::string& root, const Cpuid& cpuid);
void WritePciDeviceJson(JsonWriter& writer, const std::string& root, const std::string& bdf);
bool WritePciDeviceFields(JsonWriter& writer, const std::string& root, const std::optional<std::string>& bdf,
						  bool first = false);

} // namespace host
} // namespace cmt

namespace cmt {
namespace nv {

struct Module
{
	std::optional<std::string> version;
	std::optional<std::string> srcversion;
	std::optional<std::string> banner;
};

Module NvidiaModule(const std::string& root = "/");
std::optional<std::string> ReleaseInBanner(const std::string& banner);

enum class HostLink : std::uint8_t
{
	Pcie,
	NvlinkC2c,
	Nvlink,
	Integrated,
	Unknown
};
const char* HostLinkName(HostLink k);

struct HostLinkVerdict
{
	HostLink kind = HostLink::Unknown;
	std::string evidence;
};

HostLinkVerdict DecideHostLink(std::optional<bool> integrated, std::optional<bool> native_atomics,
							   std::optional<bool> uses_host_page_tables, bool has_pcie_link, bool aarch64);

std::optional<std::string> CudaVersionString(int packed);

void WriteDriverJson(JsonWriter& writer, const std::string& root, const std::string& maps,
					 std::optional<int> driver_api, std::optional<int> runtime,
					 const std::optional<std::string>& module_sets = std::nullopt);

} // namespace nv
} // namespace cmt

namespace cmt {
namespace nv {
struct KernelPatchStatus
{
	std::optional<bool> open_kernel;
	std::optional<bool> patched;
	std::optional<std::string> set;
	std::optional<std::string> patches;
	std::vector<std::string> matching_sets;
	std::string evidence;
};

KernelPatchStatus InspectKernelPatches(const std::string& root, const std::string& sets_directory);
} // namespace nv
} // namespace cmt
