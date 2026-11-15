#include "cpu.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

#include "defines.h"
#include "error.h"
#include "helpers.h"

#include "log.h"

#include <dlfcn.h>

namespace cmt {

namespace {

void FreeCpuMask(cpu_set_t* mask)
{
	CPU_FREE(mask);
}

using CpuMask = std::unique_ptr<cpu_set_t, decltype(&FreeCpuMask)>;

CpuMask AllocateCpuMask(std::size_t count)
{
	CpuMask mask(CPU_ALLOC(count), &FreeCpuMask);
	if (!mask)
	{
		Panic("Cannot allocate CPU affinity mask");
	}
	CPU_ZERO_S(CPU_ALLOC_SIZE(count), mask.get());
	return mask;
}

Result<unsigned> ParseCpuNumber(std::string_view text)
{
	unsigned value = 0;
	const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
	if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
	{
		return UsageError("CPU list requires nonnegative integers and ascending ranges");
	}
	return value;
}

} // namespace

Result<std::vector<unsigned>> ParseThreadCpus(std::string_view text)
{
	if (text.empty())
	{
		return UsageError("CPU list must not be empty");
	}
	const long configured = sysconf(_SC_NPROCESSORS_CONF);
	if (configured <= 0)
	{
		return OsError("Cannot read configured CPU count", EIO);
	}
	std::vector<unsigned> cpus;
	while (!text.empty())
	{
		const auto comma = text.find(',');
		const auto field = text.substr(0, comma);
		const auto dash = field.find('-');
		auto first = ParseCpuNumber(field.substr(0, dash));
		if (!first)
		{
			return Failure(first.error());
		}
		auto last = dash == std::string_view::npos ? first : ParseCpuNumber(field.substr(dash + 1));
		if (!last)
		{
			return Failure(last.error());
		}
		if (*last < *first)
		{
			return UsageError("CPU list contains a descending range");
		}
		const auto count = std::uint64_t(*last) - *first + 1;
		if (count > static_cast<std::uint64_t>(configured))
		{
			return UsageError("CPU range contains more entries than the machine has configured CPUs");
		}
		for (std::uint64_t cpu = *first; cpu <= *last; ++cpu)
		{
			cpus.push_back(static_cast<unsigned>(cpu));
		}
		if (comma == std::string_view::npos)
		{
			break;
		}
		text.remove_prefix(comma + 1);
		if (text.empty())
		{
			return UsageError("CPU list has an empty entry");
		}
	}
	std::sort(cpus.begin(), cpus.end());
	cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
	return cpus;
}

namespace {

Result<std::vector<unsigned>> ReadCpuListFile(const std::string& path);
Result<std::vector<unsigned>> NumaCpusWithin(int node, std::vector<unsigned> allowed,
											 std::string_view sysfs = "/sys");

Result<std::vector<unsigned>> PlacementCpus(const ThreadPlacement& placement)
{
	auto allowed = placement.allowed_cpus.empty() ? GetThreadAffinity()
												  : Result<std::vector<unsigned>>(placement.allowed_cpus);
	if (!allowed)
	{
		return Failure(allowed.error());
	}
	std::sort(allowed->begin(), allowed->end());
	allowed->erase(std::unique(allowed->begin(), allowed->end()), allowed->end());
	std::vector<unsigned> cpus = placement.cpus.empty() ? *allowed : placement.cpus;
	std::sort(cpus.begin(), cpus.end());
	cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
	if (!std::includes(allowed->begin(), allowed->end(), cpus.begin(), cpus.end()))
	{
		return UsageError("Worker CPU placement exceeds the allowed affinity mask");
	}
	if (placement.numa_node)
	{
		auto node_cpus = NumaCpusWithin(*placement.numa_node, *allowed);
		if (!node_cpus)
		{
			return Failure(node_cpus.error());
		}
		cpus.erase(std::remove_if(cpus.begin(), cpus.end(), [&](unsigned cpu)
								  { return !std::binary_search(node_cpus->begin(), node_cpus->end(), cpu); }),
				   cpus.end());
	}
	if (cpus.empty())
	{
		return UsageError("Worker CPU and NUMA placement has no allowed CPU");
	}
	return cpus;
}

} // namespace

Result<std::vector<unsigned>> GetThreadAffinity(pthread_t thread)
{
	std::size_t count = CPU_SETSIZE;
	for (;;)
	{
		auto mask = AllocateCpuMask(count);
		const auto bytes = CPU_ALLOC_SIZE(count);
		const int code = pthread_getaffinity_np(thread, bytes, mask.get());
		if (code == 0)
		{
			std::vector<unsigned> cpus;
			cpus.reserve(static_cast<std::size_t>(CPU_COUNT_S(bytes, mask.get())));
			for (std::size_t cpu = 0; cpu < count; ++cpu)
			{
				if (CPU_ISSET_S(cpu, bytes, mask.get()))
				{
					cpus.push_back(static_cast<unsigned>(cpu));
				}
			}
			return cpus;
		}
		if (code != EINVAL)
		{
			return OsError("pthread_getaffinity_np failed", code);
		}
		if (count > static_cast<std::size_t>(INT_MAX) / 2)
		{
			return Unsupported("Kernel CPU affinity mask exceeds the supported index range");
		}
		count *= 2;
	}
}

Result<void> SetThreadAffinity(pthread_t thread, const std::vector<unsigned>& cpus)
{
	if (cpus.empty())
	{
		return UsageError("Cannot set an empty thread affinity");
	}
	auto selected = PlacementCpus(ThreadPlacement{cpus, std::nullopt});
	if (!selected)
	{
		return Failure(selected.error());
	}
	return RestoreThreadAffinity(thread, *selected);
}

Result<void> RestoreThreadAffinity(pthread_t thread, const std::vector<unsigned>& cpus)
{
	if (cpus.empty())
	{
		return UsageError("Cannot set an empty thread affinity");
	}
	const auto count = std::size_t(*std::max_element(std::begin(cpus), std::end(cpus))) + 1;
	auto mask = AllocateCpuMask(count);
	const auto bytes = CPU_ALLOC_SIZE(count);
	for (unsigned cpu : cpus)
	{
		CPU_SET_S(cpu, bytes, mask.get());
	}
	const int code = pthread_setaffinity_np(thread, bytes, mask.get());
	if (code != 0)
	{
		return OsError("pthread_setaffinity_np failed", code);
	}
	return {};
}

namespace {

Result<std::vector<unsigned>> NumaCpusWithin(int node, std::vector<unsigned> allowed, std::string_view sysfs)
{
	if (node < 0)
	{
		return UsageError("NUMA node must be nonnegative");
	}
	auto cpus = ReadCpuListFile(fmt::format("{}/devices/system/node/node{}/cpulist", sysfs, node));
	if (!cpus)
	{
		const auto& error = cpus.error();
		if (error.GetKind() == Error::Kind::Os && (error.GetCode() == ENOENT || error.GetCode() == ENOTDIR))
		{
			auto topology = IsRegularFile(fmt::format("{}/devices/system/node/online", sysfs));
			if (!topology)
			{
				return Failure(topology.error());
			}
			if (!*topology)
			{
				return Unsupported("NUMA CPU placement is unavailable on this machine");
			}
			return UsageError("Requested NUMA node does not exist");
		}
		if (error.IsUnsupported())
		{
			return UsageError("Requested NUMA node has no CPU in the allowed affinity");
		}
		if (error.IsUsage())
		{
			return Failure(Error::Kind::Verify, "NUMA node CPU list is malformed");
		}
		return Failure(error);
	}
	allowed.erase(std::remove_if(allowed.begin(), allowed.end(), [&](unsigned cpu)
								 { return !std::binary_search(cpus->begin(), cpus->end(), cpu); }),
				  allowed.end());
	if (allowed.empty())
	{
		return UsageError("Requested NUMA node has no CPU in the allowed affinity");
	}
	return allowed;
}

} // namespace

Result<std::vector<unsigned>> NumaNodeCpus(int node, std::string_view sysfs)
{
	if (node < 0)
	{
		return UsageError("NUMA node must be nonnegative");
	}
	auto allowed = GetThreadAffinity();
	if (!allowed)
	{
		return Failure(allowed.error());
	}
	return NumaCpusWithin(node, std::move(*allowed), sysfs);
}

constexpr std::size_t kCpuListBytes = 64 * 1024;

unsigned CpuNodeCount(std::string_view sysfs)
{
	const auto nodes = ReadCpuListFile(fmt::format("{}/devices/system/node/has_cpu", sysfs));
	return nodes ? static_cast<unsigned>(std::max<std::size_t>(std::size(*nodes), 1)) : 1;
}

Result<bool> ParseWorkerPlacement(std::string_view name, std::string_view value, WorkerPlacement& placement)
{
	if (name.rfind("CMT_WORKER_", 0) != 0)
	{
		return false;
	}
	if (name == "CMT_WORKER_CPUS")
	{
		if (!std::empty(placement.cpus))
		{
			return UsageError("duplicate CMT_WORKER_CPUS");
		}
		auto cpus = ParseThreadCpus(value);
		if (!cpus)
		{
			return Failure(
				Error::Kind::Usage,
				"CMT_WORKER_CPUS is not a CPU list (IDs and ascending ranges, e.g. 2,4-7, at most the "
				"configured CPUs)",
				cpus.error().GetCode());
		}
		placement.cpus = std::move(*cpus);
	}
	else if (name == "CMT_WORKER_NUMA_NODE")
	{
		int node = -1;
		const auto end = std::data(value) + std::size(value);
		const auto parsed = std::from_chars(std::data(value), end, node);
		if (placement.numa_node || std::empty(value) || parsed.ec != std::errc{} || parsed.ptr != end ||
			node < 0)
		{
			return UsageError("CMT_WORKER_NUMA_NODE must be one nonnegative integer");
		}
		placement.numa_node = node;
	}
	else if (name == "CMT_WORKER_NUMA")
	{
		if (placement.gpu_numa || (value != "0" && value != "1"))
		{
			return UsageError("CMT_WORKER_NUMA must be one 0 or 1");
		}
		placement.gpu_numa = value == "1";
	}
	else
	{
		return UsageError(
			"unknown CMT_WORKER_ variable: CMT_WORKER_CPUS, CMT_WORKER_NUMA_NODE or CMT_WORKER_NUMA");
	}
	return true;
}

Result<WorkerPlacement> WorkerPlacementFromEnv()
{
	WorkerPlacement placement;
	for (const char* name : {"CMT_WORKER_CPUS", "CMT_WORKER_NUMA_NODE", "CMT_WORKER_NUMA"})
	{
		if (const char* value = std::getenv(name))
		{
			auto parsed = ParseWorkerPlacement(name, value, placement);
			if (!parsed)
			{
				return Failure(parsed.error());
			}
		}
	}
	return placement;
}

Result<ThreadPlacement> ThreadPlacementFromEnv()
{
	auto request = WorkerPlacementFromEnv();
	if (!request)
	{
		return Failure(request.error());
	}
	return ThreadPlacement{std::move(request->cpus), request->numa_node};
}

namespace {

bool HookDebug()
{
	const char* debug = std::getenv("MEMCPY_HOOK_DEBUG");
	return debug && *debug != '\0' && std::strcmp(debug, "0") != 0;
}

} // namespace

std::string FormatCpuList(const std::vector<unsigned>& cpus)
{
	std::string text;
	for (std::size_t i = 0; i < std::size(cpus);)
	{
		std::size_t last = i;
		while (last + 1 < std::size(cpus) && cpus[last + 1] == cpus[last] + 1)
		{
			last++;
		}
		text += fmt::format(last == i ? "{}{}" : "{}{}-{}", std::empty(text) ? "" : ",", cpus[i], cpus[last]);
		i = last + 1;
	}
	return text;
}

namespace {

Result<std::string> PciSysfsDir(const std::string& bdf, std::string_view sysfs);

struct Nvml
{
	using Device = void*;
	int (*by_pci)(const char*, Device*) = nullptr;
	int (*affinity)(Device, unsigned, unsigned long*) = nullptr;
	Error error{Error::Kind::Unsupported, "libnvidia-ml.so.1 is not loadable"};
	bool ready = false;
};

const Nvml& LoadNvml()
{
	static const Nvml nvml = []
	{
		Nvml loaded;
		void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
		if (!library)
		{
			return loaded;
		}
		const auto init = reinterpret_cast<int (*)()>(dlsym(library, "nvmlInit_v2"));
		loaded.by_pci = reinterpret_cast<int (*)(const char*, Nvml::Device*)>(
			dlsym(library, "nvmlDeviceGetHandleByPciBusId_v2"));
		loaded.affinity = reinterpret_cast<int (*)(Nvml::Device, unsigned, unsigned long*)>(
			dlsym(library, "nvmlDeviceGetCpuAffinity"));
		if (!init || !loaded.by_pci || !loaded.affinity)
		{
			loaded.error =
				Error(Error::Kind::Unsupported, "NVML lacks nvmlInit_v2, a PCI lookup or CPU affinity");
			return loaded;
		}
		const int status = init();
		loaded.error = Error(Error::Kind::Unsupported, "nvmlInit_v2 failed", status);
		loaded.ready = status == 0;
		return loaded;
	}();
	return nvml;
}

} // namespace

Result<std::vector<unsigned>> NvmlGpuCpus(const std::string& pci_bus_id)
{
	const Nvml& nvml = LoadNvml();
	if (!nvml.ready)
	{
		return Failure(nvml.error);
	}
	Nvml::Device device = nullptr;
	int status = nvml.by_pci(pci_bus_id.c_str(), &device); // by PCI address: indices differ between APIs
	if (status != 0)
	{
		return Failure(Error::Kind::Unsupported, "NVML has no device at this PCI address", status);
	}
	constexpr std::size_t kBits = sizeof(unsigned long) * CHAR_BIT;
	const long configured = sysconf(_SC_NPROCESSORS_CONF);
	const std::size_t words = (static_cast<std::size_t>(std::max(configured, 1L)) + kBits - 1) / kBits;
	std::vector<unsigned long> mask(words);
	status = nvml.affinity(device, static_cast<unsigned>(words), std::data(mask));
	if (status != 0)
	{
		return Failure(Error::Kind::Unsupported, "nvmlDeviceGetCpuAffinity failed", status);
	}
	std::vector<unsigned> cpus;
	for (std::size_t cpu = 0; cpu < words * kBits; cpu++)
	{
		if ((mask[cpu / kBits] >> (cpu % kBits)) & 1UL)
		{
			cpus.push_back(static_cast<unsigned>(cpu));
		}
	}
	if (std::empty(cpus))
	{
		return Failure(Error::Kind::Unsupported, "NVML reports no CPU for the GPU");
	}
	return cpus;
}

const char* CpuSourceName(CpuSource source)
{
	switch (source)
	{
	case CpuSource::None:
		return "none";
	case CpuSource::Nvml:
		return "nvml";
	case CpuSource::LocalCpulist:
		return "sysfs local_cpulist";
	case CpuSource::NodeCpulist:
		return "sysfs node cpulist";
	}
	return "none";
}

GpuCpuSet GpuLocalCpus(const std::string& pci_bus_id, const GpuCpus& nvml, std::string_view sysfs)
{
	GpuCpuSet set;
	const auto note = [&](std::string_view step, const Error& error)
	{
		set.sysfs_error +=
			fmt::format("{}{}: {}", std::empty(set.sysfs_error) ? "" : "; ", step, error.Message());
	};
	const auto node = PciNumaNode(pci_bus_id, sysfs);
	if (node)
	{
		set.node = *node;
	}
	auto listed =
		nvml ? nvml(pci_bus_id)
			 : Result<std::vector<unsigned>>(Failure(Error::Kind::Unsupported, "NVML not consulted"));
	if (listed)
	{
		set.cpus = std::move(*listed);
		set.source = CpuSource::Nvml;
		return set;
	}
	set.nvml_error = listed.error().Message();
	const auto dir = PciSysfsDir(pci_bus_id, sysfs);
	auto local =
		dir ? ReadCpuListFile(*dir + "/local_cpulist") : Result<std::vector<unsigned>>(Failure(dir.error()));
	if (local)
	{
		set.cpus = std::move(*local);
		set.source = CpuSource::LocalCpulist;
		return set;
	}
	note("local_cpulist", local.error());
	if (!node)
	{
		note("numa_node", node.error());
		return set;
	}
	if (!*node)
	{
		set.sysfs_error += "; numa_node: unknown (-1 or missing)";
		return set;
	}
	auto listed_node = ReadCpuListFile(fmt::format("{}/devices/system/node/node{}/cpulist", sysfs, **node));
	if (listed_node)
	{
		set.cpus = std::move(*listed_node);
		set.source = CpuSource::NodeCpulist;
		return set;
	}
	note(fmt::format("node{} cpulist", **node), listed_node.error());
	return set;
}

GpuPlacement GpuLocalPlacement(const std::string& pci_bus_id, const GpuCpus& nvml, std::string_view sysfs)
{
	GpuPlacement placement;
	placement.local = GpuLocalCpus(pci_bus_id, nvml, sysfs);
	const auto& local = placement.local;
	if (std::empty(local.cpus))
	{
		placement.reason = fmt::format("the CPUs of GPU {} are unknown (NVML: {}; sysfs: {})", pci_bus_id,
									   local.nvml_error, local.sysfs_error);
		return placement;
	}
	const auto where = fmt::format("{}{}", CpuSourceName(local.source),
								   local.node ? fmt::format(", sysfs node {}", *local.node) : std::string{});
	auto allowed = GetThreadAffinity();
	if (!allowed)
	{
		placement.reason = fmt::format("this thread's CPUs are unknown ({})", allowed.error().Message());
		return placement;
	}
	std::set_intersection(std::begin(local.cpus), std::end(local.cpus), std::begin(*allowed),
						  std::end(*allowed), std::back_inserter(placement.cpus));
	placement.off_gpu = std::empty(placement.cpus);
	placement.reason = placement.off_gpu
						   ? fmt::format("GPU {} is local to CPUs {} ({}), where this process may "
										 "not run",
										 pci_bus_id, FormatCpuList(local.cpus), where)
						   : where;
	return placement;
}

Result<ThreadPlacement>
ResolveWorkerPlacement(const WorkerPlacement& request,
					   const std::function<Result<std::optional<std::string>>()>& gpu_pci,
					   std::string_view sysfs, const GpuCpus& nvml)
{
	if (!std::empty(request.cpus) || request.numa_node)
	{
		return ThreadPlacement{request.cpus, request.numa_node};
	}
	if (request.gpu_numa == false || CpuNodeCount(sysfs) < 2)
	{
		return ThreadPlacement{};
	}
	const auto inherit = [&](const std::string& why, bool warn = false)
	{
		static std::atomic<bool> said{false};
		if (warn ? !said.exchange(true, std::memory_order_relaxed) : HookDebug())
		{
			logging::WriteLine(STDERR_FILENO, "[cmt] ", "worker NUMA: {}; workers inherit the caller's CPUs",
							   why);
		}
		return Result<ThreadPlacement>{ThreadPlacement{}};
	};
	auto pci = gpu_pci();
	if (!pci)
	{
		return Failure(pci.error());
	}
	if (!*pci)
	{
		return inherit("no current GPU");
	}
	auto local = GpuLocalPlacement(**pci, nvml, sysfs);
	if (local.off_gpu)
	{
		return inherit(
			fmt::format("{}, so staged GPU->CPU copies run about 35% slower; to run there: "
						"{}taskset -c {} <command>; `cmt run --numa off` (with LD_PRELOAD and no tune "
						"table, CMT_WORKER_NUMA=0) hides this",
						local.reason,
						local.local.node
							? fmt::format("numactl --cpunodebind={} <command> or ", *local.local.node)
							: "",
						FormatCpuList(local.local.cpus)),
			true);
	}
	if (std::empty(local.cpus))
	{
		return inherit(local.reason);
	}
	if (HookDebug())
	{
		logging::WriteLine(STDERR_FILENO, "[cmt] ",
						   "worker NUMA: GPU {}: workers run on {} of its CPUs ({}; no memory policy)", **pci,
						   std::size(local.cpus), local.reason);
	}
	ThreadPlacement placement{std::move(local.cpus), std::nullopt};
	placement.automatic = true;
	return placement;
}

namespace {

Result<std::string> PciSysfsDir(const std::string& bdf, std::string_view sysfs)
{
	const auto first_colon = bdf.find(':');
	const auto second_colon = bdf.find(':', first_colon == std::string::npos ? 0 : first_colon + 1);
	const auto dot = bdf.find('.', second_colon == std::string::npos ? 0 : second_colon + 1);
	if (first_colon == std::string::npos || second_colon == std::string::npos || dot == std::string::npos)
	{
		return UsageError("GPU PCI address must have domain:bus:device.function form");
	}
	const std::string_view text(bdf);
	const std::string_view fields[] = {
		text.substr(0, first_colon), text.substr(first_colon + 1, second_colon - first_colon - 1),
		text.substr(second_colon + 1, dot - second_colon - 1), text.substr(dot + 1)};
	unsigned values[4]{};
	const unsigned limits[] = {std::numeric_limits<unsigned>::max(), 255, 31, 7};
	for (unsigned i = 0; i < 4; ++i)
	{
		const auto parsed =
			std::from_chars(fields[i].data(), fields[i].data() + fields[i].size(), values[i], 16);
		if (fields[i].empty() || parsed.ec != std::errc{} ||
			parsed.ptr != fields[i].data() + fields[i].size() || values[i] > limits[i])
		{
			return UsageError("GPU PCI address has an invalid numeric field");
		}
	}
	return fmt::format("{}/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:x}", sysfs, values[0], values[1], values[2],
					   values[3]);
}

Result<std::vector<unsigned>> ReadCpuListFile(const std::string& path)
{
	auto text = ReadTextFile(path, kCpuListBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	std::string_view list(*text);
	while (!std::empty(list) && (list.back() == '\n' || list.back() == '\r'))
	{
		list.remove_suffix(1);
	}
	if (std::empty(list))
	{
		return Failure(Error::Kind::Unsupported, "the CPU list is empty");
	}
	return ParseThreadCpus(list);
}

} // namespace

Result<std::optional<int>> PciNumaNode(const std::string& bdf, std::string_view sysfs)
{
	const auto dir = PciSysfsDir(bdf, sysfs);
	if (!dir)
	{
		return Failure(dir.error());
	}
	auto contents = ReadTextFile(*dir + "/numa_node");
	if (!contents)
	{
		const auto& error = contents.error();
		if (error.GetKind() == Error::Kind::Os && (error.GetCode() == ENOENT || error.GetCode() == ENOTDIR))
		{
			return std::optional<int>{};
		}
		return Failure(error);
	}
	std::string_view node_text(*contents);
	while (!node_text.empty() && (node_text.back() == '\n' || node_text.back() == '\r'))
	{
		node_text.remove_suffix(1);
	}
	int node = -1;
	const auto parsed = std::from_chars(node_text.data(), node_text.data() + node_text.size(), node);
	if (node_text.empty() || parsed.ec != std::errc{} || parsed.ptr != node_text.data() + node_text.size() ||
		node < -1)
	{
		return Failure(Error::Kind::Verify, "GPU PCI NUMA node is malformed");
	}
	return node == -1 ? std::optional<int>{} : std::optional<int>{node};
}

Result<void> ApplyThreadPlacement(pthread_attr_t& attributes, const ThreadPlacement& placement)
{
	if (placement.cpus.empty() && !placement.numa_node)
	{
		return {};
	}
	auto cpus = PlacementCpus(placement);
	if (!cpus)
	{
		return Failure(cpus.error());
	}
	const auto count = std::size_t(cpus->back()) + 1;
	auto mask = AllocateCpuMask(count);
	const auto bytes = CPU_ALLOC_SIZE(count);
	for (unsigned cpu : *cpus)
	{
		CPU_SET_S(cpu, bytes, mask.get());
	}
	const int code = pthread_attr_setaffinity_np(&attributes, bytes, mask.get());
	if (code != 0)
	{
		return OsError("pthread_attr_setaffinity_np failed", code);
	}
	return {};
}

Result<void> Thread::StartTask(void* (*entry)(void*), void* task, const ThreadPlacement& placement)
{
	int code = 0;
	if (placement.cpus.empty() && !placement.numa_node)
	{
		code = pthread_create(&handle, nullptr, entry, task);
	}
	else
	{
		pthread_attr_t attributes;
		code = pthread_attr_init(&attributes);
		if (code != 0)
		{
			return OsError("pthread_attr_init failed", code);
		}
		const auto applied = ApplyThreadPlacement(attributes, placement);
		if (applied)
		{
			code = pthread_create(&handle, &attributes, entry, task);
		}
		const int destroyed = pthread_attr_destroy(&attributes);
		if (destroyed != 0)
		{
			Panic("Cannot destroy initialized pthread attributes");
		}
		if ((!applied || code != 0) && placement.automatic)
		{
			if (HookDebug())
			{
				logging::WriteLine(STDERR_FILENO, "[cmt] ",
								   "worker NUMA: cannot start a thread on the GPU's CPUs "
								   "({}); it inherits the caller's CPUs",
								   applied ? fmt::format("pthread_create: errno {}", code)
										   : std::string(applied.error().Message()));
			}
			code = pthread_create(&handle, nullptr, entry, task);
		}
		else if (!applied)
		{
			return Failure(applied.error());
		}
	}
	if (code != 0)
	{
		return OsError("pthread_create failed", code);
	}
	joinable = true;
	return {};
}

Result<void> PinThread(int cpu)
{
	if (cpu < 0)
	{
		return {};
	}
	return SetThreadAffinity(pthread_self(), {static_cast<unsigned>(cpu)});
}

namespace host {
namespace {
std::string TrimCpuField(const std::string& text)
{
	const auto first = text.find_first_not_of(" \t\r\n");
	if (first == std::string::npos)
	{
		return {};
	}
	return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::optional<std::string> CpuInfoField(const std::string& line, const std::string& key)
{
	if (line.rfind(key, 0) != 0)
	{
		return std::nullopt;
	}
	const std::size_t colon = line.find(':');
	if (colon == std::string::npos || TrimCpuField(line.substr(0, colon)) != key)
	{
		return std::nullopt;
	}
	return TrimCpuField(line.substr(colon + 1));
}
} // namespace

CpuInfo ParseCpuInfo(const std::string& cpuinfo)
{
	CpuInfo c;
	std::optional<std::string> implementer, part;
	std::istringstream in(cpuinfo);
	for (std::string l; std::getline(in, l);)
	{
		if (std::empty(TrimCpuField(l)) && (c.model || implementer))
		{
			break;
		}
		if (auto v = CpuInfoField(l, "model name"); v && !c.model)
		{
			c.model = v;
		}
		else if (auto vv = CpuInfoField(l, "vendor_id"); vv && !c.vendor)
		{
			c.vendor = vv;
		}
		else if (auto im = CpuInfoField(l, "CPU implementer"); im && !implementer)
		{
			implementer = im;
		}
		else if (auto p = CpuInfoField(l, "CPU part"); p && !part)
		{
			part = p;
		}
		else if (auto f = CpuInfoField(l, "flags"); f && !c.flags)
		{
			std::istringstream w(*f);
			c.flags.emplace();
			for (std::string x; w >> x;)
			{
				c.flags->push_back(x);
			}
		}
		else if (auto a = CpuInfoField(l, "Features"); a && !c.flags)
		{
			std::istringstream w(*a);
			c.flags.emplace();
			for (std::string x; w >> x;)
			{
				c.flags->push_back(x);
			}
			c.arm_features = true;
		}
	}
	if (!c.model && implementer)
	{
		c.model = fmt::format("arm64 implementer {} part {}", *implementer, part.value_or(""));
	}
	if (!c.vendor && implementer)
	{
		c.vendor = "implementer " + *implementer;
	}
	return c;
}

} // namespace host

Result<std::string> CpuModelName()
{
	auto text = ReadTextFile("/proc/cpuinfo");
	if (!text)
	{
		return Failure(text.error());
	}
	auto info = host::ParseCpuInfo(*text);
	if (!info.model)
	{
		return Unsupported("CPU model is unavailable");
	}
	return std::move(*info.model);
}

} // namespace cmt
