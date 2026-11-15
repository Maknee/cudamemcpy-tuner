#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <pthread.h>

#include "error.h"

namespace cmt {

struct ThreadPlacement
{
	std::vector<unsigned> cpus;
	std::optional<int> numa_node;
	std::vector<unsigned> allowed_cpus{};
	bool automatic = false;
};

[[nodiscard]] Result<std::vector<unsigned>> ParseThreadCpus(std::string_view text);

struct WorkerPlacement
{
	std::vector<unsigned> cpus;
	std::optional<int> numa_node;
	std::optional<bool> gpu_numa;

	bool Empty() const
	{
		return std::empty(cpus) && !numa_node && !gpu_numa;
	}
};

[[nodiscard]] Result<bool> ParseWorkerPlacement(std::string_view name, std::string_view value,
												WorkerPlacement& placement);
[[nodiscard]] Result<WorkerPlacement> WorkerPlacementFromEnv();
[[nodiscard]] Result<ThreadPlacement> ThreadPlacementFromEnv();

std::string FormatCpuList(const std::vector<unsigned>& cpus);

[[nodiscard]] Result<std::vector<unsigned>> NvmlGpuCpus(const std::string& pci_bus_id);
using GpuCpus = std::function<Result<std::vector<unsigned>>(const std::string& pci_bus_id)>;

enum class CpuSource : std::uint8_t
{
	None,
	Nvml,
	LocalCpulist,
	NodeCpulist,
};
const char* CpuSourceName(CpuSource source);

struct GpuCpuSet
{
	std::vector<unsigned> cpus;
	std::optional<int> node;
	CpuSource source = CpuSource::None;
	std::string nvml_error;
	std::string sysfs_error;
};

[[nodiscard]] GpuCpuSet GpuLocalCpus(const std::string& pci_bus_id, const GpuCpus& nvml = NvmlGpuCpus,
									 std::string_view sysfs = "/sys");

struct GpuPlacement
{
	std::vector<unsigned> cpus;
	GpuCpuSet local;
	std::string reason;
	bool off_gpu = false;
};

[[nodiscard]] GpuPlacement GpuLocalPlacement(const std::string& pci_bus_id, const GpuCpus& nvml,
											 std::string_view sysfs);

[[nodiscard]] Result<ThreadPlacement>
ResolveWorkerPlacement(const WorkerPlacement& request,
					   const std::function<Result<std::optional<std::string>>()>& gpu_pci,
					   std::string_view sysfs = "/sys", const GpuCpus& nvml = NvmlGpuCpus);
[[nodiscard]] Result<void> ApplyThreadPlacement(pthread_attr_t& attributes, const ThreadPlacement& placement);
[[nodiscard]] Result<std::optional<int>> PciNumaNode(const std::string& bdf, std::string_view sysfs = "/sys");
unsigned CpuNodeCount(std::string_view sysfs = "/sys");

[[nodiscard]] Result<std::vector<unsigned>> GetThreadAffinity(pthread_t thread = pthread_self());
[[nodiscard]] Result<void> SetThreadAffinity(pthread_t thread, const std::vector<unsigned>& cpus);
[[nodiscard]] Result<void> RestoreThreadAffinity(pthread_t thread, const std::vector<unsigned>& cpus);
[[nodiscard]] Result<std::vector<unsigned>> NumaNodeCpus(int node, std::string_view sysfs = "/sys");

Result<void> PinThread(int cpu);

class Thread
{
public:
	Thread() = default;
	Thread(const Thread&) = delete;
	Thread& operator=(const Thread&) = delete;

	Thread(Thread&& other) noexcept : handle(other.handle), joinable(std::exchange(other.joinable, false))
	{
	}

	Thread& operator=(Thread&& other) noexcept
	{
		if (this != &other)
		{
			JoinForDestruction();
			handle = other.handle;
			joinable = std::exchange(other.joinable, false);
		}
		return *this;
	}

	~Thread()
	{
		JoinForDestruction();
	}

	template <class Callable>
	[[nodiscard]] Result<void> Start(Callable&& function)
	{
		if (joinable)
		{
			Panic("Thread::Start called on a running thread");
		}
		auto placement = ThreadPlacementFromEnv();
		if (!placement)
		{
			return Failure(placement.error());
		}
		return Start(std::forward<Callable>(function), *placement);
	}

	template <class Callable>
	[[nodiscard]] Result<void> Start(Callable&& function, const ThreadPlacement& placement)
	{
		if (joinable)
		{
			Panic("Thread::Start called on a running thread");
		}
		using Task = std::decay_t<Callable>;
		auto task = std::unique_ptr<Task>(new (std::nothrow) Task(std::forward<Callable>(function)));
		if (!task)
		{
			Panic("Cannot allocate thread task");
		}
		auto started = StartTask(
			[](void* value) -> void*
			{
				std::unique_ptr<Task> owned(static_cast<Task*>(value));
				(*owned)();
				return nullptr;
			},
			task.get(), placement);
		if (!started)
		{
			return Failure(started.error());
		}
		task.release();
		return {};
	}

	[[nodiscard]] Result<void> Join()
	{
		if (!joinable)
		{
			return {};
		}
		const int code = pthread_join(handle, nullptr);
		if (code != 0)
		{
			return Failure(Error::Kind::Os, "pthread_join failed", code);
		}
		joinable = false;
		return {};
	}

	bool Joinable() const noexcept
	{
		return joinable;
	}

	pthread_t NativeHandle() const noexcept
	{
		return handle;
	}

private:
	[[nodiscard]] Result<void> StartTask(void* (*entry)(void*), void* task, const ThreadPlacement& placement);

	void JoinForDestruction() noexcept
	{
		auto joined = Join();
		if (!joined)
		{
			Panic(joined.error().Message());
		}
	}

	pthread_t handle{};
	bool joinable = false;
};

Result<std::string> CpuModelName();

namespace host {

struct CpuInfo
{
	std::optional<std::string> model;
	std::optional<std::string> vendor;
	std::optional<std::vector<std::string>> flags;
	bool arm_features = false;
};

CpuInfo ParseCpuInfo(const std::string& cpuinfo);

} // namespace host
} // namespace cmt
