#pragma once

// A sysfs tree with two CPU nodes, for the NUMA placement checks (test_sync, test_config, test_tuner): node 0
// holds the first CPU this thread may use, node 1 the rest, and the GPU 0000:01:00.0 is on node 1. The host
// running the test may have one node; the placement code reads only the tree it is given.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "check.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace check {

inline constexpr const char* kFixtureGpu = "0000:01:00.0";

struct TwoNodeSysfs
{
	std::string root;
	std::vector<unsigned> node0, node1;
	unsigned gpu_queries = 0; ///< how often Resolve asked "CUDA" for the GPU
	cmt::GpuCpus nvml;		  ///< empty: no NVML on this fake host, so sysfs answers

	explicit TwoNodeSysfs(const std::vector<unsigned>& allowed)
	{
		char path[] = "/tmp/cmt-sysfs-XXXXXX";
		CHECK(mkdtemp(path) != nullptr);
		root = path;
		node0 = {allowed.front()};
		node1.assign(std::begin(allowed) + 1, std::end(allowed));
		Write("devices/system/node/online", "0-1");
		Write("devices/system/node/has_cpu", "0-1");
		Write("devices/system/node/node0/cpulist", List(node0));
		Write("devices/system/node/node1/cpulist", List(node1));
		Write(std::string("bus/pci/devices/") + kFixtureGpu + "/numa_node", "1");
	}

	~TwoNodeSysfs()
	{
		std::error_code error;
		std::filesystem::remove_all(root, error);
	}

	TwoNodeSysfs(const TwoNodeSysfs&) = delete;
	TwoNodeSysfs& operator=(const TwoNodeSysfs&) = delete;

	static std::string List(const std::vector<unsigned>& cpus)
	{
		std::string text;
		for (std::size_t i = 0; i < std::size(cpus); i++)
		{
			text += (i == 0 ? "" : ",") + std::to_string(cpus[i]);
		}
		return text;
	}

	void Write(const std::string& relative, const std::string& text)
	{
		const auto path = std::filesystem::path(root) / relative;
		std::filesystem::create_directories(path.parent_path());
		CHECK(cmt::WriteTextFile(path.string(), text + "\n"));
	}

	// cmt::ResolveWorkerPlacement on this tree with the CMT_WORKER_* environment, as the hook resolves it.
	cmt::Result<cmt::ThreadPlacement>
	Resolve(cmt::Result<std::optional<std::string>> gpu = std::optional<std::string>{kFixtureGpu})
	{
		auto request = cmt::WorkerPlacementFromEnv();
		if (!request)
		{
			return cmt::Failure(request.error());
		}
		return cmt::ResolveWorkerPlacement(
			*request,
			[&]
			{
				gpu_queries++;
				return gpu;
			},
			root, nvml);
	}
};

// Everything `f` writes to stderr.
template <class F>
std::string Stderr(F&& f)
{
	std::fflush(stderr);
	int fds[2];
	CHECK(pipe(fds) == 0);
	const int saved = dup(STDERR_FILENO);
	CHECK(dup2(fds[1], STDERR_FILENO) >= 0);
	close(fds[1]);
	f();
	CHECK(dup2(saved, STDERR_FILENO) >= 0);
	close(saved);
	std::string text;
	char buffer[4096];
	for (ssize_t n; (n = read(fds[0], buffer, sizeof buffer)) > 0;)
	{
		text.append(buffer, static_cast<std::size_t>(n));
	}
	close(fds[0]);
	return text;
}

} // namespace check
