#include <charconv>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include "common/aliases.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "tests/check.h"
#include "tests/numa_fixture.h"

TEST(exclusive_lock_protects_shared_updates)
{
	Mutex mutex;
	unsigned count = 0;
	std::vector<std::thread> workers;
	for (unsigned i = 0; i < 8; ++i)
	{
		workers.emplace_back(
			[&]
			{
				for (unsigned j = 0; j < 1000; ++j)
				{
					std::lock_guard<Mutex> guard(mutex);
					++count;
				}
			});
	}
	for (auto& worker : workers)
	{
		worker.join();
	}
	CHECK_EQ(count, 8000u);
	std::unique_lock<Mutex> owner(mutex, std::defer_lock);
	CHECK(owner.try_lock());
	bool acquired = true;
	std::thread contender(
		[&]
		{
			std::unique_lock<Mutex> other(mutex, std::try_to_lock);
			acquired = other.owns_lock();
		});
	contender.join();
	CHECK(!acquired);
}

TEST(condition_variable_handoff_and_shutdown)
{
	Mutex mutex;
	ConditionVariable ready, done;
	bool available = false, stop = false;
	unsigned sent = 0, received = 0;
	std::vector<std::thread> workers;
	for (unsigned i = 0; i < 4; ++i)
	{
		workers.emplace_back(
			[&]
			{
				std::unique_lock<Mutex> guard(mutex);
				for (;;)
				{
					while (!available && !stop)
					{
						ready.Wait(guard.mutex());
					}
					if (stop)
					{
						return;
					}
					received += sent;
					available = false;
					done.Signal();
				}
			});
	}
	for (unsigned i = 1; i <= 1000; ++i)
	{
		std::unique_lock<Mutex> guard(mutex);
		sent = i;
		available = true;
		ready.Signal();
		// Recheck the predicate after each wake, including spurious wakes.
		while (available)
		{
			done.Wait(guard.mutex());
		}
	}
	{
		std::lock_guard<Mutex> guard(mutex);
		stop = true;
		ready.SignalAll();
	}
	for (auto& worker : workers)
	{
		worker.join();
	}
	CHECK_EQ(received, 500500u);
}


namespace {

struct WorkerEnvironment
{
	std::optional<std::string> cpus, node, automatic;

	WorkerEnvironment()
	{
		if (const char* value = std::getenv("CMT_WORKER_CPUS"))
		{
			cpus = value;
		}
		if (const char* value = std::getenv("CMT_WORKER_NUMA_NODE"))
		{
			node = value;
		}
		if (const char* value = std::getenv("CMT_WORKER_NUMA"))
		{
			automatic = value;
		}
		CHECK(unsetenv("CMT_WORKER_CPUS") == 0);
		CHECK(unsetenv("CMT_WORKER_NUMA_NODE") == 0);
		CHECK(unsetenv("CMT_WORKER_NUMA") == 0);
	}

	~WorkerEnvironment()
	{
		Restore("CMT_WORKER_CPUS", cpus);
		Restore("CMT_WORKER_NUMA_NODE", node);
		Restore("CMT_WORKER_NUMA", automatic);
	}

	static void Restore(const char* name, const std::optional<std::string>& value)
	{
		CHECK((value ? setenv(name, value->c_str(), 1) : unsetenv(name)) == 0);
	}
};

struct RestoreAffinity
{
	std::vector<unsigned> cpus = check::Require(cmt::GetThreadAffinity());

	~RestoreAffinity()
	{
		if (cpus.empty())
		{
			return;
		}
		const auto count = std::size_t(cpus.back()) + 1;
		const auto bytes = CPU_ALLOC_SIZE(count);
		cpu_set_t* mask = CPU_ALLOC(count);
		if (!mask)
		{
			cmt::Panic("Cannot allocate saved test affinity");
		}
		CPU_ZERO_S(bytes, mask);
		for (unsigned cpu : cpus)
		{
			CPU_SET_S(cpu, bytes, mask);
		}
		// Restoration deliberately widens the temporary test mask back to the
		// saved original. Production placement never widens caller affinity.
		CHECK(pthread_setaffinity_np(pthread_self(), bytes, mask) == 0);
		CPU_FREE(mask);
	}
};

} // namespace

TEST(worker_placement_environment_is_strict)
{
	WorkerEnvironment environment;
	const auto unset = check::Require(cmt::WorkerPlacementFromEnv());
	CHECK(unset.Empty());
	const auto disabled = check::Require(cmt::ThreadPlacementFromEnv());
	CHECK(disabled.cpus.empty() && !disabled.numa_node && disabled.allowed_cpus.empty());
	for (const char* invalid : {"", "-1", "1,", ",1", "1,,2", "2-1", "1-x", "1 2", "4294967296"})
	{
		CHECK(setenv("CMT_WORKER_CPUS", invalid, 1) == 0);
		CHECK_ERROR_KIND(cmt::ThreadPlacementFromEnv(), Usage);
	}
	CHECK(unsetenv("CMT_WORKER_CPUS") == 0);
	for (const char* invalid : {"", "-1", "1x", "2147483648"})
	{
		CHECK(setenv("CMT_WORKER_NUMA_NODE", invalid, 1) == 0);
		CHECK_ERROR_KIND(cmt::ThreadPlacementFromEnv(), Usage);
	}
	CHECK(unsetenv("CMT_WORKER_NUMA_NODE") == 0);
	for (const char* invalid : {"", "yes", "2", "01"})
	{
		CHECK(setenv("CMT_WORKER_NUMA", invalid, 1) == 0);
		CHECK_ERROR_KIND(cmt::WorkerPlacementFromEnv(), Usage);
	}
	CHECK(setenv("CMT_WORKER_NUMA", "1", 1) == 0);
	CHECK(check::Require(cmt::WorkerPlacementFromEnv()).gpu_numa == true);
	CHECK(setenv("CMT_WORKER_NUMA", "0", 1) == 0);
	CHECK(check::Require(cmt::WorkerPlacementFromEnv()).gpu_numa == false);
	// The table and the tuner parse with the same function: a duplicate or an unknown name is refused,
	// and other names are not placement.
	cmt::WorkerPlacement recorded;
	CHECK(check::Require(cmt::ParseWorkerPlacement("CMT_WORKER_NUMA", "1", recorded)));
	CHECK_ERROR_KIND(cmt::ParseWorkerPlacement("CMT_WORKER_NUMA", "1", recorded), Usage);
	CHECK_ERROR_KIND(cmt::ParseWorkerPlacement("CMT_WORKER_FOO", "1", recorded), Usage);
	CHECK(!check::Require(cmt::ParseWorkerPlacement("CUDA_VISIBLE_DEVICES", "0", recorded)));
	const auto parsed = check::Require(cmt::ParseThreadCpus("4096,4096-4096,2"));
	CHECK((parsed == std::vector<unsigned>{2, 4096}));
	CHECK_ERROR_KIND(cmt::ParseThreadCpus("0-4294967295"), Usage);
}

namespace {

using check::Stderr;
using TwoNodeSysfs = check::TwoNodeSysfs;

bool Unplaced(const cmt::Result<cmt::ThreadPlacement>& placement)
{
	return placement && placement->cpus.empty() && !placement->numa_node;
}

} // namespace

TEST(workers_go_to_the_gpu_node_by_default_on_a_two_node_host)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	TwoNodeSysfs sysfs(allowed);
	const auto placed = sysfs.Resolve();
	CHECK(placed && placed->cpus == sysfs.node1 && !placed->numa_node); // CPUs, resolved once, not a node
	CHECK(placed && placed->automatic); // a preference: a thread that cannot use it starts unplaced
	// CMT_WORKER_NUMA=0 turns it off without asking CUDA; explicit CPUs win without asking either.
	sysfs.gpu_queries = 0;
	CHECK(setenv("CMT_WORKER_NUMA", "0", 1) == 0);
	CHECK(Unplaced(sysfs.Resolve()));
	CHECK(setenv("CMT_WORKER_CPUS", std::to_string(allowed.front()).c_str(), 1) == 0);
	const auto explicit_cpus = sysfs.Resolve();
	CHECK(explicit_cpus && explicit_cpus->cpus == std::vector<unsigned>{allowed.front()});
	CHECK_EQ(sysfs.gpu_queries, 0u);
	// A broken CUDA context is an error, not a topology question.
	CHECK(unsetenv("CMT_WORKER_CPUS") == 0);
	CHECK(unsetenv("CMT_WORKER_NUMA") == 0);
	CHECK_ERROR_KIND(sysfs.Resolve(cmt::Failure(cmt::Error::Kind::Cuda, "fake CUDA failure")), Cuda);
}

// NVML first (it can know the GPU's CPUs when sysfs says numa_node -1), sysfs when NVML is missing or fails.
TEST(gpu_cpus_come_from_nvml_then_sysfs)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	TwoNodeSysfs sysfs(allowed);
	std::string asked;
	sysfs.nvml = [&](const std::string& pci) -> cmt::Result<std::vector<unsigned>>
	{
		asked = pci;
		return sysfs.node0; // unlike sysfs's node 1, so the source shows
	};
	auto placed = sysfs.Resolve();
	CHECK(placed && placed->cpus == sysfs.node0 && asked == check::kFixtureGpu);
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/numa_node", "-1"); // a VM's answer
	placed = sysfs.Resolve();
	CHECK(placed && placed->cpus == sysfs.node0);
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/numa_node", "1");
	sysfs.nvml = [](const std::string&) -> cmt::Result<std::vector<unsigned>>
	{ return cmt::Failure(cmt::Error::Kind::Unsupported, "fake NVML failure"); };
	placed = sysfs.Resolve();
	CHECK(placed && placed->cpus == sysfs.node1);
	const auto set = cmt::GpuLocalCpus(check::kFixtureGpu, sysfs.nvml, sysfs.root);
	CHECK(set.cpus == sysfs.node1 && set.node == 1 && set.source == cmt::CpuSource::NodeCpulist &&
		  set.nvml_error == "fake NVML failure");
	CHECK(set.sysfs_error.rfind("local_cpulist: ", 0) == 0); // why the device's own list did not answer
	CHECK_EQ(std::string(cmt::CpuSourceName(set.source)), std::string("sysfs node cpulist"));
	// The device's own local_cpulist comes before its node's list, also when numa_node is -1.
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/local_cpulist",
				TwoNodeSysfs::List(sysfs.node0));
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/numa_node", "-1");
	placed = sysfs.Resolve();
	CHECK(placed && placed->cpus == sysfs.node0);
	CHECK(cmt::GpuLocalCpus(check::kFixtureGpu, sysfs.nvml, sysfs.root).source ==
		  cmt::CpuSource::LocalCpulist);
	// A comma list longer than 4 KiB, as a sparse mask on a large host writes it, is still read.
	std::string sparse;
	while (std::size(sparse) < 5000)
	{
		sparse += TwoNodeSysfs::List(sysfs.node0) + ",";
	}
	sparse += TwoNodeSysfs::List(sysfs.node0);
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/local_cpulist", sparse);
	CHECK(cmt::GpuLocalCpus(check::kFixtureGpu, sysfs.nvml, sysfs.root).cpus == sysfs.node0);
	// None answers: unplaced, never an error.
	std::filesystem::remove(std::filesystem::path(sysfs.root) / "bus/pci/devices" / check::kFixtureGpu /
							"local_cpulist");
	CHECK(Unplaced(sysfs.Resolve()));
	const auto none = cmt::GpuLocalCpus(check::kFixtureGpu, sysfs.nvml, sysfs.root);
	CHECK(none.cpus.empty() && none.source == cmt::CpuSource::None);
	// Both reasons are kept for the record: the missing local_cpulist and the unknown node.
	CHECK(none.sysfs_error.find("local_cpulist: ") != std::string::npos &&
		  none.sysfs_error.find("numa_node") != std::string::npos);
}

TEST(a_single_node_host_is_unchanged_and_asks_cuda_nothing)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	TwoNodeSysfs sysfs(allowed);
	sysfs.Write("devices/system/node/has_cpu", "0");
	CHECK(Unplaced(sysfs.Resolve()));
	std::filesystem::remove(std::filesystem::path(sysfs.root) / "devices/system/node/has_cpu");
	CHECK(Unplaced(sysfs.Resolve())); // no topology at all: one node
	CHECK(setenv("CMT_WORKER_NUMA", "1", 1) == 0); // what older tables recorded: the default, gate included
	CHECK(Unplaced(sysfs.Resolve()));
	CHECK_EQ(sysfs.gpu_queries, 0u);
}

TEST(an_unknown_or_unreachable_gpu_node_leaves_workers_unplaced_and_never_fails)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	TwoNodeSysfs sysfs(allowed);
	for (const char* requested : {"", "1"}) // the default, then an explicit CMT_WORKER_NUMA=1
	{
		if (*requested != '\0')
		{
			CHECK(setenv("CMT_WORKER_NUMA", requested, 1) == 0);
		}
		CHECK(Unplaced(sysfs.Resolve(std::optional<std::string>{}))); // no current GPU
		sysfs.Write("bus/pci/devices/0000:01:00.0/numa_node", "-1");
		CHECK(Unplaced(sysfs.Resolve()));
		sysfs.Write("bus/pci/devices/0000:01:00.0/numa_node", "x");
		CHECK(Unplaced(sysfs.Resolve()));
		sysfs.Write("bus/pci/devices/0000:01:00.0/numa_node", "7"); // a node sysfs does not have
		CHECK(Unplaced(sysfs.Resolve()));
		sysfs.Write("bus/pci/devices/0000:01:00.0/numa_node", "1");
		// The application runs on node 0 only, so no CPU of the GPU's node 1 is allowed: the one line of the
		// process says so, with the remedy, even by default (the line is once per process: the first pass).
		{
			RestoreAffinity restore;
			check::Require(cmt::SetThreadAffinity(pthread_self(), sysfs.node0));
			cmt::Result<cmt::ThreadPlacement> placement;
			const auto said = Stderr([&] { placement = sysfs.Resolve(); });
			CHECK(Unplaced(placement));
			CHECK(*requested != '\0' ||
				  said.find("taskset -c " + cmt::FormatCpuList(sysfs.node1)) != std::string::npos);
		}
		CHECK(check::Require(cmt::GetThreadAffinity()) == allowed);
	}
}

TEST(worker_sees_requested_affinity_before_its_task_runs)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	CHECK(!allowed.empty());
	if (allowed.empty())
	{
		return;
	}
	const unsigned selected = allowed.back();
	cmt::Result<std::vector<unsigned>> observed =
		cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
	int first_cpu = -1;
	cmt::Thread worker;
	check::Require(worker.Start(
		[&]
		{
			first_cpu = sched_getcpu();
			observed = cmt::GetThreadAffinity();
		},
		cmt::ThreadPlacement{{selected}, std::nullopt}));
	check::Require(worker.Join());
	CHECK(observed && *observed == std::vector<unsigned>{selected});
	CHECK(first_cpu >= 0 && static_cast<unsigned>(first_cpu) == selected);
	CHECK(check::Require(cmt::GetThreadAffinity()) == allowed);

	CHECK(setenv("CMT_WORKER_CPUS", std::to_string(selected).c_str(), 1) == 0);
	observed = cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
	check::Require(worker.Start([&] { observed = cmt::GetThreadAffinity(); }));
	check::Require(worker.Join());
	CHECK(observed && *observed == std::vector<unsigned>{selected});
}

TEST(disabled_worker_placement_inherits_and_main_affinity_is_restored)
{
	WorkerEnvironment environment;
	const auto original = check::Require(cmt::GetThreadAffinity());
	CHECK(!original.empty());
	if (original.empty())
	{
		return;
	}
	{
		RestoreAffinity restore;
		check::Require(cmt::SetThreadAffinity(pthread_self(), {original.front()}));
		check::Require(cmt::PinThread(-1));
		cmt::Result<std::vector<unsigned>> observed =
			cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
		cmt::Thread worker;
		check::Require(worker.Start([&] { observed = cmt::GetThreadAffinity(); }));
		check::Require(worker.Join());
		CHECK(observed && *observed == std::vector<unsigned>{original.front()});
		if (original.size() > 1)
		{
			CHECK_ERROR_KIND(cmt::SetThreadAffinity(pthread_self(), original), Usage);
		}
	}
	CHECK(check::Require(cmt::GetThreadAffinity()) == original);
}

TEST(invalid_worker_placement_never_runs_and_releases_the_task)
{
	WorkerEnvironment environment;
	bool ran = false;
	unsigned destroyed = 0;

	struct Probe
	{
		unsigned& count;

		~Probe()
		{
			++count;
		}
	};

	cmt::Thread worker;
	auto started = worker.Start(
		[owned = std::unique_ptr<Probe>(new Probe{destroyed}), &ran]
		{
			(void)owned;
			ran = true;
		},
		cmt::ThreadPlacement{{std::numeric_limits<unsigned>::max()}, std::nullopt});
	CHECK(!started && started.error().IsUsage());
	CHECK(!worker.Joinable() && !ran && destroyed == 1);
	CHECK_ERROR_KIND(cmt::SetThreadAffinity(pthread_self(), {}), Usage);
	CHECK_ERROR_KIND(cmt::NumaNodeCpus(-1), Usage);
}

TEST(numa_placement_intersects_explicit_cpus_and_caller_affinity)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	std::error_code error;
	std::filesystem::directory_iterator nodes("/sys/devices/system/node", error);
	if (error)
	{
		CHECK(!cmt::NumaNodeCpus(0));
		return;
	}
	for (; nodes != std::filesystem::directory_iterator{}; nodes.increment(error))
	{
		if (error)
		{
			CHECK(!error);
			return;
		}
		const std::string name = nodes->path().filename().string();
		if (name.rfind("node", 0) != 0)
		{
			continue;
		}
		int node = -1;
		const auto parsed = std::from_chars(name.data() + 4, name.data() + name.size(), node);
		if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() || node < 0)
		{
			continue;
		}
		auto local = cmt::NumaNodeCpus(node);
		if (!local)
		{
			CHECK(local.error().IsUnsupported() || local.error().IsUsage());
			if (local.error().IsUnsupported())
			{
				bool ran = false;
				cmt::Thread worker;
				CHECK_ERROR_KIND(worker.Start([&] { ran = true; }, cmt::ThreadPlacement{{}, node}),
								 Unsupported);
				CHECK(!ran && !worker.Joinable());
				return;
			}
			continue;
		}
		CHECK(!local->empty());
		if (local->empty())
		{
			continue;
		}
		CHECK(std::includes(allowed.begin(), allowed.end(), local->begin(), local->end()));
		const unsigned selected = local->front();
		cmt::Result<std::vector<unsigned>> observed =
			cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
		cmt::Thread worker;
		check::Require(worker.Start([&] { observed = cmt::GetThreadAffinity(); },
									cmt::ThreadPlacement{{selected}, node}));
		check::Require(worker.Join());
		CHECK(observed && *observed == std::vector<unsigned>{selected});
		const auto other = std::find_if(allowed.begin(), allowed.end(), [&](unsigned cpu)
										{ return !std::binary_search(local->begin(), local->end(), cpu); });
		if (other != allowed.end())
		{
			bool ran = false;
			CHECK_ERROR_KIND(worker.Start([&] { ran = true; }, cmt::ThreadPlacement{{*other}, node}), Usage);
			CHECK(!ran && !worker.Joinable());
		}
		return;
	}
	CHECK(!error);
}

// A thread that cannot start on its GPU-local CPUs (here the cpuset shrank since they were resolved; a sandbox
// that denies sched_setaffinity fails the same start) still starts, on the caller's CPUs; an explicit placement
// still fails.
TEST(an_automatic_placement_that_cannot_start_runs_unplaced)
{
	WorkerEnvironment environment;
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	RestoreAffinity restore;
	check::Require(cmt::SetThreadAffinity(pthread_self(), {allowed.front()}));
	cmt::ThreadPlacement placement{{allowed.back()}, std::nullopt};
	cmt::Thread worker;
	CHECK_ERROR_KIND(worker.Start([] {}, placement), Usage);
	placement.automatic = true;
	cmt::Result<std::vector<unsigned>> observed =
		cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
	CHECK(worker.Start([&] { observed = cmt::GetThreadAffinity(); }, placement));
	CHECK(worker.Join());
	CHECK(observed && *observed == std::vector<unsigned>{allowed.front()});
}

// Under a seccomp filter that denies sched_setaffinity, as a sandbox does, pthread_create with an affinity
// attribute fails with EPERM: an automatic placement still starts and runs (unplaced), an explicit one fails.
// In a forked child, because a seccomp filter cannot be removed.
TEST(an_automatic_placement_starts_where_a_sandbox_denies_sched_setaffinity)
{
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	const pid_t child = fork();
	CHECK(child >= 0);
	if (child == 0)
	{
		sock_filter filter[] = {
			BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
			BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_sched_setaffinity, 0, 1),
			BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
			BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		};
		sock_fprog program = {static_cast<unsigned short>(std::size(filter)), filter};
		if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
			prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0)
		{
			_exit(77);
		}
		cmt::ThreadPlacement placement{{allowed.back()}, std::nullopt};
		cmt::Thread thread;
		const bool explicit_refused = !thread.Start([] {}, placement);
		placement.automatic = true;
		bool ran = false;
		const bool started = static_cast<bool>(thread.Start([&] { ran = true; }, placement));
		const bool joined = started && static_cast<bool>(thread.Join());
		_exit(explicit_refused && joined && ran ? 0 : 1);
	}
	int status = 0;
	CHECK(waitpid(child, &status, 0) == child);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 77)
	{
		std::printf("  seccomp unavailable; skipped\n");
		return;
	}
	CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

TEST(pci_numa_lookup_rejects_paths_and_accepts_normalized_bus_ids)
{
	for (const std::string invalid : {"../numa_node", "0000:00:00.0/../../", "0000:00:20.0", "0000:00:00.8",
									  "0000:GG:00.0", "0:0:0.0junk"})
	{
		CHECK_ERROR_KIND(cmt::PciNumaNode(invalid), Usage);
	}
	const auto narrow = cmt::PciNumaNode("FFFF:FF:1F.7");
	const auto padded = cmt::PciNumaNode("0000ffff:ff:1f.7");
	CHECK(narrow && padded && *narrow == *padded);
}

TEST(detached_worker_attributes_apply_affinity_before_entry)
{
	const auto allowed = check::Require(cmt::GetThreadAffinity());
	CHECK(!allowed.empty());
	if (allowed.empty())
	{
		return;
	}

	struct Observation
	{
		cmt::Result<std::vector<unsigned>> cpus =
			cmt::Failure(cmt::Error::Kind::Verify, "Detached task did not run");
		std::atomic<bool> done{false};
	};

	Observation observed;
	pthread_attr_t attributes;
	const int initialized = pthread_attr_init(&attributes);
	CHECK(initialized == 0);
	if (initialized != 0)
	{
		return;
	}
	const int detached = pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
	const auto placed =
		cmt::ApplyThreadPlacement(attributes, cmt::ThreadPlacement{{allowed.back()}, std::nullopt});
	CHECK(detached == 0 && placed);
	if (detached != 0 || !placed)
	{
		CHECK(pthread_attr_destroy(&attributes) == 0);
		return;
	}
	int detach_state = 0;
	CHECK(pthread_attr_getdetachstate(&attributes, &detach_state) == 0);
	CHECK(detach_state == PTHREAD_CREATE_DETACHED);
	pthread_t handle;
	const int created = pthread_create(
		&handle, &attributes,
		[](void* argument) -> void*
		{
			auto* state = static_cast<Observation*>(argument);
			state->cpus = cmt::GetThreadAffinity();
			// This is the last access to state; the parent may release it after acquire.
			state->done.store(true, std::memory_order_release);
			return nullptr;
		},
		&observed);
	CHECK(pthread_attr_destroy(&attributes) == 0);
	CHECK(created == 0);
	if (created != 0)
	{
		return;
	}
	while (!observed.done.load(std::memory_order_acquire))
	{
		std::this_thread::yield();
	}
	CHECK(observed.cpus && *observed.cpus == std::vector<unsigned>{allowed.back()});
}

TEST(legacy_worker_uses_saved_allowed_mask_without_widening_caller)
{
	WorkerEnvironment environment;
	const auto original = check::Require(cmt::GetThreadAffinity());
	if (original.size() < 2)
	{
		std::puts("note: saved-affinity worker placement needs two allowed CPUs");
		return;
	}
	{
		RestoreAffinity restore;
		const unsigned caller_cpu = original.front(), worker_cpu = original.back();
		check::Require(cmt::SetThreadAffinity(pthread_self(), {caller_cpu}));
		cmt::ThreadPlacement placement{{worker_cpu}, std::nullopt, original};
		cmt::Result<std::vector<unsigned>> observed =
			cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
		cmt::Thread worker;
		check::Require(worker.Start([&] { observed = cmt::GetThreadAffinity(); }, placement));
		check::Require(worker.Join());
		CHECK(observed && *observed == std::vector<unsigned>{worker_cpu});
		CHECK(check::Require(cmt::GetThreadAffinity()) == std::vector<unsigned>{caller_cpu});

		bool ran = false;
		placement.cpus = {std::numeric_limits<unsigned>::max()};
		CHECK_ERROR_KIND(worker.Start([&] { ran = true; }, placement), Usage);
		CHECK(!ran && !worker.Joinable());

		// A saved mask is permission, not an instruction to widen inherited affinity.
		placement.cpus.clear();
		observed = cmt::Failure(cmt::Error::Kind::Verify, "Worker did not run");
		check::Require(worker.Start([&] { observed = cmt::GetThreadAffinity(); }, placement));
		check::Require(worker.Join());
		CHECK(observed && *observed == std::vector<unsigned>{caller_cpu});
		CHECK(check::Require(cmt::GetThreadAffinity()) == std::vector<unsigned>{caller_cpu});
	}
	CHECK(check::Require(cmt::GetThreadAffinity()) == original);
}

TEST_MAIN()
