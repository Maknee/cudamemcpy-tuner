// Canonical measured tuning results through the preload hook and fake CUDA.
// The parent writes fixtures before exec: the hook reads configuration only once.
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "config.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "copying/chunk_pipeline.h"
#if defined(CMT_TEST_SHARE)
#include "share.h"
#endif

extern char** environ;

namespace {
constexpr std::size_t kChunk = 64u << 10;
// Two one-sample processes: the fewest timed samples selection accepts.
constexpr unsigned kRounds = 2;
using Dptr = unsigned long long;
using CopyTo = int (*)(Dptr, const void*, std::size_t);
using CopyFrom = int (*)(void*, Dptr, std::size_t);
using CopyToAsync = int (*)(Dptr, const void*, std::size_t, void*);
using CopyFromAsync = int (*)(void*, Dptr, std::size_t, void*);

template <class Function>
Function Load(void* library, const char* name)
{
	void* address = dlsym(library, name);
	if (!address)
	{
		std::fprintf(stderr, "missing symbol %s: %s\n", name, dlerror());
		std::exit(2);
	}
	Function function;
	std::memcpy(&function, &address, sizeof(function));
	return function;
}

std::set<std::string> Workers()
{
	std::set<std::string> workers;
	DIR* directory = opendir("/proc/self/task");
	CHECK(directory != nullptr);
	if (!directory)
	{
		return workers;
	}
	while (dirent* entry = readdir(directory))
	{
		if (entry->d_name[0] == '.')
		{
			continue;
		}
		auto name = cmt::ReadTextFile(std::string("/proc/self/task/") + entry->d_name + "/comm");
		if (name && name->rfind("cmt-pipe-", 0) == 0)
		{
			workers.insert(entry->d_name);
		}
	}
	closedir(directory);
	return workers;
}

cmt::Result<std::string> Fixture(const std::string& scenario)
{
	using namespace cmt;
	auto model = CpuModelName();
	if (!model)
	{
		return Failure(model.error());
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.tuning", true);
	JsonField(writer, "version", 1);
	JsonField(writer, "state", "complete");
	JsonField(writer, "selection_policy", kSelectionPolicyVersion);
	JsonKey(writer, "request");
	writer.start_object();
	JsonField(writer, "process_rounds", kRounds, true);
	JsonField(writer, "max_workers", 2);
	JsonField(writer, "max_pinned_mib", 1);
	JsonKey(writer, "environment");
	writer.start_array();
	if (scenario == "recorded_cpus" || scenario == "invalid_recorded_cpus" ||
		scenario == "duplicate_placement")
	{
		auto cpus = GetThreadAffinity();
		if (!cpus || cpus->empty())
		{
			return UsageError("fixture requires an allowed CPU");
		}
		const auto cpu = scenario == "invalid_recorded_cpus" ? 999999u : cpus->front();
		const auto entry = "CMT_WORKER_CPUS=" + std::to_string(cpu);
		writer.escape_and_append_with_quotes(entry);
		if (scenario == "recorded_cpus")
		{
			writer.append_comma();
			writer.escape_and_append_with_quotes("CMT_WORKER_NUMA=1");
		}
		if (scenario == "duplicate_placement")
		{
			writer.append_raw(",");
			writer.escape_and_append_with_quotes(entry);
		}
	}
	writer.end_array();
	writer.end_object();
	JsonKey(writer, "source_machine");
	writer.start_object();
	JsonKey(writer, "host", true);
	writer.start_object();
	JsonKey(writer, "cpu", true);
	writer.start_object();
	JsonField(writer, "model", scenario == "cpu_mismatch" ? "a different CPU" : *model, true);
	JsonField(writer, "arch", ArchName(NativeArch()));
	writer.end_object();
	writer.end_object();
	JsonKey(writer, "gpus");
	writer.start_array();
	writer.start_object();
	JsonField(writer, "name",
			  scenario == "gpu_mismatch" || scenario == "gpu_mismatch_remote" ? "a different GPU" : "CMT fake GPU",
			  true);
	writer.end_object();
	writer.end_array();
	// A share (`cmt export`) keys the tune by the driver too.
	JsonKey(writer, "driver");
	writer.start_object();
	JsonKey(writer, "kernel_module", true);
	writer.start_object();
	JsonField(writer, "version", "615.71.09", true);
	writer.end_object();
	writer.end_object();
	writer.end_object();
	JsonKey(writer, "runs");
	writer.start_array();
	bool first = true;
	// Synthetic records exercise policy admission; these are not performance measurements.
	for (unsigned round = 1; round <= kRounds; ++round)
	{
		for (Dir direction : {Dir::HtoD, Dir::DtoH})
		{
			for (std::uint64_t bytes : {128ull << 10, 512ull << 10, 1ull << 20})
			{
				for (Mode mode : {Mode::Driver, Mode::Chunks})
				{
					CopyRun run;
					run.process_round = round;
					run.direction = direction;
					run.bytes = bytes;
					run.mode = mode;
					run.status = "complete";
					run.verified = true;
					run.execution = mode == Mode::Driver ? "bare_driver" : "hooked_copy";
					run.settings = {direction == Dir::HtoD ? 1 : 2, 2, kChunk, "memcpy", false};
					run.settings.wc_slots = direction == Dir::DtoH;
					const bool driver_wins = direction == Dir::HtoD && bytes == (512u << 10);
					run.gbps = (mode == Mode::Driver) == driver_wins ? 20 : 10;
					run.seconds = static_cast<double>(bytes) / (run.gbps * 1e9);
					JsonWriter record;
					WriteCopyRun(record, run);
					auto text = FinishJson(record);
					if (!text)
					{
						return Failure(text.error());
					}
					while (!text->empty() && text->back() != '}')
					{
						text->pop_back();
					}
					text->pop_back();
					if (mode == Mode::Chunks)
					{
						*text +=
							fmt::format(",\"tune_audit_valid\":1,\"tune_audit_requested_workers\":{},\"tune_"
										"audit_slots\":2,\"tune_audit_slot_bytes\":{},\"tune_audit_wc_"
										"slots\":{},\"tune_audit_bench_dma\":0,\"tune_audit_pace_gbps\":0,"
										"\"tune_audit_write_back\":{},\"tune_audit_flush_bytes\":{},"
										"\"tune_audit_measured_rejected\":0,\"tune_audit_measured_errors\":0,"
										"\"tune_audit_measured_mixed\":0,\"tune_audit_measured_calls\":1,"
										"\"tune_audit_measured_accepted\":1,\"tune_audit_measured_bytes\":{}",
										run.settings.workers, kChunk, run.settings.wc_slots ? 1 : 0,
										static_cast<unsigned>(WriteBack::None), CopyCtx{}.flush_batch, bytes);
					}
					*text +=
						fmt::format(",\"sample_count\":1,\"median_sample_index\":0,\"samples\":[{{"
									"\"seconds\":{},\"copies_per_sample\":1,\"elapsed_seconds\":{}}}]}}",
									run.seconds, scenario == "bad_sample" ? run.seconds * 2 : run.seconds);
					if (!first)
					{
						writer.append_comma();
					}
					first = false;
					writer.append_raw(*text);
				}
			}
		}
	}
	writer.end_array();
	writer.end_object();
	return FinishJson(writer);
}

// The "generic" scenario: cuMemcpy / cuMemcpyAsync (cudaMemcpyDefault) under the measured table above, whose
// rows stage HtoD and DtoH from 128 KiB except HtoD [512 KiB, 1 MiB), a Driver row; below 128 KiB there is no
// row. Every case runs synchronously and asynchronously and checks the bytes; a staged case must reach the
// chunks pipeline (one audited call, one completed job, the fake's generic entry not called), a passed case
// must reach the fake's generic function unchanged (no audited call). Each memory kind is one classification
// rule of hook.cpp (CopyGeneric); the fake reports the attributes the RTX 3090 driver does
// (fake_libcuda.cpp).
enum class Memory : std::uint8_t
{
	Pageable,
	Pinned,
	TailRegistered, // pageable whose last page is registered
	Device,			// fake kinds from here (fake_libcuda.cpp, DeviceKind)
	Vmm,
	Managed,
	Peer,
	ShortDevice, // a pool allocation: the range is half of its mapping
	Unmapped,	 // a reservation with nothing mapped
	Split,		 // four adjacent mappings
	Many,		 // 4 KiB mappings, more than the hook walks
	MiddleHole,
	TailHole,
	Mixed, // second half mapped from another device
};

struct Region
{
	Memory memory;
	std::size_t bytes;
	uint8_t* data = nullptr;
	void* registered = nullptr;
};

constexpr std::size_t kPage = 4096;
constexpr std::size_t kStagedBytes = 1u << 20;	  // a Chunks row in both directions
constexpr std::size_t kMixedBytes = 600u << 10;	  // HtoD Driver row, DtoH Chunks row
constexpr std::size_t kUnstagedBytes = 64u << 10; // no row in either direction

struct GenericDriver
{
	int (*copy)(Dptr, Dptr, std::size_t);
	int (*copy_async)(Dptr, Dptr, std::size_t, void*);
	int (*device_alloc)(Dptr*, std::size_t, int);
	int (*device_free)(Dptr);
	int (*host_alloc)(void**, std::size_t, unsigned);
	int (*host_free)(void*);
	int (*host_register)(void*, std::size_t, unsigned);
	int (*host_unregister)(void*);
	unsigned (*queries)();
	long (*passed)();
	void (*fail_queries)(unsigned long long);
	void (*set_capture)(void*, int);
	long (*captured_generic)();
	int (*sync)();
	long (*dma_count)();
	void (*reset_log)();
	cmt::tune_audit::Begin begin;
	cmt::tune_audit::End end;
	CopyTo htod;
	CopyFrom dtoh;
	CopyToAsync htod_async;
	CopyFromAsync dtoh_async;
};

Dptr Address(const void* p)
{
	return static_cast<Dptr>(reinterpret_cast<uintptr_t>(p));
}

Region Allocate(const GenericDriver& d, Memory memory, std::size_t bytes)
{
	Region r{memory, bytes};
	void* p = nullptr;
	switch (memory)
	{
	case Memory::Pageable:
		p = std::aligned_alloc(kPage, (bytes + kPage - 1) / kPage * kPage);
		break;
	case Memory::Pinned:
		CHECK(d.host_alloc(&p, bytes, 0) == 0);
		break;
	case Memory::TailRegistered:
		p = std::aligned_alloc(kPage, bytes);
		CHECK(bytes % kPage == 0);
		r.registered = static_cast<uint8_t*>(p) + bytes - kPage;
		CHECK(d.host_register(r.registered, kPage, 0) == 0);
		break;
	default:
	{
		Dptr device = 0;
		const int kind = static_cast<int>(memory) - static_cast<int>(Memory::Device);
		CHECK(d.device_alloc(&device, bytes, kind) == 0);
		p = reinterpret_cast<void*>(static_cast<uintptr_t>(device));
	}
	}
	CHECK(p != nullptr);
	r.data = static_cast<uint8_t*>(p);
	return r;
}

void Release(const GenericDriver& d, const Region& r)
{
	switch (r.memory)
	{
	case Memory::Pageable:
		std::free(r.data);
		break;
	case Memory::Pinned:
		CHECK(d.host_free(r.data) == 0);
		break;
	case Memory::TailRegistered:
		CHECK(d.host_unregister(r.registered) == 0);
		std::free(r.data);
		break;
	default:
		CHECK(d.device_free(Address(r.data)) == 0);
	}
}

bool IsHostPageable(Memory memory)
{
	return memory == Memory::Pageable || memory == Memory::TailRegistered;
}

enum class Expect : std::uint8_t
{
	Staged,	  // the chunks pipeline ran
	Declined, // the hook took it, its route chose the driver (a Driver row): the generic function ran again
	Passed,	  // the original generic function ran
	Refused	  // the original generic function ran and refused it (rc 1) without writing a byte
};

const char* MemoryName(Memory memory)
{
	constexpr const char* names[] = {"pageable", "pinned",		"tail_registered", "device",   "vmm",
									 "managed",	 "peer",		"short_device",	   "unmapped", "split",
									 "many",	 "middle_hole", "tail_hole",	   "mixed"};
	static_assert(std::size(names) == static_cast<std::size_t>(Memory::Mixed) + 1);
	return names[static_cast<unsigned>(memory)];
}

// Which attribute queries the fake fails during the copy: none, every one, or the source pointer's only.
enum class Fail : std::uint8_t
{
	None,
	All,
	Source
};

void Generic(const GenericDriver& d, void* stream, Memory to, Memory from, std::size_t bytes, Expect expect,
			 unsigned queries, Fail fail = Fail::None, std::size_t device_bytes = 0,
			 std::size_t device_offset = 0)
{
	// The device end (either kind from Device on) may be allocated larger and copied at an offset into it.
	const auto extent = [&](Memory memory)
	{ return memory >= Memory::Device && device_bytes != 0 ? device_bytes : bytes; };
	const auto offset = [&](Memory memory) { return memory >= Memory::Device ? device_offset : 0; };
	for (const bool async : {false, true})
	{
		const Region dst_region = Allocate(d, to, extent(to));
		const Region src_region = Allocate(d, from, extent(from));

		struct
		{
			uint8_t* data;
		} dst{dst_region.data + offset(to)}, src{src_region.data + offset(from)};
		const unsigned seed = static_cast<unsigned>(bytes) + (async ? 7u : 3u);
		for (std::size_t i = 0; i < bytes; i++)
		{
			src.data[i] = check::pattern(i, seed);
		}
		std::memset(dst.data, 0xCD, bytes);
		d.reset_log();
		const unsigned queries_before = d.queries();
		const long passed_before = d.passed();
		CHECK(d.begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1);
		d.fail_queries(fail == Fail::All ? ~0ull : fail == Fail::Source ? Address(src.data) : 0);
		const int rc = async ? d.copy_async(Address(dst.data), Address(src.data), bytes, stream)
							 : d.copy(Address(dst.data), Address(src.data), bytes);
		d.fail_queries(0);
		bool done_on_return = true;
		if (async && IsHostPageable(to))
		{
			// A DtoH into pageable memory is complete on return.
			for (std::size_t i = 0; i < bytes && done_on_return; i++)
			{
				done_on_return = dst.data[i] == check::pattern(i, seed);
			}
		}
		if (async && IsHostPageable(from))
		{
			std::memset(src.data, 0x11, bytes); // a pageable source is consumed on return
		}
		cmt::tune_audit::Snapshot observed;
		CHECK(d.end(&observed, sizeof(observed)) == 1);
		CHECK(d.sync() == 0);
		const bool refused = expect == Expect::Refused;
		bool correct = true;
		for (std::size_t i = 0; i < bytes && correct; i++)
		{
			correct = dst.data[i] == (refused ? 0xCD : check::pattern(i, seed));
		}
		done_on_return = done_on_return || refused;
		const char* mode = async ? "async" : "sync";
		CHECKF(rc == (refused ? 1 : 0) && correct && done_on_return,
			   "generic %s %s <- %s %zu: rc %d, bytes %s%s", mode, MemoryName(to), MemoryName(from), bytes,
			   rc, correct ? "ok" : "wrong", done_on_return ? "" : ", not done on return");
		const long generic_calls = d.passed() - passed_before;
		const bool staged =
			observed.calls == 1 && generic_calls == 0 && observed.job.completed == 1 && d.dma_count() > 0;
		const bool declined = observed.calls == 1 && generic_calls == 1 && observed.job.completed == 0;
		const bool passed = observed.calls == 0 && generic_calls == 1;
		CHECKF(expect == Expect::Staged		? staged
			   : expect == Expect::Declined ? declined
											: passed, // Passed and Refused
			   "generic %s %s <- %s %zu: expected %s; audited calls %llu, jobs %llu, DMAs %ld, generic calls "
			   "%ld",
			   mode, MemoryName(to), MemoryName(from), bytes,
			   expect == Expect::Staged		? "staged"
			   : expect == Expect::Declined ? "the hook, then the generic driver function"
											: "the generic driver function",
			   static_cast<unsigned long long>(observed.calls),
			   static_cast<unsigned long long>(observed.job.completed), d.dma_count(),
			   d.passed() - passed_before);
		CHECKF(d.queries() - queries_before == queries,
			   "generic %s %s <- %s %zu: %u attribute queries, %u expected", mode, MemoryName(to),
			   MemoryName(from), bytes, d.queries() - queries_before, queries);
		Release(d, src_region);
		Release(d, dst_region);
	}
}

bool FromHook(void* p)
{
	Dl_info info;
	if (p == nullptr || dladdr(p, &info) == 0 || info.dli_fname == nullptr)
	{
		return false;
	}
	return std::strstr(info.dli_fname, "libcmt_hook") != nullptr;
}

// A typed copy (cuMemcpyHtoD/DtoH_v2, sync and async) between pageable memory and device memory of `kind`, or
// from `host_kind` used as the host end. Staged: the pipeline ran and the bytes landed. Refused: the driver
// refused it (rc 1), no DMA was issued and not a byte was written; a staged chunk into a hole kills the
// process on the real driver.
void Typed(const GenericDriver& d, void* stream, Memory kind, Memory host_kind, bool to_device, bool staged)
{
	const std::size_t bytes = kStagedBytes;
	for (const bool async : {false, true})
	{
		const Region device = Allocate(d, kind, bytes);
		const Region host = Allocate(d, host_kind, bytes);
		const Region& from = to_device ? host : device;
		const Region& to = to_device ? device : host;
		const unsigned seed = async ? 41u : 43u;
		for (std::size_t i = 0; i < bytes; i++)
		{
			from.data[i] = check::pattern(i, seed);
		}
		std::memset(to.data, 0xCD, bytes);
		d.reset_log();
		CHECK(d.begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1);
		const int rc = to_device ? (async ? d.htod_async(Address(device.data), host.data, bytes, stream)
										  : d.htod(Address(device.data), host.data, bytes))
								 : (async ? d.dtoh_async(host.data, Address(device.data), bytes, stream)
										  : d.dtoh(host.data, Address(device.data), bytes));
		cmt::tune_audit::Snapshot observed;
		CHECK(d.end(&observed, sizeof(observed)) == 1);
		CHECK(d.sync() == 0);
		bool as_expected = true;
		for (std::size_t i = 0; i < bytes && as_expected; i++)
		{
			as_expected = to.data[i] == (staged ? check::pattern(i, seed) : 0xCD);
		}
		CHECKF(rc == (staged ? 0 : 1) && as_expected && (observed.job.completed == 1) == staged &&
				   (d.dma_count() > 0) == staged,
			   "typed %s %s %s <- %s: rc %d, bytes %s, jobs %llu, DMAs %ld (expected %s)",
			   async ? "async" : "sync", to_device ? "htod" : "dtoh", MemoryName(to.memory),
			   MemoryName(from.memory), rc, as_expected ? "as expected" : "WRONG",
			   static_cast<unsigned long long>(observed.job.completed), d.dma_count(),
			   staged ? "staged" : "refused by the driver");
		Release(d, host);
		Release(d, device);
	}
}

// An async generic copy keeps the application's stream: on a non-blocking stream S, behind an op queued on S
// that the fake runs only when S is waited for (FAKE_STRICT_LEGACY=1: the legacy stream does not wait for S).
//   DtoH  a pinned HtoD queued on S writes D; the generic DtoH from D on S must see those bytes, complete on
//         return (pageable destination).
//   HtoD  a pinned DtoH queued on S reads D; the generic HtoD to D on S must not overwrite D before it.
// Both are staged. Run on the legacy stream instead (stream dropped, or the async entry run as the sync one),
// the staged DMAs are not ordered after S's op and the bytes come out wrong.
void StreamOrder(const GenericDriver& d, void* nonblocking)
{
	const std::size_t bytes = kStagedBytes;
	const Region device = Allocate(d, Memory::Device, bytes);
	const Region host = Allocate(d, Memory::Pageable, bytes);
	void* pinned = nullptr;
	CHECK(d.host_alloc(&pinned, bytes, 0) == 0);
	auto* p = static_cast<uint8_t*>(pinned);
	for (std::size_t i = 0; i < bytes; i++)
	{
		p[i] = check::pattern(i, 61);
	}
	std::memset(device.data, 0xCD, bytes);
	CHECK(d.htod_async(Address(device.data), pinned, bytes, nonblocking) == 0); // queued, not run
	CHECK(d.begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1);
	CHECK(d.copy_async(Address(host.data), Address(device.data), bytes, nonblocking) == 0);
	cmt::tune_audit::Snapshot observed;
	CHECK(d.end(&observed, sizeof(observed)) == 1);
	bool ok = observed.job.completed == 1;
	for (std::size_t i = 0; i < bytes && ok; i++)
	{
		ok = host.data[i] == check::pattern(i, 61);
	}
	CHECKF(ok, "a staged generic DtoH on a non-blocking stream did not follow the op queued before it");
	CHECK(d.sync() == 0);

	std::memset(p, 0, bytes);
	for (std::size_t i = 0; i < bytes; i++)
	{
		device.data[i] = check::pattern(i, 62);
		host.data[i] = check::pattern(i, 63);
	}
	CHECK(d.dtoh_async(pinned, Address(device.data), bytes, nonblocking) == 0); // queued, not run
	CHECK(d.begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1);
	CHECK(d.copy_async(Address(device.data), Address(host.data), bytes, nonblocking) == 0);
	CHECK(d.end(&observed, sizeof(observed)) == 1);
	CHECK(d.sync() == 0);
	ok = observed.job.completed == 1;
	for (std::size_t i = 0; i < bytes && ok; i++)
	{
		ok = p[i] == check::pattern(i, 62) && device.data[i] == check::pattern(i, 63);
	}
	CHECKF(ok, "a staged generic HtoD on a non-blocking stream overtook the op queued before it");
	CHECK(d.host_free(pinned) == 0);
	Release(d, host);
	Release(d, device);
}

int GenericChild(void* library)
{
	GenericDriver d{};
	d.copy = Load<decltype(d.copy)>(library, "cuMemcpy");
	d.copy_async = Load<decltype(d.copy_async)>(library, "cuMemcpyAsync");
	d.device_alloc = Load<decltype(d.device_alloc)>(library, "fake_device_alloc");
	d.device_free = Load<decltype(d.device_free)>(library, "cuMemFree_v2");
	d.host_alloc = Load<decltype(d.host_alloc)>(library, "cuMemHostAlloc");
	d.host_free = Load<decltype(d.host_free)>(library, "cuMemFreeHost");
	d.host_register = Load<decltype(d.host_register)>(library, "cuMemHostRegister_v2");
	d.host_unregister = Load<decltype(d.host_unregister)>(library, "cuMemHostUnregister");
	d.queries = Load<decltype(d.queries)>(library, "fake_trace_queries");
	d.passed = Load<decltype(d.passed)>(library, "fake_generic_calls");
	d.fail_queries = Load<decltype(d.fail_queries)>(library, "fake_fail_pointer_queries");
	d.set_capture = Load<decltype(d.set_capture)>(library, "fake_set_capture");
	d.captured_generic = Load<decltype(d.captured_generic)>(library, "fake_captured_generic");
	d.sync = Load<decltype(d.sync)>(library, "cuCtxSynchronize");
	d.dma_count = Load<decltype(d.dma_count)>(library, "fake_dma_log_count");
	d.reset_log = Load<decltype(d.reset_log)>(library, "fake_dma_log_reset");
	d.begin = Load<cmt::tune_audit::Begin>(RTLD_DEFAULT, "cmt_tune_audit_begin");
	d.end = Load<cmt::tune_audit::End>(RTLD_DEFAULT, "cmt_tune_audit_end");
	d.htod = Load<CopyTo>(library, "cuMemcpyHtoD_v2");
	d.dtoh = Load<CopyFrom>(library, "cuMemcpyDtoH_v2");
	d.htod_async = Load<CopyToAsync>(library, "cuMemcpyHtoDAsync_v2");
	d.dtoh_async = Load<CopyFromAsync>(library, "cuMemcpyDtoHAsync_v2");
	const auto create_stream = Load<int (*)(void**, unsigned)>(library, "cuStreamCreate");
	const auto get_proc =
		Load<int (*)(const char*, void**, int, unsigned long long)>(library, "cuGetProcAddress");

	// Lookups: the legacy-stream exports are redirected, through dlsym and through cuGetProcAddress; a
	// per-thread-stream query and the _ptds / _ptsz exports keep the driver's function.
	CHECK(FromHook(dlsym(library, "cuMemcpy")) && FromHook(dlsym(library, "cuMemcpyAsync")));
	CHECK(!FromHook(dlsym(library, "cuMemcpy_ptds")) && !FromHook(dlsym(library, "cuMemcpyAsync_ptsz")));
	for (const char* name : {"cuMemcpy", "cuMemcpyAsync"})
	{
		void* fn = nullptr;
		CHECKF(get_proc(name, &fn, 12000, 0) == 0 && FromHook(fn), "%s at 12000 is not redirected", name);
		CHECKF(get_proc(name, &fn, 4000, 0) == 0 && FromHook(fn), "%s at 4000 is not redirected", name);
		CHECKF(get_proc(name, &fn, 12000, 2) == 0 && fn != nullptr && !FromHook(fn),
			   "%s per-thread-stream is redirected", name);
	}

	void* stream = nullptr;
	CHECK(create_stream(&stream, 0) == 0);
	// Staged: device memory of this device, inside its allocation, and pageable memory at both host ends.
	// Three queries: destination, source, the host end's last byte; none more in the typed path.
	Generic(d, stream, Memory::Device, Memory::Pageable, kStagedBytes, Expect::Staged, 3);
	Generic(d, stream, Memory::Pageable, Memory::Device, kStagedBytes, Expect::Staged, 3);
	// VMM memory has CONTEXT NULL; the ordinal says it is this device's.
	Generic(d, stream, Memory::Vmm, Memory::Pageable, kStagedBytes, Expect::Staged, 3);
	Generic(d, stream, Memory::Pageable, Memory::Vmm, kStagedBytes, Expect::Staged, 3);
	// One direction's row is staged: the gate opens, and the HtoD route's Driver row sends the copy back to
	// the generic function the application called (not the typed one).
	Generic(d, stream, Memory::Device, Memory::Pageable, kMixedBytes, Expect::Declined, 3);
	Generic(d, stream, Memory::Pageable, Memory::Device, kMixedBytes, Expect::Staged, 3);
	// Passed: managed memory on either side (type 2, as device memory is).
	Generic(d, stream, Memory::Managed, Memory::Pageable, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Pageable, Memory::Managed, kStagedBytes, Expect::Passed, 2);
	// Pinned host memory, and pageable memory whose last page is registered.
	Generic(d, stream, Memory::Device, Memory::Pinned, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Pinned, Memory::Device, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Device, Memory::TailRegistered, kStagedBytes, Expect::Passed, 3);
	Generic(d, stream, Memory::TailRegistered, Memory::Device, kStagedBytes, Expect::Passed, 3);
	// Device to device, host to host, another GPU.
	Generic(d, stream, Memory::Device, Memory::Device, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Pageable, Memory::Pageable, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Peer, Memory::Pageable, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Pageable, Memory::Peer, kStagedBytes, Expect::Passed, 2);
	// [p, p + n) leaves its allocation, but not its mapping (a pool chunk): only the range check stops it.
	Generic(d, stream, Memory::ShortDevice, Memory::Pageable, kStagedBytes, Expect::Passed, 2);
	Generic(d, stream, Memory::Pageable, Memory::ShortDevice, kStagedBytes, Expect::Passed, 2);
	// The range boundary, on a pool-like allocation (range = the first half of its mapping): a copy ending
	// exactly at the range's end is staged, one ending a byte past it goes to the driver.
	Generic(d, stream, Memory::ShortDevice, Memory::Pageable, kStagedBytes, Expect::Staged, 3, Fail::None,
			2 * kStagedBytes, 0);
	Generic(d, stream, Memory::ShortDevice, Memory::Pageable, kStagedBytes, Expect::Passed, 2, Fail::None,
			2 * kStagedBytes, 1);
	Generic(d, stream, Memory::Pageable, Memory::ShortDevice, kStagedBytes, Expect::Staged, 3, Fail::None,
			2 * kStagedBytes, 0);
	Generic(d, stream, Memory::Pageable, Memory::ShortDevice, kStagedBytes, Expect::Passed, 2, Fail::None,
			2 * kStagedBytes, 1);
	// Mappings. Adjacent ones are walked (one query each) and staged; a hole in the middle or at the tail, or
	// an unmapped reservation, goes to the driver, which refuses it without writing (a staged chunk into a
	// hole kills the process on the real driver); so does a hole read as a host pointer (it is type 0, as
	// pageable memory is, but inside a range). Past 64 following mappings, or into another device's mapping:
	// the driver.
	Generic(d, stream, Memory::Split, Memory::Pageable, kStagedBytes, Expect::Staged, 6);
	Generic(d, stream, Memory::Pageable, Memory::Split, kStagedBytes, Expect::Staged, 6);
	Generic(d, stream, Memory::MiddleHole, Memory::Pageable, kStagedBytes, Expect::Refused, 3);
	Generic(d, stream, Memory::Pageable, Memory::MiddleHole, kStagedBytes, Expect::Refused, 3);
	Generic(d, stream, Memory::TailHole, Memory::Pageable, kStagedBytes, Expect::Refused, 3);
	Generic(d, stream, Memory::Pageable, Memory::TailHole, kStagedBytes, Expect::Refused, 3);
	Generic(d, stream, Memory::Unmapped, Memory::Pageable, kStagedBytes, Expect::Refused, 2);
	Generic(d, stream, Memory::Device, Memory::Unmapped, kStagedBytes, Expect::Refused, 2);
	Generic(d, stream, Memory::Many, Memory::Pageable, kStagedBytes, Expect::Passed, 66);
	Generic(d, stream, Memory::Mixed, Memory::Pageable, kStagedBytes, Expect::Passed, 3);
	// Below every staged row: no query at all.
	Generic(d, stream, Memory::Device, Memory::Pageable, kUnstagedBytes, Expect::Passed, 0);
	Generic(d, stream, Memory::Pageable, Memory::Device, kUnstagedBytes, Expect::Passed, 0);

	// A failed query is not a classification: the copy goes to the driver, also when the other end is device
	// memory and the failed end would read as pageable from the query's untouched outputs.
	Generic(d, stream, Memory::Device, Memory::Pageable, kStagedBytes, Expect::Passed, 1, Fail::All);
	Generic(d, stream, Memory::Device, Memory::Pageable, kStagedBytes, Expect::Passed, 2, Fail::Source);

	void* nonblocking = nullptr;
	CHECK(create_stream(&nonblocking, 1) == 0);
	StreamOrder(d, nonblocking);

	// Typed copies get the same device-range check (TypedEnds): a hole, at either end, goes to the
	// driver.
	Typed(d, stream, Memory::Device, Memory::Pageable, true, true);
	Typed(d, stream, Memory::Device, Memory::Pageable, false, true);
	Typed(d, stream, Memory::MiddleHole, Memory::Pageable, true, false);
	Typed(d, stream, Memory::TailHole, Memory::Pageable, false, false);
	Typed(d, stream, Memory::Device, Memory::Unmapped, true, false);

	// nullptr reads as pageable to the driver's query: it must be refused by value, and the driver reports
	// it.
	{
		const Region device = Allocate(d, Memory::Device, kStagedBytes);
		const unsigned queries_before = d.queries();
		const long passed_before = d.passed();
		CHECK(d.copy(Address(device.data), 0, kStagedBytes) == 1);
		CHECK(d.copy_async(0, Address(device.data), kStagedBytes, stream) == 1);
		CHECK(d.queries() == queries_before && d.passed() - passed_before == 2);
		// Zero bytes: straight to the driver.
		CHECK(d.copy(Address(device.data), Address(device.data), 0) == 0 && d.queries() == queries_before);
		Release(d, device);
	}

	// Stream capture keeps the typed path's rule: a copy on a capturing stream goes to the driver, which
	// captures it, and nothing is staged. It must be captured as the generic call the application made: the
	// typed one records a different graph node (memory types 1/2 instead of 4/4, on the 615 driver).
	for (const bool to_device : {true, false})
	{
		const Region device = Allocate(d, Memory::Device, kStagedBytes);
		const Region host = Allocate(d, Memory::Pageable, kStagedBytes);
		d.reset_log();
		const long generic_before = d.captured_generic();
		d.set_capture(stream, 0);
		const int rc = to_device
						   ? d.copy_async(Address(device.data), Address(host.data), kStagedBytes, stream)
						   : d.copy_async(Address(host.data), Address(device.data), kStagedBytes, stream);
		d.set_capture(nullptr, 0);
		CHECKF(rc == 0 && d.dma_count() == 0 && d.captured_generic() - generic_before == 1,
			   "captured generic %s: rc %d, %ld DMAs, %ld generic copies captured (0, 0, 1 expected)",
			   to_device ? "htod" : "dtoh", rc, d.dma_count(), d.captured_generic() - generic_before);
		Release(d, host);
		Release(d, device);
	}
	return check::exit_code();
}

int Child(const std::string& scenario)
{
	void* library = dlopen("libcuda.so.1", RTLD_NOW);
	if (!library)
	{
		std::fprintf(stderr, "%s\n", dlerror());
		return 2;
	}
	if (scenario == "generic")
	{
		return GenericChild(library);
	}
	const auto htod = Load<CopyTo>(library, "cuMemcpyHtoD_v2");
	const auto dtoh = Load<CopyFrom>(library, "cuMemcpyDtoH_v2");
	const auto htod_async = Load<CopyToAsync>(library, "cuMemcpyHtoDAsync_v2");
	const auto dtoh_async = Load<CopyFromAsync>(library, "cuMemcpyDtoHAsync_v2");
	const auto sync = Load<int (*)()>(library, "cuCtxSynchronize");
	const auto create_stream = Load<int (*)(void**, unsigned)>(library, "cuStreamCreate");
	const auto destroy_context = Load<int (*)(void*)>(library, "cuCtxDestroy_v2");
	const auto allocations = Load<long (*)()>(library, "fake_host_alloc_attempts");
	const auto allocated_bytes = Load<long (*)()>(library, "fake_host_alloc_bytes");
	const auto free_bytes = Load<long (*)()>(library, "fake_host_free_bytes");
	const auto flags = Load<long (*)(unsigned)>(library, "fake_host_alloc_flag_attempts");
	const auto reset_log = Load<void (*)()>(library, "fake_dma_log_reset");
	const auto dma_count = Load<long (*)()>(library, "fake_dma_log_count");
	const auto begin = Load<cmt::tune_audit::Begin>(RTLD_DEFAULT, "cmt_tune_audit_begin");
	const auto end = Load<cmt::tune_audit::End>(RTLD_DEFAULT, "cmt_tune_audit_end");
	void* stream = nullptr;
	CHECK(create_stream(&stream, 0) == 0);
	const bool enabled = (scenario == "rows" || scenario == "contaminated" || scenario == "recorded_cpus" ||
						  scenario == "environment_cpus" || scenario == "share" || scenario == "gpu_mismatch_remote");
	const auto copy = [&](cmt::Dir direction, std::size_t bytes, int mode)
	{
		check::Buffer host(bytes), device(bytes);
		for (std::size_t i = 0; i < bytes; ++i)
		{
			(direction == cmt::Dir::HtoD ? host : device)[i] = check::pattern(i, 37);
		}
		const Dptr pointer = reinterpret_cast<uintptr_t>(device.data());
		void* caller_stream = mode == 1 ? stream : nullptr;
		reset_log();
		CHECK(begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1);
		int result;
		if (direction == cmt::Dir::HtoD)
		{
			result = mode == 0 ? htod(pointer, host.data(), bytes)
							   : htod_async(pointer, host.data(), bytes, caller_stream);
			std::memset(host.data(), 0xCC, bytes);
		}
		else
		{
			result = mode == 0 ? dtoh(host.data(), pointer, bytes)
							   : dtoh_async(host.data(), pointer, bytes, caller_stream);
		}
		CHECK(result == 0);
		cmt::tune_audit::Snapshot observed;
		CHECK(end(&observed, sizeof(observed)) == 1);
		if (direction == cmt::Dir::HtoD)
		{
			CHECK(sync() == 0);
		}
		auto& output = direction == cmt::Dir::HtoD ? device : host;
		bool correct = true;
		for (std::size_t i = 0; i < bytes; ++i)
		{
			correct = correct && output.data()[i] == check::pattern(i, 37);
		}
		CHECK(correct);
		CHECK(sync() == 0);
		// Above the largest measured size (1 MiB) the 1 MiB row still applies.
		const bool staged = enabled && bytes >= (128u << 10) &&
							!(direction == cmt::Dir::HtoD && bytes >= (512u << 10) && bytes < (1u << 20));
		if (staged)
		{
			CHECK(observed.job.completed == 1);
			CHECK(observed.job.bench_dma == 0 && observed.job.pace_gbps == 0);
			CHECK(observed.job.wc_slots == (direction == cmt::Dir::DtoH ? 1u : 0u));
			CHECK(observed.job.requested_workers == (direction == cmt::Dir::HtoD ? 1u : 2u));
			CHECK(observed.job.slots == 2 && observed.job.slot_bytes == kChunk);
			CHECK(dma_count() == static_cast<long>((bytes + kChunk - 1) / kChunk));
		}
		else
		{
			CHECK(observed.job.completed == 0);
			CHECK(dma_count() == 0);
		}
	};
	for (int pass = 0; pass < 2; ++pass)
	{
		const auto before = allocations();
		const auto workers = Workers();
		for (auto direction : {cmt::Dir::HtoD, cmt::Dir::DtoH})
		{
			for (std::size_t bytes : {1u, (128u << 10) - 1, 128u << 10, (512u << 10) - 1, 512u << 10,
									  (1u << 20) - 1, 1u << 20, (1u << 20) + 1})
			{
				for (int mode = 0; mode < 3; ++mode)
				{
					copy(direction, bytes, mode);
				}
			}
		}
		if (pass == 1)
		{
			CHECK(allocations() == before);
			CHECK(Workers() == workers);
		}
	}
	if (enabled)
	{
		CHECK(flags(0) > 0);
		CHECK(flags(4) > 0);
		// One caller-only HtoD lane and two DtoH lanes need one helper in total.
		CHECK(Workers().size() == 1);
		CHECK(allocated_bytes() == 6 * static_cast<long>(kChunk));
		if (scenario == "recorded_cpus")
		{
			auto allowed = cmt::GetThreadAffinity();
			CHECK(allowed && !allowed->empty());
			for (const auto& tid : Workers())
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
				CHECK(sched_getaffinity(static_cast<pid_t>(std::stoi(tid)), sizeof(mask), &mask) == 0);
				CHECK(CPU_COUNT(&mask) == 1);
				CHECK(CPU_ISSET(allowed->front(), &mask));
			}
		}
	}
	else
	{
		if (scenario != "invalid_recorded_cpus")
		{
			CHECK(allocations() == 0);
		}
		CHECK(Workers().empty());
	}
	const auto allocated = allocated_bytes();
	CHECK(destroy_context(reinterpret_cast<void*>(0x1000)) == 0);
	CHECK(free_bytes() == allocated);
	return check::exit_code();
}
} // namespace

int main(int argc, char** argv)
{
	if (argc == 3 && std::string(argv[1]) == "--child")
	{
		return Child(argv[2]);
	}
	if (argc != 3)
	{
		std::fprintf(stderr, "usage: test_tune_table_e2e HOOK FAKE_LIBDIR\n");
		return 2;
	}
	auto executable = cmt::SelfExe();
	CHECK(executable.has_value());
	if (!executable)
	{
		return 2;
	}
	char temporary[] = "/tmp/cmt-policy-test-XXXXXX";
	if (!mkdtemp(temporary))
	{
		return 2;
	}
	// How often the hook says, on stderr, that it ignores a table from another machine (context.cpp,
	// MatchesTuningMachine): once per process, whatever its number of copies; never under CMT_REMOTE_CONFIG=1
	// (`cmt run --remote_config`), which uses the table and says so only with MEMCPY_HOOK_DEBUG=1.
	const std::map<std::string, std::size_t> ignored_lines{
		{"rows", 0}, {"cpu_mismatch", 1}, {"gpu_mismatch", 1}, {"gpu_mismatch_remote", 0}, {"share", 0}};
	std::vector<std::string> scenarios{"rows",			   "generic",		  "contaminated",		"recorded_cpus",
									   "environment_cpus", "invalid_recorded_cpus", "duplicate_placement",
									   "cpu_mismatch",	   "gpu_mismatch",	  "gpu_mismatch_remote", "bad_sample",
									   "malformed"};
#if defined(CMT_TEST_SHARE)
	scenarios.push_back("share");
#endif
	for (const auto& name : scenarios)
	{
		const char* scenario = name.c_str();
		auto text = Fixture(scenario);
		CHECK(text.has_value());
		if (!text)
		{
			return 2;
		}
#if defined(CMT_TEST_SHARE)
		if (name == "share")
		{
			// A share holds nothing but the selection line the hook reads.
			auto line = text ? cmt::Reselect(*text) : text;
			CHECK(line && cmt::SelectionLineBytes(*line).has_value());
			text = line ? cmt::share::Export(*line) : line;
			CHECK(text.has_value());
			if (!text)
			{
				return 2;
			}
		}
#endif
		const std::string path = std::string(temporary) + "/" + scenario + ".json";
		CHECK(cmt::WriteTextFile(path, std::string(scenario) == "malformed" ? "{" : *text).has_value());
		if (std::string(scenario) == "rows" || std::string(scenario) == "recorded_cpus")
		{
			// `cmt run --numa on|off` (CMT_OVERRIDE numa=) replaces what the table recorded: "rows" recorded
			// nothing (the hook's default applies), "recorded_cpus" CMT_WORKER_NUMA=1.
			const auto recorded = cmt::tuning::Load(path.c_str());
			CHECK(recorded &&
				  (*recorded)->worker_placement.gpu_numa ==
					  (std::string(scenario) == "rows" ? std::nullopt : std::optional<bool>{true}));
			for (const bool on : {false, true})
			{
				const auto overrides = cmt::ParseRunOverrides(on ? "numa=on" : "numa=off");
				CHECK(overrides && cmt::FormatRunOverrides(*overrides) == (on ? "numa=on" : "numa=off"));
				const auto table = overrides ? cmt::tuning::Load(path.c_str(), *overrides) : recorded;
				// on is the default (nothing recorded), off is recorded off.
				CHECK(table && (*table)->worker_placement.gpu_numa ==
								   (on ? std::nullopt : std::optional<bool>{false}));
			}
		}
		const std::string log_path = std::string(temporary) + "/" + scenario + ".stderr";
		const pid_t child = fork();
		if (child == 0)
		{
			std::vector<std::string> remove;
			for (char** entry = environ; entry && *entry; ++entry)
			{
				std::string name(*entry);
				name.resize(name.find('='));
				if (name.rfind("CMT_", 0) == 0 || name.rfind("STAGING_", 0) == 0 ||
					name.rfind("MEMCPY_HOOK_", 0) == 0)
				{
					remove.push_back(name);
				}
			}
			for (const auto& name : remove)
			{
				unsetenv(name.c_str());
			}
			setenv("LD_PRELOAD", argv[1], 1);
			setenv("LD_LIBRARY_PATH", argv[2], 1);
			setenv("CMT_CONFIG", path.c_str(), 1);
			setenv("CMT_TUNE_AUDIT", "1", 1);
			setenv("MEMCPY_HOOK_NO_EAGER", "1", 1);
			if (std::string(scenario) == "generic")
			{
				setenv("FAKE_STRICT_LEGACY", "1", 1); // the legacy stream does not wait for non-blocking ones
				// The rejected-query note goes to stderr once (four rejected queries in the child).
				setenv("MEMCPY_HOOK_DEBUG", "1", 1);
			}
			if (name == "gpu_mismatch_remote")
			{
				setenv("CMT_REMOTE_CONFIG", "1", 1);
			}
			if (std::string(scenario) == "generic" || ignored_lines.count(name))
			{
				const int log = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
				if (log < 0 || dup2(log, STDERR_FILENO) < 0)
				{
					_exit(126);
				}
				close(log);
			}
			if (std::string(scenario) == "contaminated" || std::string(scenario) == "recorded_cpus")
			{
				setenv("STAGING_BENCH_DMA", "1", 1);
				setenv("STAGING_THREADS_COPY_NUM_BUFFERS", "32", 1);
				setenv("STAGING_BUFFER_SIZE", "16384", 1);
				setenv("STAGING_SLOT_ALLOC", "mmap_4k", 1);
				setenv("STAGING_COPY_ISA", "sse2", 1);
				setenv("STAGING_COPY_FLUSH_BYTES", "64", 1);
			}
			// This CPU does not exist, so it would turn staging off; a table ignores it, whether it recorded a
			// placement (recorded_cpus) or none (environment_cpus: the default applies, as it did measured).
			if (std::string(scenario) == "recorded_cpus" || std::string(scenario) == "environment_cpus")
			{
				setenv("CMT_WORKER_CPUS", "999999", 1);
			}
			execl(executable->c_str(), executable->c_str(), "--child", scenario, nullptr);
			_exit(127);
		}
		CHECK(child > 0);
		if (child > 0)
		{
			int status = 0;
			CHECK(waitpid(child, &status, 0) == child);
			CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
			if (std::string(scenario) == "generic")
			{
				auto log = cmt::ReadTextFile(log_path);
				CHECK(log.has_value());
				std::size_t notes = 0;
				for (std::size_t at = 0;
					 log && (at = log->find("cuPointerGetAttributes rejected", at)) != std::string::npos;
					 at++)
				{
					notes++;
				}
				CHECKF(notes == 1, "the rejected-query note appeared %zu times, once expected", notes);
				if (WEXITSTATUS(status) != 0 && log)
				{
					std::fputs(log->c_str(), stderr); // the child's own report went to the log
				}
			}
			if (ignored_lines.count(name))
			{
				const auto log = cmt::ReadTextFile(log_path);
				CHECK(log.has_value());
				std::size_t got = 0;
				for (std::size_t at = 0; log && (at = log->find("copies use the stock driver", at)) != std::string::npos;
					 at++)
				{
					got++;
				}
				const std::size_t want = ignored_lines.at(name);
				CHECKF(got == want, "%s: %zu ignored-table lines, %zu expected", scenario, got, want);
				if ((WEXITSTATUS(status) != 0 || got != want) && log)
				{
					std::fputs(log->c_str(), stderr);
				}
			}
		}
	}
	std::error_code error;
	std::filesystem::remove_all(temporary, error);
	CHECK(!error);
	return check::exit_code();
}
