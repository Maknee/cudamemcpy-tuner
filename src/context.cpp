#include "context.h"

namespace cmt {
namespace ring {

using namespace cmt::cu;

namespace {

class StagingResources
{
public:
	explicit StagingResources(cu::CUcontext context) : context(context)
	{
	}

	Result<void> Setup(const HookConfig& config);
	void Teardown() noexcept;

	bool HasEnvPath() const;
	bool HasTunedPath() const;
	bool Fits(const CopySettings& settings) const;

	cu::CUresult Copy(const CopyRequest& request, const CopySettings* settings = nullptr,
					  CopyObservation* observation = nullptr);

private:
	ChunkPipeline* Pipeline(bool wc_slots) const;
	cu::CUcontext context;
	CopySettings environment;
	std::array<std::unique_ptr<ChunkPipeline>, 2> pipelines;
	Mutex copy_mutex;
};

} // namespace

ChunkPipeline* StagingResources::Pipeline(bool wc_slots) const
{
	return pipelines[wc_slots ? 1 : 0].get();
}

Result<void> StagingResources::Setup(const HookConfig& config)
{
	const bool measured = config.tune_table != nullptr;
	environment.workers = config.pipeline_workers;
	environment.slots = measured ? 0 : config.num_buffers;
	environment.chunk_bytes = config.pipeline_chunk_bytes;
	environment.wc_slots = config.slot_alloc == cmt::StagingAllocation::WriteCombinedPinned;
	std::array<PipelineCapacity, 2> capacities{};
	const auto include = [&](const CopySettings& settings)
	{
		auto& capacity = capacities[settings.wc_slots ? 1 : 0];
		capacity.workers = std::max(capacity.workers, settings.workers);
		capacity.slots = std::max(capacity.slots, settings.slots);
		capacity.pool_bytes = std::max(capacity.pool_bytes, RequiredPinnedBytes(settings));
	};
	if (environment.slots != 0)
	{
		include(environment);
	}
	if (config.tune_table != nullptr)
	{
		for (const auto& row : config.tune_table->rows)
		{
			if (row.mode != cmt::Mode::Chunks)
			{
				continue;
			}
			include(row.settings);
		}
	}
	Result<ThreadPlacement> placement{ThreadPlacement{}};
	if (capacities[0].workers > 1 || capacities[1].workers > 1)
	{
		const auto inputs = HookPlacementInputs(config);
		placement = cu::CurrentWorkerPlacement(g_api, inputs.explicit_cpus, context, inputs.recorded);
	}
	if (!placement)
	{
		return Failure(placement.error());
	}
	for (std::size_t index = 0; index < capacities.size(); ++index)
	{
		const auto& capacity = capacities[index];
		if (capacity.workers == 0)
		{
			continue;
		}
		const ThreadPlacement workers = capacity.workers > 1 ? *placement : ThreadPlacement{};
		HookConfig allocation_config = measured ? HookConfig{} : config;
		if (index == 1)
		{
			allocation_config.slot_alloc = cmt::StagingAllocation::WriteCombinedPinned;
		}
		else if (allocation_config.slot_alloc == cmt::StagingAllocation::WriteCombinedPinned)
		{
			allocation_config.slot_alloc = cmt::StagingAllocation::CachedPinned;
		}
		auto created = ChunkPipeline::Create(context, allocation_config, capacity, workers);
		if (!created)
		{
			Teardown();
			return Failure(created.error());
		}
		auto started = (*created)->Start(workers, !measured && !std::empty(config.pool_cpus));
		if (!started)
		{
			Teardown();
			return Failure(started.error());
		}
		pipelines[index] = std::move(*created);
	}
	if (!HasTunedPath())
	{
		return Failure(Error::Kind::Unsupported, "staging setup produced no pipeline");
	}
	return {};
}

void StagingResources::Teardown() noexcept
{
	for (auto& pipeline : pipelines)
	{
		pipeline.reset();
	}
}

bool StagingResources::HasEnvPath() const
{
	return environment.slots != 0 && Fits(environment);
}

bool StagingResources::HasTunedPath() const
{
	return pipelines[0] != nullptr || pipelines[1] != nullptr;
}

bool StagingResources::Fits(const CopySettings& settings) const
{
	const auto* pipeline = Pipeline(settings.wc_slots);
	return pipeline != nullptr && pipeline->Fits(settings);
}

// Waiting here can deadlock with work queued later on another application stream.
cu::CUresult StagingResources::Copy(const CopyRequest& request, const CopySettings* selected,
									CopyObservation* observation)
{
	if (!copy_mutex.try_lock())
	{
		return kNotStarted;
	}
	std::lock_guard<Mutex> guard(copy_mutex, std::adopt_lock);
	const RelaxedCaptureScope relaxed;
	CopySettings settings = selected != nullptr ? *selected : environment;
	if (selected == nullptr)
	{
		settings.loop =
			CopyLoopFor(request.direction == cmt::Dir::HtoD ? Direction::ToStaging : Direction::FromStaging);
	}
	auto* pipeline = Pipeline(settings.wc_slots);
	if (pipeline == nullptr || !pipeline->Fits(settings))
	{
		return kNotStarted;
	}
	return request.direction == cmt::Dir::HtoD ? pipeline->HtoD(settings, request.device, request.host,
																request.bytes, request.stream, observation)
											   : pipeline->DtoH(settings, request.host, request.device,
																request.bytes, request.stream, observation);
}

namespace {

struct Context
{
	CUcontext ctx = nullptr;
	CUdevice device = -1;
	uint64_t seq = 0;
	unsigned users = 0;

	Context* next = nullptr;
	std::unique_ptr<StagingResources> staging;
	bool staging_failed = false;
	bool building = false;
};

struct Registry
{
	Mutex lock;
	ConditionVariable built;
	ConditionVariable released;
	Context* head = nullptr;
	uint64_t next_seq = 0;
};

Registry& GetRegistry()
{
	static Registry* r = new Registry;
	return *r;
}

std::atomic<bool> g_forked_child{false};
std::atomic<bool> g_copies_disabled{false};

std::atomic<bool> g_used{false};

void ReportStagingFailureOnce(CUcontext context, const Error& error)
{
	static std::atomic<bool> reported{false};
	if (!reported.exchange(true, std::memory_order_relaxed))
	{
		LogLine("error: context {} staging setup failed: {} (code={}); "
				"its pageable copies use the original driver",
				fmt::ptr(context), error.Message(), error.GetCode());
	}
}

class ContextUse
{
public:
	ContextUse() = default;
	ContextUse(const ContextUse&) = delete;
	ContextUse& operator=(const ContextUse&) = delete;

	~ContextUse()
	{
		if (context != nullptr)
		{
			Registry& reg = GetRegistry();
			std::lock_guard<Mutex> guard(reg.lock);
			if (--context->users == 0)
			{
				reg.released.SignalAll();
			}
		}
	}

	void Hold(Context* c)
	{
		++c->users;
		context = c;
	}

private:
	Context* context = nullptr;
};

bool WantsStaging(const HookConfig& cfg)
{
	return cfg.num_buffers >= 1 || (cfg.tune_table && cfg.tune_table->HasStagedCopies());
}

Context* FindLocked(const Registry& reg, CUcontext ctx)
{
	for (Context* c = reg.head; c != nullptr; c = c->next)
	{
		if (c->ctx == ctx)
		{
			return c;
		}
	}
	return nullptr;
}

CUdevice CurrentDevice()
{
	CUdevice dev = -1;
	if (g_api.cuCtxGetDevice == nullptr || g_api.cuCtxGetDevice(&dev) != kSuccess)
	{
		return -1;
	}
	return dev;
}

} // namespace

void NoteUsed()
{
	if (!g_used.load(std::memory_order_relaxed))
	{
		g_used.store(true);
	}
}

bool ForkedChild()
{
	if (!g_forked_child.load(std::memory_order_relaxed))
	{
		return false;
	}
	static std::atomic<bool> said{false};
	if (!said.exchange(true, std::memory_order_relaxed))
	{
		LogLine("note: pid {} is a fork() child of a process that had used the hook; its CUDA "
				"copies go to the driver unchanged.",
				static_cast<int>(getpid()));
	}
	return true;
}

namespace {

StagingResources* StagingForCurrentContext(ContextUse& hold)
{
	CUcontext ctx = nullptr;
	const CUresult rc = g_api.cuCtxGetCurrent != nullptr ? g_api.cuCtxGetCurrent(&ctx) : -1;
	if (rc != kSuccess || ctx == nullptr)
	{
		LogDebug("pool_get: no current context (rc={})", rc);
		return nullptr;
	}
	const HookConfig& cfg = GetConfig();
	Registry& reg = GetRegistry();
	std::unique_ptr<Context> spare;
	std::unique_lock<Mutex> lk(reg.lock);
	Context* c = nullptr;
	for (;;)
	{
		c = FindLocked(reg, ctx);
		if (c == nullptr)
		{
			if (spare == nullptr)
			{
				lk.unlock();
				spare = std::make_unique<Context>();
				spare->ctx = ctx;
				spare->device = CurrentDevice();
				lk.lock();
				continue;
			}
			c = spare.release();
			c->seq = reg.next_seq++;
			c->next = reg.head;
			reg.head = c;
		}
		const bool need_staging = c->staging == nullptr &&
								  !c->staging_failed && WantsStaging(cfg);
		if (!need_staging)
		{
			break;
		}
		if (!c->building)
		{
			break;
		}
		reg.built.Wait(
			lk.mutex());
	}
	hold.Hold(c);
	const bool need_staging =
		c->staging == nullptr && !c->staging_failed && WantsStaging(cfg);
	if (need_staging && !c->building)
	{
		c->building = true;
		lk.unlock();
		std::unique_ptr<StagingResources> staging;
		bool staging_tried = false;
		Result<void> setup_result;
		const RelaxedCaptureScope relaxed;

		if (need_staging)
		{
			staging_tried = true;
			staging = std::make_unique<StagingResources>(ctx);
			setup_result = staging->Setup(cfg);
			if (!setup_result)
			{
				staging.reset();
			}
		}
		lk.lock();

		if (staging != nullptr)
		{
			c->staging = std::move(staging);
		}
		else if (staging_tried)
		{
			c->staging_failed = true;
		}
		c->building = false;
		reg.built.SignalAll();
		if (!setup_result)
		{
			lk.unlock();
			ReportStagingFailureOnce(ctx, setup_result.error());
			lk.lock();
		}
	}
	return c->staging.get();
}

} // namespace

bool CopiesDisabled()
{
	return g_copies_disabled.load(std::memory_order_relaxed);
}

void DisableCopies()
{
	g_copies_disabled.store(true, std::memory_order_relaxed);
}

bool UsedInProcess()
{
	return g_used.load();
}

namespace {

void DropContext(CUcontext ctx) noexcept
{
	std::unique_ptr<Context> victim;
	{
		Registry& reg = GetRegistry();
		std::unique_lock<Mutex> guard(reg.lock);
		for (Context** link = &reg.head; *link != nullptr; link = &(*link)->next)
		{
			if ((*link)->ctx == ctx)
			{
				victim.reset(*link);
				*link = victim->next;
				break;
			}
		}
		while (victim != nullptr && victim->users != 0)
		{
			reg.released.Wait(guard.mutex());
		}
	}
	victim.reset();
	pinning::ReleaseContext(ctx);
}

} // namespace

void ResetContext(CUcontext ctx)
{
	DropContext(ctx);
}

void ResetDevice(CUdevice dev)
{
	Registry& reg = GetRegistry();
	uint64_t limit = 0;
	{
		std::lock_guard<Mutex> guard(reg.lock);
		limit = reg.next_seq;
	}
	for (;;)
	{
		CUcontext ctx = nullptr;
		{
			std::lock_guard<Mutex> guard(reg.lock);
			for (Context* c = reg.head; c != nullptr; c = c->next)
			{
				if (c->seq < limit && (c->device == dev || c->device < 0))
				{
					ctx = c->ctx;
					break;
				}
			}
		}
		if (ctx == nullptr)
		{
			pinning::ReleaseAllRegistrations();
			return;
		}
		DropContext(ctx);
	}
}

void AfterForkInChild()
{
	g_forked_child.store(true, std::memory_order_relaxed);
}

} // namespace ring
} // namespace cmt

namespace cmt {
namespace tune_audit {
namespace {
struct Event
{
	Outcome outcome = Outcome::Unknown;
	ring::CopyObservation job;
	pinning::Observation pinning;
	uint64_t row_min = 0;
	uint64_t row_max = 0;
};

struct State
{
	bool enabled = false;
	bool in_call = false;
	Snapshot snapshot;
	Event event;
};

std::atomic<bool> ever_enabled{false};
thread_local State state;

bool SameJob(const ring::CopyObservation& a, const ring::CopyObservation& b)
{
	return a.completed == b.completed && a.requested_workers == b.requested_workers &&
		   a.actual_workers == b.actual_workers && a.slots == b.slots &&
		   a.max_slots_used == b.max_slots_used && a.slot_bytes == b.slot_bytes && a.variant == b.variant &&
		   a.write_back == b.write_back && a.flush_bytes == b.flush_bytes && a.bench_dma == b.bench_dma &&
		   a.pace_gbps == b.pace_gbps && a.pace_quantum == b.pace_quantum && a.wc_slots == b.wc_slots;
}

bool SamePinning(const pinning::Observation& a, const pinning::Observation& b)
{
	return a.completed == b.completed && a.registered_bytes == b.registered_bytes &&
		   a.released_bytes == b.released_bytes && a.registered_dma_bytes == b.registered_dma_bytes &&
		   a.pageable_dma_bytes == b.pageable_dma_bytes && a.borrowed_bytes == b.borrowed_bytes &&
		   a.chunks == b.chunks && a.first_chunk_bytes == b.first_chunk_bytes &&
		   a.max_chunk_bytes == b.max_chunk_bytes && a.max_registered_bytes == b.max_registered_bytes &&
		   a.read_only == b.read_only;
}
} // namespace

Call::Call(cmt::Dir dir, std::size_t requested_bytes, bool async)
	: direction(dir), bytes(requested_bytes), asynchronous(async)
{
	if (!ever_enabled.load(std::memory_order_relaxed) || !state.enabled)
	{
		return;
	}
	if (state.in_call)
	{
		state.event.outcome = Outcome::Nested;
		return;
	}
	state.in_call = true;
	state.event = {};
	owns_audit = true;
}

cu::CUresult Call::Finish(cu::CUresult rc) noexcept
{
	if (!owns_audit)
	{
		return rc;
	}
	auto& s = state.snapshot;
	auto& e = state.event;
	if (e.job.completed && e.outcome == Outcome::Unknown)
	{
		e.outcome = Outcome::Chunks;
	}
	if (s.calls == 0)
	{
		s.direction = static_cast<uint64_t>(direction);
		s.copy_bytes = bytes;
		s.row_min = e.row_min;
		s.row_max = e.row_max;
		s.job = e.job;
		s.pinning = e.pinning;
	}
	else if (s.direction != static_cast<uint64_t>(direction) || s.copy_bytes != bytes ||
			 s.row_min != e.row_min || s.row_max != e.row_max || !SameJob(s.job, e.job) ||
			 !SamePinning(s.pinning, e.pinning))
	{
		++s.mixed;
	}
	++s.calls;
	if (UINT64_MAX - s.bytes < bytes)
	{
		++s.mixed;
	}
	s.bytes += bytes;
	++(asynchronous ? s.async_calls : s.sync_calls);
	s.errors += rc != cu::kSuccess;
	++s.outcomes[static_cast<unsigned>(e.outcome)];
	state.in_call = false;
	owns_audit = false;
	return rc;
}

void Reason(Outcome outcome) noexcept
{
	if (ever_enabled.load(std::memory_order_relaxed) && state.in_call &&
		state.event.outcome != Outcome::Nested)
	{
		state.event.outcome = outcome;
	}
}

void Row(const cmt::Row& row) noexcept
{
	if (ever_enabled.load(std::memory_order_relaxed) && state.in_call)
	{
		state.event.row_min = row.min_bytes;
		state.event.row_max = row.max_bytes;
	}
}

void Otherwise(Outcome outcome) noexcept
{
	if (ever_enabled.load(std::memory_order_relaxed) && state.in_call &&
		state.event.outcome == Outcome::Unknown)
	{
		state.event.outcome = outcome;
	}
}

ring::CopyObservation* Observation() noexcept
{
	return ever_enabled.load(std::memory_order_relaxed) && state.in_call ? &state.event.job : nullptr;
}

pinning::Observation* PinningObservation() noexcept
{
	return ever_enabled.load(std::memory_order_relaxed) && state.in_call ? &state.event.pinning : nullptr;
}
} // namespace tune_audit
} // namespace cmt

extern "C" int cmt_tune_audit_begin(uint32_t version, uint32_t size) noexcept
{
	if (version != cmt::tune_audit::kVersion || size != sizeof(cmt::tune_audit::Snapshot))
	{
		return 0;
	}
	auto& s = cmt::tune_audit::state;
	if (s.enabled)
	{
		return 0;
	}
	s.snapshot = {};
	s.event = {};
	s.in_call = false;
	s.enabled = true;
	cmt::tune_audit::ever_enabled.store(true, std::memory_order_relaxed);
	return 1;
}

extern "C" int cmt_tune_audit_end(cmt::tune_audit::Snapshot* out, uint32_t size) noexcept
{
	if (!cmt::tune_audit::ever_enabled.load(std::memory_order_relaxed))
	{
		return 0;
	}
	auto& s = cmt::tune_audit::state;
	if (!out || size != sizeof(*out) || !s.enabled || s.in_call)
	{
		return 0;
	}
	*out = s.snapshot;
	s.enabled = false;
	return 1;
}

namespace cmt {
namespace ring {
namespace {
void ReportSettingsMismatchOnce(const CopySettings& settings)
{
	static std::atomic<bool> said{false};
	if (said.exchange(true, std::memory_order_relaxed))
	{
		return;
	}
	LogLine("error: selected workers={} slots={} chunk_bytes={} wc={} do not fit their staging pool; "
			"such copies use the original driver. Reported once.",
			settings.workers, settings.slots, settings.chunk_bytes, settings.wc_slots);
}

} // namespace

cu::CUresult CopyMeasured(const CopyRequest& request, const cmt::Row& row)
{
	ContextUse lifetime;
	StagingResources* staging = StagingForCurrentContext(lifetime);
	if (staging == nullptr || !staging->HasTunedPath())
	{
		tune_audit::Reason(tune_audit::Outcome::NoPipeline);
		LogDebug("tune table: no chunks pipeline in this context; {} {} bytes uses the original driver",
				 cmt::DirName(request.direction), request.bytes);
		return kNotStarted;
	}
	if (!staging->Fits(row.settings))
	{
		ReportSettingsMismatchOnce(row.settings);
		tune_audit::Reason(tune_audit::Outcome::KnobsMismatch);
		return kNotStarted;
	}
	if (row.loop_fell_back)
	{
		tune_audit::Reason(tune_audit::Outcome::LoopFallback);
	}
	const CUresult result = staging->Copy(request, &row.settings, tune_audit::Observation());
	if (result == kNotStarted)
	{
		tune_audit::Reason(tune_audit::Outcome::NotStarted);
	}
	return result;
}

cu::CUresult CopyEnvironment(const CopyRequest& request)
{
	ContextUse lifetime;
	StagingResources* staging = StagingForCurrentContext(lifetime);
	if (staging == nullptr || !staging->HasEnvPath())
	{
		return kNotStarted;
	}
	return staging->Copy(request, nullptr, tune_audit::Observation());
}
} // namespace ring
} // namespace cmt

namespace cmt {
namespace ring {
namespace {
bool FirstMachineNote()
{
	static std::atomic<bool> said{false};
	return !said.load(std::memory_order_relaxed) && !said.exchange(true, std::memory_order_relaxed);
}
} // namespace

bool MatchesTuningMachine()
{
	const auto& config = GetConfig();
	const auto& table = config.tune_table;
	if (!table)
	{
		return true;
	}
	struct Cpu
	{
		bool matches = false;
		char model[192] = {};
	};
	static const Cpu cpu = []
	{
		const auto& source = *GetConfig().tune_table;
		auto model = CpuModelName();
#if defined(__x86_64__)
		constexpr std::string_view architecture = "x86_64";
#elif defined(__aarch64__)
		constexpr std::string_view architecture = "aarch64";
#else
		constexpr std::string_view architecture = "unknown";
#endif
		Cpu result;
		result.matches = model && *model == source.source_cpu_model && architecture == source.source_cpu_arch;
		const std::string_view shown = model ? std::string_view(*model) : std::string_view("an unknown CPU");
		const std::size_t bytes = std::min(std::size(shown), sizeof(result.model) - 1);
		std::memcpy(result.model, std::data(shown), bytes);
		return result;
	}();
	if (!g_api.cuCtxGetDevice || !g_api.cuDeviceGetName)
	{
		return false;
	}
	CUdevice device = -1;
	if (g_api.cuCtxGetDevice(&device) != kSuccess)
	{
		return false;
	}
	static thread_local CUdevice checked_device = -1;
	static thread_local bool gpu_matches = false;
	static thread_local char gpu_name[256] = {};
	if (device != checked_device)
	{
		char name[sizeof(gpu_name)] = {};
		if (g_api.cuDeviceGetName(name, sizeof(name) - 1, device) != kSuccess)
		{
			return false;
		}
		gpu_matches = table->source_gpu_name == name;
		std::memcpy(gpu_name, name, sizeof(gpu_name));
		checked_device = device;
	}
	if (cpu.matches && gpu_matches)
	{
		return true;
	}
	if (config.remote_config)
	{
		if (FirstMachineNote())
		{
			LogDebug("using the remote tune of {} / {} on this {} / {}", table->source_gpu_name,
					 table->source_cpu_model, gpu_name, cpu.model);
		}
		return true;
	}
	if (FirstMachineNote())
	{
		LogLine("tune table {} was measured on {} / {}, not on this {} / {}: copies use the stock driver. "
				"Tune here with cmt tune --search quick",
				config.tune_table_path, table->source_gpu_name, table->source_cpu_model, gpu_name, cpu.model);
	}
	return false;
}

} // namespace ring
} // namespace cmt
