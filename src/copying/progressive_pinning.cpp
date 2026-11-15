#include "cuda/kernel_status.h"
#include "copying/progressive_pinning.h"

#if defined(CMT_HOOK_PRELOAD_ONLY)
#include "hook.h"

extern "C" void __libc_free(void*);
extern "C" void* __libc_realloc(void*, size_t);

namespace cmt {
namespace {
using FreeFn = void (*)(void*);
using ReallocFn = void* (*)(void*, size_t);
std::atomic<FreeFn> g_next_free{nullptr};
std::atomic<ReallocFn> g_next_realloc{nullptr};
thread_local bool t_resolving_allocator = false;

bool ResolveNextAllocator()
{
	if (t_resolving_allocator)
	{
		return false;
	}
	t_resolving_allocator = true;
	const auto lookup = GetLoaderFunctions().lookup;
	void* f = lookup != nullptr ? lookup(RTLD_NEXT, "free") : nullptr;
	void* r = lookup != nullptr ? lookup(RTLD_NEXT, "realloc") : nullptr;
	g_next_free.store(f != nullptr ? reinterpret_cast<FreeFn>(f) : &__libc_free, std::memory_order_release);
	g_next_realloc.store(r != nullptr ? reinterpret_cast<ReallocFn>(r) : &__libc_realloc,
						 std::memory_order_release);
	t_resolving_allocator = false;
	return true;
}

// CUDA callbacks cannot call CUDA; mark driver-created threads to avoid reentry.
using PthreadCreateFn = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
std::atomic<PthreadCreateFn> g_real_pthread_create{nullptr};

PthreadCreateFn OriginalPthreadCreate()
{
	PthreadCreateFn fn = g_real_pthread_create.load(std::memory_order_acquire);
	if (fn != nullptr)
	{
		return fn;
	}
	void* p = dlvsym(RTLD_NEXT, "pthread_create", kRequiredGlibcSymbolVersion);
	fn = reinterpret_cast<PthreadCreateFn>(p);
	g_real_pthread_create.store(fn, std::memory_order_release);
	return fn;
}

struct DriverThreadStart
{
	void* (*fn)(void*);
	void* arg;
};

void* DriverThreadStartEntry(void* p)
{
	const DriverThreadStart start = *static_cast<DriverThreadStart*>(p);
	std::free(p);
	pinning::MarkDriverThread();
	return start.fn(start.arg);
}

} // namespace

void FreeToAllocator(void* p)
{
	FreeFn next = g_next_free.load(std::memory_order_acquire);
	if (next == nullptr)
	{
		if (!ResolveNextAllocator())
		{
			return;
		}
		next = g_next_free.load(std::memory_order_acquire);
	}
	next(p);
}

namespace pinning {
void PrepareInterposition()
{
	ResolveNextAllocator();
	OriginalPthreadCreate();
}
} // namespace pinning

namespace copying {
namespace services {
void ReleaseAllocation(void* pointer)
{
	FreeToAllocator(pointer);
}

} // namespace services
} // namespace copying
} // namespace cmt

#ifndef MADV_DONTNEED_LOCKED
#define MADV_DONTNEED_LOCKED 24
#endif

extern "C" __attribute__((visibility("default"))) void free(void* p) noexcept
{
	if (p == nullptr)
	{
		return;
	}
	if (cmt::pinning::Active() && cmt::pinning::OnFree(p, malloc_usable_size(p)))
	{
		return;
	}
	cmt::FreeToAllocator(p);
}

extern "C" __attribute__((visibility("default"))) void* realloc(void* p, size_t n) noexcept
{
	if (p != nullptr && cmt::pinning::Active())
	{
		const size_t usable = malloc_usable_size(p);
		if (cmt::pinning::Holds(p, usable))
		{
			if (n == 0)
			{
				if (!cmt::pinning::OnFree(p, usable))
				{
					cmt::FreeToAllocator(p);
				}
				return nullptr;
			}
			void* q = malloc(n);
			if (q == nullptr)
			{
				errno = ENOMEM;
				return nullptr;
			}
			std::memcpy(q, p, usable < n ? usable : n);
			if (!cmt::pinning::OnFree(p, usable))
			{
				cmt::FreeToAllocator(p);
			}
			return q;
		}
	}
	cmt::ReallocFn next = cmt::g_next_realloc.load(std::memory_order_acquire);
	if (next == nullptr)
	{
		if (!cmt::ResolveNextAllocator())
		{
			errno = ENOMEM;
			return nullptr;
		}
		next = cmt::g_next_realloc.load(std::memory_order_acquire);
	}
	return next(p, n);
}

extern "C" __attribute__((visibility("default"))) int munmap(void* addr, size_t len) noexcept
{
	if (cmt::pinning::Active())
	{
		cmt::pinning::OnUnmap(addr, len);
	}
	return static_cast<int>(syscall(SYS_munmap, addr, len));
}

extern "C" __attribute__((visibility("default"))) void* mremap(void* old, size_t old_size, size_t new_size,
															   int flags, ...) noexcept
{
	void* new_addr = nullptr;
	if ((flags & MREMAP_FIXED) != 0)
	{
		va_list ap;
		va_start(ap, flags);
		new_addr = va_arg(ap, void*);
		va_end(ap);
	}
	if (cmt::pinning::Active())
	{
		cmt::pinning::OnUnmap(old, old_size);
		if (new_addr != nullptr)
		{
			cmt::pinning::OnUnmap(new_addr, new_size);
		}
	}
	return reinterpret_cast<void*>(syscall(SYS_mremap, old, old_size, new_size, flags, new_addr));
}

extern "C" __attribute__((visibility("default"))) int madvise(void* addr, size_t len, int advice) noexcept
{
	if ((advice == MADV_DONTNEED || advice == MADV_DONTNEED_LOCKED || advice == MADV_FREE ||
		 advice == MADV_REMOVE) &&
		cmt::pinning::Active())
	{
		cmt::pinning::OnUnmap(addr, len);
	}
	return static_cast<int>(syscall(SYS_madvise, addr, len, advice));
}

extern "C" __attribute__((visibility("default"))) int
pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg) noexcept
{
	cmt::PthreadCreateFn real = cmt::OriginalPthreadCreate();
	if (real == nullptr)
	{
		return EAGAIN;
	}
	const auto from = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
	if (cmt::IsCudaDriverAddress(from))
	{
		auto* start_state = static_cast<cmt::DriverThreadStart*>(malloc(sizeof(cmt::DriverThreadStart)));
		if (start_state == nullptr)
		{
			cmt::Panic("cannot allocate CUDA driver thread state");
		}
		start_state->fn = start;
		start_state->arg = arg;
		const int rc = real(thread, attr, &cmt::DriverThreadStartEntry, start_state);
		if (rc != 0)
		{
			free(start_state);
		}
		return rc;
	}
	return real(thread, attr, start, arg);
}

#else

namespace cmt {
namespace pinning {
namespace detail {

constexpr uint64_t kPinMask = (uint64_t{1} << 24) - 1;
constexpr uint64_t kDead = uint64_t{1} << 24;
constexpr uint64_t kClaim = uint64_t{1} << 25;
constexpr uint64_t kLeaked = uint64_t{1} << 26;
constexpr unsigned kPhaseShift = 28;
constexpr uint64_t kPhaseMask = uint64_t{3} << kPhaseShift;
constexpr uint64_t kFree = 0;
constexpr uint64_t kFilling = 1;
constexpr uint64_t kReserved = 2;
constexpr uint64_t kRegistered = 3;
constexpr unsigned kGenShift = 32;

inline uint64_t PhaseOf(uint64_t w)
{
	return (w & kPhaseMask) >> kPhaseShift;
}

inline uint64_t PinsOf(uint64_t w)
{
	return w & kPinMask;
}

inline uint64_t GenOf(uint64_t w)
{
	return w >> kGenShift;
}

inline uint64_t WithPhase(uint64_t w, uint64_t p)
{
	return (w & ~kPhaseMask) | (p << kPhaseShift);
}

struct Slot
{
	std::atomic<uint64_t> word{0};
	std::atomic<uintptr_t> start{0};
	std::atomic<uintptr_t> end{0};
	std::atomic<uintptr_t> ustart{0};
	std::atomic<uintptr_t> uend{0};
	std::atomic<cu::CUcontext> ctx{nullptr};
};

constexpr unsigned kSlots = 1024;
Slot g_slots[kSlots];
std::atomic<unsigned> g_hwm{0};
std::atomic<long> g_used{0};
std::atomic<std::size_t> g_charged_bytes{0};
std::atomic<bool> g_forked_child{false};

thread_local CopyUse* t_copy_use = nullptr;

struct CopyUseRegistry
{
	Mutex mutex;
	CopyUse* head = nullptr;

	bool CanReserve(const CopyUse* owner, uintptr_t start, uintptr_t end) const
	{
		if (owner == nullptr || !owner->tracked || !owner->valid || start >= end || start < owner->start ||
			end > owner->end)
		{
			return false;
		}
		for (const CopyUse* use = head; use != nullptr; use = use->next)
		{
			if (use != owner && use->start < end && start < use->end)
			{
				return false;
			}
		}
		return true;
	}
};

CopyUseRegistry& CopyUses()
{
	static CopyUseRegistry* registry = new CopyUseRegistry;
	return *registry;
}

std::atomic<int> g_reaper{0};
std::atomic<bool> g_stop{false};

struct View
{
	uint64_t w;
	uintptr_t s;
	uintptr_t e;
	uintptr_t us;
	uintptr_t ue;
	cu::CUcontext ctx;
};

bool ViewSlot(unsigned i, View* v)
{
	Slot& sl = g_slots[i];
	for (;;)
	{
		const uint64_t w1 = sl.word.load(std::memory_order_acquire);
		const uint64_t ph = PhaseOf(w1);
		if (ph == kFree)
		{
			return false;
		}
		if (ph == kFilling)
		{
			*v = View{w1, 0, UINTPTR_MAX, 0, UINTPTR_MAX, nullptr};
			return true;
		}
		v->s = sl.start.load(std::memory_order_relaxed);
		v->e = sl.end.load(std::memory_order_relaxed);
		v->us = sl.ustart.load(std::memory_order_relaxed);
		v->ue = sl.uend.load(std::memory_order_relaxed);
		v->ctx = sl.ctx.load(std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_acquire);
		const uint64_t w2 = sl.word.load(std::memory_order_relaxed);
		if (GenOf(w2) == GenOf(w1))
		{
			v->w = w2;
			return true;
		}
	}
}

bool MarkDead(unsigned i, uint64_t seen)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	uint64_t w = word.load(std::memory_order_acquire);
	for (;;)
	{
		if (GenOf(w) != GenOf(seen) || PhaseOf(w) == kFree)
		{
			return false;
		}
		if ((w & kDead) != 0 || word.compare_exchange_weak(w, w | kDead, std::memory_order_acq_rel))
		{
			return true;
		}
	}
}

void ReleaseSlot(unsigned i)
{
	Slot& slot = g_slots[i];
	const std::size_t bytes =
		slot.end.load(std::memory_order_relaxed) - slot.start.load(std::memory_order_relaxed);
	g_charged_bytes.fetch_sub(bytes, std::memory_order_relaxed);
	std::atomic<uint64_t>& word = slot.word;
	uint64_t w = word.load(std::memory_order_relaxed);
	while (!word.compare_exchange_weak(w, (GenOf(w) + 1) << kGenShift, std::memory_order_release,
									   std::memory_order_relaxed))
	{
	}
	g_used.fetch_sub(1, std::memory_order_relaxed);
}

bool PinSlot(unsigned i, uint64_t gen)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	uint64_t w = word.load(std::memory_order_acquire);
	for (;;)
	{
		if (GenOf(w) != gen || PhaseOf(w) != kRegistered || (w & (kDead | kClaim)) != 0 ||
			PinsOf(w) == kPinMask)
		{
			return false;
		}
		if (word.compare_exchange_weak(w, w + 1, std::memory_order_acq_rel))
		{
			return true;
		}
	}
}

bool TryClaim(unsigned i, uint64_t gen)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	uint64_t w = word.load(std::memory_order_acquire);
	for (;;)
	{
		if (GenOf(w) != gen || PhaseOf(w) != kRegistered || (w & (kClaim | kLeaked)) != 0 || PinsOf(w) != 0 ||
			(w & kDead) == 0)
		{
			return false;
		}
		if (word.compare_exchange_weak(w, w | kDead | kClaim, std::memory_order_acq_rel))
		{
			return true;
		}
	}
}

uintptr_t PageSize()
{
	static std::atomic<uintptr_t> pg{0};
	uintptr_t p = pg.load(std::memory_order_relaxed);
	if (p == 0)
	{
		const long reported = sysconf(_SC_PAGESIZE);
		if (reported <= 0 || (reported & (reported - 1)) != 0)
		{
			Panic("cannot determine a valid host page size for registration");
		}
		p = static_cast<uintptr_t>(reported);
		pg.store(p, std::memory_order_relaxed);
	}
	return p;
}

bool PageRange(const void* pointer, std::size_t bytes, uintptr_t* begin, uintptr_t* end)
{
	const auto address = reinterpret_cast<uintptr_t>(pointer);
	const uintptr_t page = PageSize();
	if (bytes > UINTPTR_MAX - address || address + bytes > UINTPTR_MAX - (page - 1))
	{
		return false;
	}
	*begin = address & ~(page - 1);
	*end = (address + bytes + page - 1) & ~(page - 1);
	return true;
}

constexpr int kCachedDevices = 64;
std::atomic<std::size_t> g_device_memory[kCachedDevices];

std::size_t CurrentDeviceMemory()
{
	if (cu::g_api.cuCtxGetDevice == nullptr || cu::g_api.cuDeviceTotalMem_v2 == nullptr)
	{
		return 0;
	}
	cu::CUdevice device = 0;
	t_no_driver++;
	cu::CUresult rc = cu::g_api.cuCtxGetDevice(&device);
	--t_no_driver;
	const bool cacheable = rc == cu::kSuccess && device >= 0 && device < kCachedDevices;
	if (cacheable)
	{
		const std::size_t known = g_device_memory[device].load(std::memory_order_relaxed);
		if (known != 0)
		{
			return known;
		}
	}
	std::size_t bytes = 0;
	if (rc == cu::kSuccess)
	{
		t_no_driver++;
		rc = cu::g_api.cuDeviceTotalMem_v2(&bytes, device);
		--t_no_driver;
	}
	if (rc != cu::kSuccess || bytes == 0)
	{
		static std::atomic<bool> reported{false};
		if (!reported.exchange(true, std::memory_order_relaxed))
		{
			LogLine("note: cannot read the GPU's memory size (device={} rc={}), progressive pinning's default "
					"limit; such copies take the driver or staging path. CMT_PINNING_MAX_MIB sets a limit "
					"instead",
					device, rc);
		}
		return 0;
	}
	if (cacheable)
	{
		g_device_memory[device].store(bytes, std::memory_order_relaxed);
	}
	return bytes;
}

std::size_t RegistrationBudget(const PinningSettings& selected)
{
	const std::size_t run_time =
		GetConfig().pinning_max_bytes != 0 ? GetConfig().pinning_max_bytes : CurrentDeviceMemory();
	return selected.max_registered_bytes ? std::min<std::size_t>(run_time, *selected.max_registered_bytes)
										 : run_time;
}

void Backoff(unsigned round)
{
	if (round < 64)
	{
		sched_yield();
		return;
	}
	const long us = std::min<long>(1000, 10L << std::min<unsigned>(6, (round - 64) / 16));
	timespec ts{0, us * 1000};
	nanosleep(&ts, nullptr);
}

} // namespace detail

using namespace detail;

bool Enabled()
{
	const HookConfig& cfg = GetConfig();
	if (!cfg.valid)
	{
		return false;
	}
	const bool requested = cfg.fixed_benchmark
							   ? cfg.benchmark_pinning.has_value()
							   : cfg.progressive_pinning == ProgressivePinning::On ||
									 (cfg.progressive_pinning == ProgressivePinning::Auto && cfg.tune_table &&
									  cfg.tune_table->HasProgressivePinning());
	if (!requested)
	{
		return false;
	}
	const bool available = ProgressivePinningAvailable();
	if (!available && cfg.progressive_pinning == ProgressivePinning::On)
	{
		static std::atomic<bool> reported{false};
		if (!reported.exchange(true, std::memory_order_relaxed))
		{
			LogLine(
				"note: progressive pinning requested, but verified patched-kernel provenance is unavailable; "
				"using the ordinary driver/staging path");
		}
	}
	return available;
}

PinningSettings ManualSettings()
{
	const auto& cfg = GetConfig();
	return {cfg.pinning_first_bytes, cfg.pinning_step_bytes, std::nullopt, cfg.pinning_read_only};
}

CopyUse::CopyUse(const void* host, std::size_t bytes)
{
	if (bytes == 0 || t_no_driver != 0 || g_forked_child.load(std::memory_order_relaxed))
	{
		return;
	}
	NoDriverScope bookkeeping;
	if (!Enabled() && !Active())
	{
		return;
	}
	valid = host != nullptr && PageRange(host, bytes, &start, &end);
	if (!valid)
	{
		start = 0;
		end = UINTPTR_MAX;
	}
	CopyUseRegistry& registry = CopyUses();
	std::lock_guard<Mutex> guard(registry.mutex);
	previous = t_copy_use;
	next = registry.head;
	registry.head = this;
	t_copy_use = this;
	tracked = true;
}

CopyUse::~CopyUse()
{
	if (!tracked)
	{
		return;
	}
	if (g_forked_child.load(std::memory_order_relaxed))
	{
		t_copy_use = previous;
		tracked = false;
		return;
	}
	if (t_copy_use != this)
	{
		Panic("registration copy-use scopes must be released in stack order");
	}
	NoDriverScope bookkeeping;
	CopyUseRegistry& registry = CopyUses();
	std::lock_guard<Mutex> guard(registry.mutex);
	CopyUse** link = &registry.head;
	while (*link != this)
	{
		link = &(*link)->next;
	}
	*link = next;
	t_copy_use = previous;
	tracked = false;
}

bool CopyUse::HasRegistrationOverlap() const
{
	if (!tracked)
	{
		return false;
	}
	if (!valid)
	{
		return true;
	}
	const unsigned hwm = g_hwm.load(std::memory_order_acquire);
	for (unsigned i = 0; i < hwm; ++i)
	{
		View view;
		if (ViewSlot(i, &view) && view.s < end && start < view.e && (view.w & kLeaked) == 0)
		{
			return true;
		}
	}
	return false;
}

bool Active()
{
	return g_used.load(std::memory_order_relaxed) > 0 && !g_forked_child.load(std::memory_order_relaxed);
}

} // namespace pinning
} // namespace cmt

namespace cmt {
namespace pinning {
namespace detail {

using namespace cu;

namespace {

struct QNode
{
	QNode* next;
	std::size_t usable;
};

std::atomic<QNode*> g_quarantine{nullptr};
std::atomic<long> g_leaked{0};

std::atomic<uint64_t> g_wake{0};
std::atomic<bool> g_draining{false};
thread_local bool notifying_reaper = false;
constexpr int kReaperRetryMilliseconds = 50;

struct ReaperWake
{
	Mutex mutex;
	ConditionVariable changed;
};

ReaperWake& WakeState()
{
	static ReaperWake* state = new ReaperWake;
	return *state;
}

class NotifyGuard
{
public:
	NotifyGuard()
	{
		notifying_reaper = true;
	}

	~NotifyGuard()
	{
		notifying_reaper = false;
	}
};

} // namespace

void WakeReaper()
{
	g_wake.fetch_add(1, std::memory_order_release);
	if (notifying_reaper || g_reaper.load(std::memory_order_acquire) != 2 ||
		g_forked_child.load(std::memory_order_relaxed))
	{
		return;
	}
	NotifyGuard guard;
	auto& wake = WakeState();
	wake.mutex.lock();
	wake.changed.Signal();
	wake.mutex.unlock();
}

void Quarantine(void* p, std::size_t usable)
{
	if (usable < sizeof(QNode))
	{
		g_leaked.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto* node = static_cast<QNode*>(p);
	node->usable = usable;
	QNode* head = g_quarantine.load(std::memory_order_relaxed);
	do
	{
		node->next = head;
	} while (!g_quarantine.compare_exchange_weak(head, node, std::memory_order_release,
												 std::memory_order_relaxed));
	WakeReaper();
}

constexpr unsigned kRetiredContexts = 64;
std::atomic<CUcontext> g_retired[kRetiredContexts];
std::atomic<bool> g_any_retired{false};
std::atomic<bool> g_retired_overflow{false};

void RetireContext(CUcontext ctx)
{
	if (ctx == nullptr)
	{
		return;
	}
	g_any_retired.store(true, std::memory_order_release);
	for (std::atomic<CUcontext>& entry : g_retired)
	{
		CUcontext seen = entry.load(std::memory_order_acquire);
		if (seen == ctx || (seen == nullptr && (entry.compare_exchange_strong(seen, ctx) || seen == ctx)))
		{
			return;
		}
	}
	g_retired_overflow.store(true, std::memory_order_release);
}

bool ContextRetired(CUcontext ctx)
{
	if (!g_any_retired.load(std::memory_order_acquire))
	{
		return false;
	}
	if (g_retired_overflow.load(std::memory_order_acquire))
	{
		return true;
	}
	for (const std::atomic<CUcontext>& entry : g_retired)
	{
		if (entry.load(std::memory_order_acquire) == ctx)
		{
			return true;
		}
	}
	return false;
}

void UnretireContext(CUcontext ctx, bool all)
{
	if (!g_any_retired.load(std::memory_order_acquire))
	{
		return;
	}
	for (std::atomic<CUcontext>& entry : g_retired)
	{
		CUcontext seen = entry.load(std::memory_order_acquire);
		if (seen != nullptr && (all || seen == ctx))
		{
			entry.compare_exchange_strong(seen, nullptr);
		}
	}
	if (all)
	{
		g_retired_overflow.store(false, std::memory_order_release);
	}
}

void ReportLeakOnce(CUresult rc)
{
	static std::atomic<bool> said{false};
	if (rc != kErrorDeinitialized && !said.exchange(true, std::memory_order_relaxed))
	{
		LogLine("error: progressive pinning cannot confirm that a DMA finished or cannot release a registration "
				"(rc={}, usually a sticky CUDA error); the registration and its memory are kept (leaked), not "
				"released, and that context's copies use the original driver. Reported once.",
				rc);
	}
}

void LeakSlot(unsigned i, bool drop_pin, CUresult rc)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	uint64_t w = word.load(std::memory_order_relaxed);
	uint64_t nw;
	do
	{
		nw = ((drop_pin ? w - 1 : w) & ~kClaim) | kDead | kLeaked;
	} while (!word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed));
	ReportLeakOnce(rc);
}

void UnregisterClaimed(unsigned i)
{
	Slot& sl = g_slots[i];
	const uintptr_t s = sl.start.load(std::memory_order_relaxed);
	const uintptr_t e = sl.end.load(std::memory_order_relaxed);
	const CUcontext ctx = sl.ctx.load(std::memory_order_relaxed);
	t_no_driver++;
	CUcontext cur = nullptr;
	CUresult rc = ctx == nullptr || g_api.cuCtxGetCurrent == nullptr || g_api.cuCtxSetCurrent == nullptr ||
						  g_api.cuMemHostUnregister == nullptr
					  ? kErrorNotSupported
					  : g_api.cuCtxGetCurrent(&cur);
	bool switched = false;
	if (rc == kSuccess && cur != ctx)
	{
		rc = g_api.cuCtxSetCurrent(ctx);
		switched = rc == kSuccess;
	}
	if (rc == kSuccess)
	{
		const RelaxedCaptureScope relaxed;
		rc = g_api.cuMemHostUnregister(reinterpret_cast<void*>(s));
	}
	if (switched)
	{
		const CUresult restored = g_api.cuCtxSetCurrent(cur);
		if (restored != kSuccess && restored != kErrorDeinitialized)
		{
			Panic("cannot restore the caller's CUDA context after unregistration");
		}
	}
	--t_no_driver;
	LogDebug("pinning: unregister {:#x} + {} KiB rc={}", static_cast<unsigned long>(s),
			 static_cast<unsigned long>((e - s) >> 10), rc);
	if (rc != kSuccess)
	{
		LeakSlot(i, false, rc);
		RetireContext(ctx);
		return;
	}
	ReleaseSlot(i);
	if (g_quarantine.load(std::memory_order_relaxed) != nullptr)
	{
		WakeReaper();
	}
}

void ReleasePin(unsigned i, bool kill)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	uint64_t w = word.load(std::memory_order_relaxed);
	uint64_t nw;
	do
	{
		nw = (w - 1) | (kill ? kDead : 0);
	} while (!word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed));
	if ((nw & kDead) != 0 && PinsOf(nw) == 0 && (nw & (kClaim)) == 0 && TryClaim(i, GenOf(nw)))
	{
		UnregisterClaimed(i);
	}
}

bool UserOverlapAny(uintptr_t a, uintptr_t b)
{
	const unsigned hwm = g_hwm.load(std::memory_order_acquire);
	for (unsigned i = 0; i < hwm; i++)
	{
		View v;
		if (ViewSlot(i, &v) && v.us < b && a < v.ue)
		{
			return true;
		}
	}
	return false;
}

bool MarkRange(uintptr_t a, uintptr_t b, bool user, bool* live = nullptr)
{
	bool hit = false;
	const unsigned hwm = g_hwm.load(std::memory_order_acquire);
	for (unsigned i = 0; i < hwm; i++)
	{
		View v;
		if (!ViewSlot(i, &v))
		{
			continue;
		}
		const bool overlaps = user ? (v.us < b && a < v.ue) : (v.s < b && a < v.e);
		if (overlaps && MarkDead(i, v.w))
		{
			hit = true;
			if (live != nullptr && (v.w & kLeaked) == 0)
			{
				*live = true;
			}
		}
	}
	return hit;
}

void Drain()
{
	if (g_draining.exchange(true, std::memory_order_acquire))
	{
		return;
	}
	QNode* list = g_quarantine.exchange(nullptr, std::memory_order_acq_rel);
	for (QNode* q = list; q != nullptr; q = q->next)
	{
		const auto a = reinterpret_cast<uintptr_t>(q);
		MarkRange(a, a + q->usable, true);
	}
	const unsigned hwm = g_hwm.load(std::memory_order_acquire);
	for (unsigned i = 0; i < hwm && !g_stop.load(std::memory_order_acquire); i++)
	{
		View v;
		if (ViewSlot(i, &v) && PhaseOf(v.w) == kRegistered && (v.w & kDead) != 0 && TryClaim(i, GenOf(v.w)))
		{
			UnregisterClaimed(i);
		}
	}
	QNode* keep = nullptr;
	while (list != nullptr)
	{
		QNode* next = list->next;
		const auto a = reinterpret_cast<uintptr_t>(list);
		if (UserOverlapAny(a, a + list->usable))
		{
			list->next = keep;
			keep = list;
		}
		else
		{
			copying::services::ReleaseAllocation(list);
		}
		list = next;
	}
	while (keep != nullptr)
	{
		QNode* next = keep->next;
		QNode* head = g_quarantine.load(std::memory_order_relaxed);
		do
		{
			keep->next = head;
		} while (!g_quarantine.compare_exchange_weak(head, keep, std::memory_order_release,
													 std::memory_order_relaxed));
		keep = next;
	}
	g_draining.store(false, std::memory_order_release);
}

namespace {

void* ReaperMain(void*)
{
	pthread_setname_np(pthread_self(), "cmt-pinning");
	for (;;)
	{
		const uint64_t seen = g_wake.load(std::memory_order_acquire);
		if (g_stop.load(std::memory_order_acquire))
		{
			return nullptr;
		}
		Drain();
		NotifyGuard guard;
		auto& wake = WakeState();
		wake.mutex.lock();
		if (!g_stop.load(std::memory_order_acquire) && g_wake.load(std::memory_order_acquire) == seen)
		{
			wake.changed.WaitWithTimeout(&wake.mutex, absl::Milliseconds(kReaperRetryMilliseconds));
		}
		wake.mutex.unlock();
	}
}

} // namespace

void EnsureReaper()
{
	int state = g_reaper.load(std::memory_order_acquire);
	if (state != 0 || !g_reaper.compare_exchange_strong(state, 1))
	{
		return;
	}
	NoDriverScope placement_guard;
	const auto inputs = HookPlacementInputs(GetConfig());
	auto placement = cu::BackgroundPlacement(inputs.explicit_cpus, inputs.recorded);
	if (!placement)
	{
		g_reaper.store(3, std::memory_order_release);
		LogLine("pinning: {}; copies release dead segments inline", placement.error().Message());
		return;
	}
	pthread_attr_t attr;
	pthread_t th;
	int rc = pthread_attr_init(&attr);
	if (rc == 0)
	{
		rc = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		if (rc == 0)
		{
			auto placed = cmt::ApplyThreadPlacement(attr, *placement);
			if (!placed)
			{
				LogLine("pinning: {}; copies release dead segments inline", placed.error().Message());
				rc = placed.error().GetCode() ? placed.error().GetCode() : EINVAL;
			}
		}
		if (rc == 0)
		{
			rc = pthread_create(&th, &attr, &ReaperMain, nullptr);
		}
		pthread_attr_destroy(&attr);
	}
	g_reaper.store(rc == 0 ? 2 : 3, std::memory_order_release);
	if (rc != 0)
	{
		LogDebug("pinning: cannot start the reaper (rc={}); copies release dead segments inline", rc);
	}
}

void Settle(uintptr_t a, uintptr_t b)
{
	for (unsigned round = 0;; round++)
	{
		bool wait = false;
		const unsigned hwm = g_hwm.load(std::memory_order_acquire);
		for (unsigned i = 0; i < hwm; i++)
		{
			View v;
			if (!ViewSlot(i, &v) || !(v.s < b && a < v.e))
			{
				continue;
			}
			const uint64_t ph = PhaseOf(v.w);
			if (ph == kFilling)
			{
				wait = true;
				continue;
			}
			if ((v.w & (kDead | kLeaked)) != kDead)
			{
				continue;
			}
			if (ph == kReserved || (v.w & kClaim) != 0)
			{
				wait = true;
				continue;
			}
			if (PinsOf(v.w) != 0)
			{
				wait = true;
				continue;
			}
			if (!TryClaim(i, GenOf(v.w)))
			{
				wait = true;
				continue;
			}
			UnregisterClaimed(i);
		}
		if (!wait)
		{
			return;
		}
		Backoff(round);
	}
}

} // namespace detail
} // namespace pinning
} // namespace cmt

namespace cmt {
namespace pinning {

using namespace cu;
using namespace detail;

namespace {

bool RequiredCopyApis(Dir direction)
{
	const bool copy_available =
		direction == Dir::H2D ? g_api.cuMemcpyHtoDAsync_v2 != nullptr : g_api.cuMemcpyDtoHAsync_v2 != nullptr;
	return copy_available && g_api.cuCtxGetCurrent != nullptr && g_api.cuCtxSetCurrent != nullptr &&
		   g_api.cuMemHostUnregister != nullptr &&
		   (g_api.cuStreamSynchronize != nullptr || g_api.cuCtxSynchronize != nullptr);
}

bool CurrentContextRetired()
{
	if (!g_any_retired.load(std::memory_order_acquire))
	{
		return false;
	}
	CUcontext ctx = nullptr;
	t_no_driver++;
	const CUresult rc = g_api.cuCtxGetCurrent(&ctx);
	--t_no_driver;
	return rc == kSuccess && ContextRetired(ctx);
}

CUresult SubmitDma(Dir dir, CUdeviceptr dev, uint8_t* host, std::size_t n, CUstream stream)
{
	t_no_driver++;
	const CUresult rc = dir == Dir::H2D ? g_api.cuMemcpyHtoDAsync_v2(dev, host, n, stream)
										: g_api.cuMemcpyDtoHAsync_v2(host, dev, n, stream);
	--t_no_driver;
	return rc;
}

CUresult WaitForDma(CUstream stream)
{
	t_no_driver++;
	const RelaxedCaptureScope relaxed;
	const CUresult rc = g_api.cuStreamSynchronize != nullptr ? g_api.cuStreamSynchronize(stream)
						: g_api.cuCtxSynchronize != nullptr	 ? g_api.cuCtxSynchronize()
															 : kErrorNotSupported;
	--t_no_driver;
	return rc;
}

void ReportReadOnlyFallback(CUdevice device, CUresult code, int supported)
{
	static std::atomic<bool> reported{false};
	if (!reported.exchange(true, std::memory_order_relaxed))
	{
		LogLine("note: requested read-only registration is unavailable (device={} rc={} "
				"attribute113={}); "
				"using writable registration",
				device, code, supported);
	}
}

unsigned RegistrationFlags(Dir dir, bool read_only)
{
	if (!read_only || dir != Dir::H2D)
	{
		return kMemHostRegisterPortable;
	}
	if (g_api.cuCtxGetCurrent == nullptr || g_api.cuCtxGetDevice == nullptr ||
		g_api.cuDeviceGetAttribute == nullptr)
	{
		ReportReadOnlyFallback(-1, kErrorNotSupported, 0);
		return kMemHostRegisterPortable;
	}
	CUcontext current = nullptr;
	CUdevice device = 0;
	int supported = 0;
	t_no_driver++;
	CUresult rc = g_api.cuCtxGetCurrent(&current);
	if (rc == kSuccess && current != nullptr)
	{
		rc = g_api.cuCtxGetDevice(&device);
		if (rc == kSuccess)
		{
			rc = g_api.cuDeviceGetAttribute(&supported, kDevAttrReadOnlyHostRegisterSupported, device);
		}
	}
	--t_no_driver;
	const bool use_read_only = rc == kSuccess && current != nullptr && supported != 0;
	if (!use_read_only)
	{
		ReportReadOnlyFallback(device, rc, supported);
	}
	LogDebug("pinning: read-only capability device={} rc={} supported={} selected={}", device, rc, supported,
			 use_read_only);
	return kMemHostRegisterPortable | (use_read_only ? kMemHostRegisterReadOnly : 0u);
}

int Reserve(uintptr_t s, uintptr_t e, uintptr_t us, uintptr_t ue, CUcontext ctx, std::size_t cap)
{
	if (t_copy_use == nullptr || g_forked_child.load(std::memory_order_relaxed))
	{
		return -1;
	}
	NoDriverScope bookkeeping;
	CopyUseRegistry& registry = CopyUses();
	std::lock_guard<Mutex> guard(registry.mutex);
	if (!registry.CanReserve(t_copy_use, s, e))
	{
		return -1;
	}
	const std::size_t bytes = e - s;
	const std::size_t charged = g_charged_bytes.load(std::memory_order_relaxed);
	if (bytes > cap || charged > cap - bytes)
	{
		return -1;
	}
	for (unsigned i = 0; i < kSlots; i++)
	{
		Slot& sl = g_slots[i];
		uint64_t w = sl.word.load(std::memory_order_relaxed);
		if (PhaseOf(w) != kFree)
		{
			continue;
		}
		unsigned h = g_hwm.load(std::memory_order_relaxed);
		while (h < i + 1 && !g_hwm.compare_exchange_weak(h, i + 1, std::memory_order_acq_rel))
		{
		}
		g_used.fetch_add(1, std::memory_order_relaxed);
		if (!sl.word.compare_exchange_strong(w, WithPhase(w, kFilling), std::memory_order_acq_rel))
		{
			g_used.fetch_sub(1, std::memory_order_relaxed);
			continue;
		}
		std::atomic_thread_fence(std::memory_order_release); // pairs with ViewSlot()'s fence
		sl.start.store(s, std::memory_order_relaxed);
		sl.end.store(e, std::memory_order_relaxed);
		sl.ustart.store(us, std::memory_order_relaxed);
		sl.uend.store(ue, std::memory_order_relaxed);
		sl.ctx.store(ctx, std::memory_order_relaxed);
		g_charged_bytes.fetch_add(bytes, std::memory_order_relaxed);
		w = sl.word.load(std::memory_order_relaxed);
		while (!sl.word.compare_exchange_weak(w, WithPhase(w, kReserved) + 1, std::memory_order_release,
											  std::memory_order_relaxed))
		{
		}
		return static_cast<int>(i);
	}
	return -1;
}

bool FinishRegistration(unsigned i, bool registered)
{
	std::atomic<uint64_t>& word = g_slots[i].word;
	if (!registered)
	{
		ReleaseSlot(i);
		return false;
	}
	uint64_t w = word.load(std::memory_order_relaxed);
	while (!word.compare_exchange_weak(w, WithPhase(w, kRegistered), std::memory_order_acq_rel,
									   std::memory_order_relaxed))
	{
	}
	if ((w & kDead) != 0)
	{
		ReleasePin(i, true);
		return false;
	}
	return true;
}

enum class Kind : uint8_t
{
	Pinned,
	Gap
};

struct Piece
{
	uintptr_t s;
	uintptr_t e;
	Kind kind;
};

struct Cover
{
	uintptr_t s;
	uintptr_t e;
	Kind kind;
};

struct Retirement
{
	unsigned slot;
	uint64_t generation;
};

// Release our pins first: another copy may be waiting for them while holding pins we need.
bool WaitForRetirements(const std::vector<Retirement>& retirements)
{
	bool released = true;
	for (const Retirement& retirement : retirements)
	{
		for (unsigned round = 0;; ++round)
		{
			const uint64_t word = g_slots[retirement.slot].word.load(std::memory_order_acquire);
			if (GenOf(word) != retirement.generation || PhaseOf(word) == kFree)
			{
				break;
			}
			if ((word & kLeaked) != 0)
			{
				released = false;
				break;
			}
			if (TryClaim(retirement.slot, retirement.generation))
			{
				UnregisterClaimed(retirement.slot);
				break;
			}
			Backoff(round);
		}
	}
	return released;
}

void PlanPieces(uintptr_t a, uintptr_t b, std::vector<Cover>* covers, std::vector<unsigned>* pinned)
{
	for (unsigned round = 0;; round++)
	{
		Settle(a, b);
		covers->clear();
		bool again = false;
		const unsigned hwm = g_hwm.load(std::memory_order_acquire);
		for (unsigned i = 0; i < hwm; i++)
		{
			View v;
			if (!ViewSlot(i, &v) || !(v.s < b && a < v.e) || (v.w & kLeaked) != 0)
			{
				continue;
			}
			const uint64_t ph = PhaseOf(v.w);
			if (ph == kFilling || (v.w & (kDead | kClaim)) != 0)
			{
				again = true;
				continue;
			}
			if (ph == kReserved)
			{
				again = true;
				continue;
			}
			if (!PinSlot(i, GenOf(v.w)))
			{
				again = true;
				continue;
			}
			pinned->push_back(i);
			covers->push_back(Cover{std::max(a, v.s), std::min(b, v.e), Kind::Pinned});
		}
		if (!again)
		{
			std::sort(std::begin(*covers), std::end(*covers),
					  [](const Cover& x, const Cover& y) { return x.s < y.s; });
			return;
		}
		for (unsigned i : *pinned)
		{
			ReleasePin(i, false);
		}
		pinned->clear();
		Backoff(round);
	}
}

constexpr uintptr_t kGpuLargePageMask = (uintptr_t{2} << 20) - 1;

std::vector<Piece> PiecesOf(uintptr_t a, uintptr_t b, const std::vector<Cover>& covers,
							const PinningSettings& settings)
{
	std::vector<Piece> out;
	const uintptr_t pg = PageSize();
	const uintptr_t step = std::max<uintptr_t>(pg, (settings.max_chunk_bytes / pg) * pg);
	const uintptr_t first =
		std::min<uintptr_t>(step, std::max<uintptr_t>(pg, (settings.first_chunk_bytes / pg) * pg));
	auto gap = [&](uintptr_t x, uintptr_t gend)
	{
		uintptr_t cur = std::empty(out) || out.back().kind != Kind::Gap ? first : step;
		for (uintptr_t s = x; s < gend;)
		{
			uintptr_t e = s + std::min(cur, gend - s);
			if (e < gend && (e & ~kGpuLargePageMask) > s)
			{
				e &= ~kGpuLargePageMask;
			}
			out.push_back(Piece{s, e, Kind::Gap});
			s = e;
			cur = cur >= step - cur ? step : cur * 2;
		}
	};
	uintptr_t x = a;
	for (const Cover& c : covers)
	{
		if (c.e <= x)
		{
			continue;
		}
		if (c.s > x)
		{
			gap(x, c.s);
			x = c.s;
		}
		out.push_back(Piece{x, c.e, c.kind});
		x = c.e;
	}
	if (x < b)
	{
		gap(x, b);
	}
	return out;
}

CUresult CopyThrough(Dir dir, CUdeviceptr dev, uint8_t* host, std::size_t n, CUstream stream,
					 const std::vector<Piece>& pcs, bool may_register, const PinningSettings& settings,
					 std::size_t budget, std::vector<Retirement>& retirements, bool& submitted_dma,
					 bool& unsettled, Observation* observation)
{
	CUcontext ctx = nullptr;
	t_no_driver++;
	const CUresult context_rc = g_api.cuCtxGetCurrent(&ctx);
	--t_no_driver;
	if (context_rc != kSuccess || ctx == nullptr)
	{
		return context_rc != kSuccess ? context_rc : kErrorInvalidContext;
	}
	const uintptr_t p = reinterpret_cast<uintptr_t>(host);
	const uintptr_t q = p + n;
	auto user = [&](const Piece& pc, uintptr_t* s, uintptr_t* e)
	{
		*s = std::max(p, pc.s);
		*e = std::min(q, pc.e);
		return *e > *s;
	};
	CUresult rc = kSuccess;
	std::vector<std::size_t> gaps;
	std::vector<unsigned> owned;
	gaps.reserve(std::size(pcs));
	owned.reserve(std::size(pcs));
	for (std::size_t i = 0; i < std::size(pcs); ++i)
	{
		uintptr_t s;
		uintptr_t e;
		if (pcs[i].kind == Kind::Gap)
		{
			gaps.push_back(i);
		}
		else if (user(pcs[i], &s, &e) && rc == kSuccess)
		{
			submitted_dma = true;
			rc = SubmitDma(dir, dev + (s - p), reinterpret_cast<uint8_t*>(s), e - s, stream);
			if (observation != nullptr && rc == kSuccess)
			{
				observation->borrowed_bytes += e - s;
			}
		}
	}
	bool registration_failed = !may_register;
	unsigned register_flags =
		may_register ? RegistrationFlags(dir, settings.read_only) : kMemHostRegisterPortable;
	if (observation != nullptr)
	{
		observation->read_only = (register_flags & kMemHostRegisterReadOnly) != 0;
	}
	for (std::size_t k : gaps)
	{
		if (rc != kSuccess)
		{
			break;
		}
		const Piece& pc = pcs[k];
		uintptr_t s;
		uintptr_t e;
		if (!user(pc, &s, &e))
		{
			continue;
		}
		bool registered = false;
		int slot = -1;
		if (!registration_failed)
		{
			slot = Reserve(pc.s, pc.e, s, e, ctx, budget);
			if (slot < 0)
			{
				registration_failed = true;
			}
		}
		if (slot >= 0)
		{
			t_no_driver++;
			const RelaxedCaptureScope relaxed;
			CUresult r = g_api.cuMemHostRegister(reinterpret_cast<void*>(pc.s), pc.e - pc.s, register_flags);
			if ((register_flags & kMemHostRegisterReadOnly) != 0 && r != kSuccess)
			{
				if (r == kErrorNotSupported)
				{
					ReportReadOnlyFallback(-1, r, 0);
					register_flags = kMemHostRegisterPortable;
					if (observation != nullptr)
					{
						observation->read_only = 0;
					}
					r = g_api.cuMemHostRegister(reinterpret_cast<void*>(pc.s), pc.e - pc.s, register_flags);
				}
				else
				{
					rc = r;
				}
			}
			--t_no_driver;
			if (r == kSuccess)
			{
				const auto owned_slot = static_cast<unsigned>(slot);
				const uint64_t generation = GenOf(g_slots[owned_slot].word.load(std::memory_order_acquire));
				retirements.push_back(Retirement{owned_slot, generation});
				if (observation != nullptr)
				{
					observation->registered_bytes += pc.e - pc.s;
					++observation->chunks;
				}
			}
			registered = FinishRegistration(static_cast<unsigned>(slot), r == kSuccess);
			if (r != kSuccess)
			{
				LogDebug("pinning: register {:#x} + {} KiB failed rc={}", static_cast<unsigned long>(pc.s),
						 static_cast<unsigned long>((pc.e - pc.s) >> 10), r);
				registration_failed = true;
			}
		}
		if (registered)
		{
			owned.push_back(static_cast<unsigned>(slot));
		}
		if (rc != kSuccess)
		{
			break;
		}
		if (!registered && !submitted_dma)
		{
			break;
		}
		submitted_dma = true;
		rc = SubmitDma(dir, dev + (s - p), reinterpret_cast<uint8_t*>(s), e - s, stream);
		if (observation != nullptr && rc == kSuccess)
		{
			if (registered)
			{
				observation->registered_dma_bytes += e - s;
			}
			else
			{
				observation->pageable_dma_bytes += e - s;
			}
		}
	}
	if (submitted_dma)
	{
		const CUresult waited = WaitForDma(stream);
		if (waited != kSuccess)
		{
			unsettled = true;
			RetireContext(ctx);
			for (unsigned i : owned)
			{
				LeakSlot(i, true, waited);
			}
			return rc != kSuccess ? rc : waited;
		}
	}
	for (unsigned i : owned)
	{
		ReleasePin(i, true);
	}
	return rc;
}

} // namespace

bool TryCopy(Dir dir, CUdeviceptr dev, void* host, std::size_t n, CUstream stream, CUresult* rc,
			 const PinningSettings* selected, Observation* observation)
{
	if (n == 0 || t_no_driver != 0 || g_forked_child.load(std::memory_order_relaxed))
	{
		return false;
	}
	const HookConfig& cfg = GetConfig();
	if (!Enabled() && !Active())
	{
		return false;
	}
	if (g_reaper.load(std::memory_order_acquire) == 3)
	{
		Drain();
	}
	uintptr_t a = 0;
	uintptr_t b = 0;
	if (host == nullptr || n > UINT64_MAX - dev || !PageRange(host, n, &a, &b))
	{
		*rc = kErrorInvalidValue;
		return true;
	}
	auto* h = static_cast<uint8_t*>(host);
	const bool api_ready = RequiredCopyApis(dir);
	const bool eligible = selected != nullptr && Enabled() && n >= cfg.pinning_min_bytes && api_ready &&
						  g_api.cuMemHostRegister != nullptr && !CurrentContextRetired();
	if (!eligible)
	{
		bool overlap = false;
		const unsigned hwm = g_hwm.load(std::memory_order_acquire);
		for (unsigned i = 0; i < hwm && !overlap; i++)
		{
			View v;
			overlap = ViewSlot(i, &v) && v.s < b && a < v.e && (v.w & kLeaked) == 0;
		}
		if (!overlap)
		{
			return false;
		}
	}
	if (!api_ready)
	{
		*rc = kErrorNotSupported;
		return true;
	}
	std::vector<unsigned> pinned;
	std::vector<Cover> covers;
	pinned.reserve(kSlots);
	covers.reserve(kSlots);
	PlanPieces(a, b, &covers, &pinned);
	const bool covered = !std::empty(covers);
	if (!covered)
	{
		if (!eligible)
		{
			return false;
		}
		t_no_driver++;
		const HostPointerKind memory = ClassifyHostPointer(host);
		--t_no_driver;
		if (memory != HostPointerKind::Pageable)
		{
			return false;
		}
	}
	const PinningSettings settings = selected != nullptr ? *selected : PinningSettings{};
	const std::size_t budget = eligible ? RegistrationBudget(settings) : 0;
	const std::vector<Piece> pcs = PiecesOf(a, b, covers, settings);
	bool may_register = eligible;
	if (may_register)
	{
		std::size_t missing = 0;
		for (const Piece& pc : pcs)
		{
			missing += pc.kind == Kind::Gap ? pc.e - pc.s : 0;
		}
		if (missing > budget)
		{
			may_register = false;
		}
		else if (missing > 0)
		{
			may_register = g_charged_bytes.load(std::memory_order_relaxed) <= budget - missing;
		}
		if (!may_register && !covered)
		{
			return false;
		}
	}
	if (may_register)
	{
		EnsureReaper();
	}
	std::vector<Retirement> retirements;
	retirements.reserve(std::size(pcs) + std::size(pinned));
	bool submitted_dma = false;
	bool unsettled = false;
	if (observation != nullptr)
	{
		*observation = {};
		observation->first_chunk_bytes = settings.first_chunk_bytes;
		observation->max_chunk_bytes = settings.max_chunk_bytes;
		observation->max_registered_bytes = settings.max_registered_bytes.value_or(0);
	}
	*rc = CopyThrough(dir, dev, h, n, stream, pcs, may_register, settings, budget, retirements, submitted_dma,
					  unsettled, observation);
	for (unsigned i : pinned)
	{
		if (unsettled)
		{
			LeakSlot(i, true, *rc);
			continue;
		}
		retirements.push_back(Retirement{i, GenOf(g_slots[i].word.load(std::memory_order_acquire))});
		ReleasePin(i, true);
	}
	const bool released = WaitForRetirements(retirements);
	if (observation != nullptr)
	{
		observation->released_bytes = released ? observation->registered_bytes : 0;
		observation->completed = submitted_dma && *rc == kSuccess;
	}
	return submitted_dma || *rc != kSuccess;
}

} // namespace pinning
} // namespace cmt

namespace cmt {
namespace pinning {

using namespace cu;
using namespace detail;

bool OnFree(void* p, std::size_t usable)
{
	if (p == nullptr || usable == 0 || !Active())
	{
		return false;
	}
	const auto a = reinterpret_cast<uintptr_t>(p);
	if (!MarkRange(a, a + usable, true))
	{
		return false;
	}
	Quarantine(p, usable);
	return true;
}

bool Holds(const void* p, std::size_t usable)
{
	if (p == nullptr || usable == 0 || !Active())
	{
		return false;
	}
	const auto a = reinterpret_cast<uintptr_t>(p);
	return UserOverlapAny(a, a + usable);
}

void OnUnmap(const void* p, std::size_t n)
{
	if (n == 0 || !Active())
	{
		return;
	}
	uintptr_t a = 0;
	uintptr_t b = 0;
	if (!PageRange(p, n, &a, &b))
	{
		return;
	}
	bool live = false;
	if (!MarkRange(a, b, false, &live) || !live)
	{
		return;
	}
	if (t_no_driver != 0)
	{
		Panic("cannot safely unmap registered application memory from a CUDA callback or internal thread");
	}
	Settle(a, b);
}

void MarkDriverThread()
{
	t_no_driver = 1;
}

void AfterForkInChild()
{
	g_forked_child.store(true, std::memory_order_relaxed);
}

void Shutdown()
{
	g_stop.store(true, std::memory_order_release);
	WakeReaper();
}

namespace {

void DropRegistrations(CUcontext ctx, bool all_contexts) noexcept
{
	if (g_forked_child.load(std::memory_order_relaxed) || !Active())
	{
		return;
	}
	const unsigned hwm = g_hwm.load(std::memory_order_acquire);
	std::array<uint64_t, kSlots> pending{};
	for (unsigned i = 0; i < hwm; ++i)
	{
		const uint64_t word = g_slots[i].word.load(std::memory_order_acquire);
		if (PhaseOf(word) != kFree)
		{
			pending[i] = word;
		}
	}
	for (unsigned round = 0;; ++round)
	{
		bool wait = false;
		for (unsigned i = 0; i < hwm; ++i)
		{
			if (pending[i] == 0)
			{
				continue;
			}
			const uint64_t generation = GenOf(pending[i]);
			View view;
			if (!ViewSlot(i, &view) || GenOf(view.w) != generation)
			{
				pending[i] = 0;
				continue;
			}
			if (PhaseOf(view.w) == kFilling)
			{
				wait = true;
				continue;
			}
			if ((!all_contexts && view.ctx != ctx) || !MarkDead(i, view.w))
			{
				pending[i] = 0;
				continue;
			}
			if (TryClaim(i, generation))
			{
				UnregisterClaimed(i);
				pending[i] = 0;
				continue;
			}
			View now;
			if (ViewSlot(i, &now) && GenOf(now.w) == generation && (now.w & kLeaked) == 0)
			{
				wait = true;
			}
			else
			{
				pending[i] = 0;
			}
		}
		if (!wait)
		{
			return;
		}
		Backoff(round);
	}
}

} // namespace

void ReleaseContext(CUcontext ctx)
{
	DropRegistrations(ctx, false);
	UnretireContext(ctx, false);
}

void ReleaseAllRegistrations()
{
	DropRegistrations(nullptr, true);
	UnretireContext(nullptr, true);
}

} // namespace pinning
} // namespace cmt

#endif
