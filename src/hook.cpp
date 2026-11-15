#include "hook.h"

#include <optional>

#include "copying/host_alloc.h"

#if !defined(CMT_HOOK_PRELOAD_ONLY)

namespace cmt {
namespace driver {

namespace {
const patch::DriverVersion* installed_version = nullptr;
using patch::DriverVersion;
void* g_orig_buffer_init = nullptr;

void CopyHostDescriptorEntry(uint8_t* copy, void* /* callback */, void* /* callback_data */)
{
	CopyHostDescriptor(*installed_version, copy);
}

void CopyHostBytesEntry(void* destination, const void* source, std::size_t bytes)
{
	CopySerial(Direction::Unknown, destination, source, bytes);
}

uint64_t ReadU64(const void* data, uint16_t offset)
{
	uint64_t value;
	std::memcpy(&value, static_cast<const uint8_t*>(data) + offset, sizeof(value));
	return value;
}

uint32_t ReadU32(const void* data, uint16_t offset)
{
	uint32_t value;
	std::memcpy(&value, static_cast<const uint8_t*>(data) + offset, sizeof(value));
	return value;
}

template <class T>
T* AsPointer(uint64_t v)
{
	return reinterpret_cast<T*>(static_cast<uintptr_t>(v));
}

struct HostOperand
{
	uint8_t* base = nullptr;
	uint64_t pitch = 0;
	uint64_t height = 0;
};

HostOperand HostOperandOf(const DriverVersion& vi, const uint8_t* op)
{
	HostOperand r;
	const uint32_t type = ReadU32(op, vi.op_kind);
	if (type == vi.op_kind_1)
	{
		const uint8_t* pointer1 = AsPointer<const uint8_t>(ReadU64(op, vi.op_some_pointer1));
		if (pointer1 == nullptr)
		{
			return r;
		}
		r.base = AsPointer<uint8_t>(ReadU64(pointer1, vi.op_some_pointer1_address)) +
				 ReadU64(op, vi.op_some_pointer1_off);
		r.pitch = ReadU64(op, vi.op_some_pointer1_width);
		r.height = ReadU64(op, vi.op_some_pointer1_height);
	}
	else if (type == vi.op_kind_2 && vi.op_some_pointer3 != 0)
	{
		r.base = AsPointer<uint8_t>(ReadU64(op, vi.op_some_pointer3));
		r.pitch = ReadU64(op, vi.op_some_pointer3_width);
		r.height = ReadU64(op, vi.op_some_pointer3_height);
	}
	else if (type == vi.op_kind_3)
	{
		r.base = AsPointer<uint8_t>(ReadU64(op, vi.op_some_pointer2));
		r.pitch = ReadU64(op, vi.op_some_pointer2_width);
		r.height = ReadU64(op, vi.op_some_pointer2_height);
	}
	if (r.base != nullptr)
	{
		r.base += (ReadU64(op, vi.op_unkz) * r.height + ReadU64(op, vi.op_unky)) * r.pitch +
				  ReadU64(op, vi.op_unkx);
	}
	return r;
}

using BufferInitFn = int (*)(void* ctx, void* buffer, uint32_t size, int type);

int InitializeStagingBufferEntry(void* ctx, void* buffer, uint32_t size, int type)
{
	const auto orig = reinterpret_cast<BufferInitFn>(g_orig_buffer_init);
	const int raw = GetConfig().staging_buffer_kib;
	const uint32_t kib = static_cast<uint32_t>(raw);
	if (kib > 0)
	{
		const uint32_t bytes = kib * 1024u;
		if (bytes != size)
		{
			LogDebug("staging buffer size {} -> {} (env {} KiB)", size, bytes, kib);
			size = bytes;
		}
	}
	return orig(ctx, buffer, size, type);
}

} // namespace

void CopyHostDescriptor(const DriverVersion& vi, uint8_t* copy)
{
	const HostOperand src = HostOperandOf(vi, copy);
	const HostOperand dst = HostOperandOf(vi, copy + vi.op_stride);
	if (src.base == nullptr || dst.base == nullptr)
	{
		return;
	}

	const uint64_t width = ReadU64(copy, vi.copying_width_bytes);
	const uint64_t height = ReadU64(copy, vi.copying_height_bytes);
	const uint64_t depth = ReadU64(copy, vi.copying_depth_bytes);

	const uint32_t dst_type = ReadU32(copy + vi.op_stride, vi.op_kind);
	const uint32_t src_type = ReadU32(copy, vi.op_kind);
	const bool staging_leg = dst_type == vi.op_kind_1 || src_type == vi.op_kind_1;
	if (staging_leg && GetConfig().bench_dma)
	{
		return;
	}
	const Direction dir = !staging_leg				 ? Direction::Unknown
						  : dst_type == vi.op_kind_1 ? Direction::ToStaging
													 : Direction::FromStaging;

	Copy2D(dir, Copy2DView{dst.base, dst.pitch, dst.height, src.base, src.pitch, src.height, width, height,
						   depth});
}

void InstallPatches(const DriverVersion* vi, uintptr_t base)
{
	const HookConfig& cfg = GetConfig();
	if (!cfg.WantH2HPatch() && !cfg.WantInternalMemcpyPatch() && !cfg.WantStagingSizeWrap())
	{
		return;
	}
	if (vi == nullptr)
	{
		LogLine("error: requested in-driver patches have no usable version row; API copies still use the "
				"driver/staging configuration");
		return;
	}
	const char* why = "";
	if (!patch::IsLibraryPatchable(base, &why))
	{
		LogLine("error: requested libcuda patches unavailable on {}: {}; API pipeline remains available",
				ArchName(kNativeArch), why);
		return;
	}
	installed_version = vi;
	if (cfg.WantH2HPatch())
	{
		if (vi->h2h_va == 0 ||
			!patch::VerifyBytes(base + vi->h2h_va, vi->h2h_prologue, sizeof(vi->h2h_prologue)))
		{
			LogLine("error: requested host-copy patch has no matching prologue; driver host-copy code "
					"remains unchanged");
		}
		else
		{
			const auto installed = patch::InstallFullReplacement(
				base + vi->h2h_va, reinterpret_cast<void*>(&CopyHostDescriptorEntry), "h2h_engine");
			if (!installed)
			{
				LogLine("error: {} (code={})", installed.error().Message(), installed.error().GetCode());
				if (installed.error().GetKind() == Error::Kind::Os)
				{
					return;
				}
			}
		}
	}
	if (cfg.WantInternalMemcpyPatch())
	{
		if (vi->internal_memcpy_va == 0 ||
			!patch::VerifyBytes(base + vi->internal_memcpy_va, vi->internal_memcpy_prologue,
								sizeof(vi->internal_memcpy_prologue)))
		{
			LogLine(
				"error: requested host-byte-copy patch has no matching prologue; driver internal memcpy code "
				"remains unchanged");
		}
		else
		{
			const auto installed = patch::InstallFullReplacement(base + vi->internal_memcpy_va,
																 reinterpret_cast<void*>(&CopyHostBytesEntry),
																 "internal_memcpy_temporal");
			if (!installed)
			{
				LogLine("error: {} (code={})", installed.error().Message(), installed.error().GetCode());
				if (installed.error().GetKind() == Error::Kind::Os)
				{
					return;
				}
			}
		}
	}
	if (cfg.WantStagingSizeWrap())
	{
		if (vi->buffer_init_va == 0 ||
			!patch::VerifyBytes(base + vi->buffer_init_va, vi->buffer_init_prologue,
								sizeof(vi->buffer_init_prologue)))
		{
			LogLine("error: requested staging-size patch has no matching prologue; driver staging size "
					"remains unchanged");
		}
		else
		{
			const auto installed = patch::InstallWrapper(
				base + vi->buffer_init_va, reinterpret_cast<void*>(&InitializeStagingBufferEntry),
				&g_orig_buffer_init, "staging_buffer_init");
			if (!installed)
			{
				LogLine("error: {} (code={})", installed.error().Message(), installed.error().GetCode());
				if (installed.error().GetKind() == Error::Kind::Os)
				{
					return;
				}
			}
		}
	}
}

} // namespace driver
} // namespace cmt

namespace cmt {

Result<HookConfig> ParseConfig(const EnvLookup& getenv_fn)
{
	return ParseConfig(getenv_fn, hwcopy::GetCpu());
}

const HookConfig& GetConfig()
{
	static const HookConfig* cfg = []
	{
		auto parsed = ParseConfig([](const char* name) { return secure_getenv(name); });
		if (!parsed)
		{
			LogLine("error: invalid hook configuration: {} (code={}); copy optimization disabled",
					parsed.error().Message(), parsed.error().GetCode());
			return new HookConfig{};
		}
		if (parsed->bench_dma)
		{
			LogLine("WARNING: STAGING_BENCH_DMA=1: staged copies skip their CPU legs and DO NOT COPY THE DATA; "
					"the application receives wrong bytes. A bandwidth aid for benchmarks only.");
		}
		return new HookConfig(std::move(*parsed));
	}();
	return *cfg;
}

} // namespace cmt

#endif

#if !defined(CMT_HOOK_CORE_ONLY)

namespace cmt {
namespace ring {
using namespace cmt::cu;

namespace {
enum class Origin : std::uint8_t
{
	Typed,
	Generic
};

const cmt::Row* TableRow(cmt::Dir direction, std::size_t bytes);

struct PointerFacts
{
	unsigned memory_type = 0;
	unsigned managed = 0;
	int ordinal = -1;
	CUdeviceptr range_start = 0;
	std::size_t range_size = 0;
	CUdeviceptr mapping_base = 0;
	std::size_t mapping_size = 0;
};

bool QueryPointer(CUdeviceptr address, PointerFacts* facts)
{
	if (g_api.cuPointerGetAttributes == nullptr)
	{
		return false;
	}
	unsigned attributes[] = {kPointerAttributeMemoryType,	 kPointerAttributeIsManaged,
							 kPointerAttributeDeviceOrdinal, kPointerAttributeRangeStartAddr,
							 kPointerAttributeRangeSize,	 kPointerAttributeMappingBaseAddr,
							 kPointerAttributeMappingSize};
	void* values[] = {&facts->memory_type, &facts->managed,		 &facts->ordinal,	  &facts->range_start,
					  &facts->range_size,  &facts->mapping_base, &facts->mapping_size};
	static_assert(std::size(attributes) == std::size(values));
	const CUresult rc = g_api.cuPointerGetAttributes(static_cast<unsigned>(std::size(attributes)), attributes,
													 values, address);
	if (rc != kSuccess)
	{
		static std::atomic<bool> said{false};
		if (!said.exchange(true, std::memory_order_relaxed))
		{
			LogDebug(
				"cuPointerGetAttributes rejected the query for {:#x} (rc={}); such copies go to the driver. "
				"Reported once.",
				address, rc);
		}
	}
	return rc == kSuccess;
}

bool IsPageable(const PointerFacts& facts)
{
	return facts.memory_type == 0 && facts.managed == 0 && facts.range_size == 0;
}

bool IsDeviceMemoryOf(const PointerFacts& facts, CUdevice device)
{
	return facts.memory_type == kMemoryTypeDevice && facts.managed == 0 && facts.ordinal == device;
}

constexpr std::size_t kMaxFollowingMappings = 64;

bool IsDeviceRange(const PointerFacts& first, CUdeviceptr address, std::size_t bytes, CUdevice device)
{
	constexpr CUdeviceptr kMax = std::numeric_limits<CUdeviceptr>::max();
	if (!IsDeviceMemoryOf(first, device) || address < first.range_start)
	{
		return false;
	}
	const CUdeviceptr offset = address - first.range_start;
	if (offset >= first.range_size || bytes > first.range_size - offset || first.mapping_size == 0 ||
		address < first.mapping_base || first.mapping_size > kMax - first.mapping_base)
	{
		return false;
	}
	const CUdeviceptr end = address + bytes;
	CUdeviceptr covered = first.mapping_base + first.mapping_size;
	for (std::size_t walked = 0; covered < end; walked++)
	{
		PointerFacts next;
		if (walked == kMaxFollowingMappings || !QueryPointer(covered, &next) ||
			!IsDeviceMemoryOf(next, device) || next.mapping_size == 0 || next.mapping_size > kMax - covered ||
			next.mapping_base != covered)
		{
			return false;
		}
		covered += next.mapping_size;
	}
	return true;
}

bool MayLeaveDriver(std::size_t bytes)
{
	const HookConfig& config = GetConfig();
	if (config.fixed_benchmark || config.progressive_pinning == ProgressivePinning::On ||
		EnvironmentStagingEnabled())
	{
		return true;
	}
	if (!config.tune_table)
	{
		return false;
	}
	for (const cmt::Dir direction : {cmt::Dir::HtoD, cmt::Dir::DtoH})
	{
		const cmt::Row* row = config.tune_table->Find(direction, bytes);
		if (row != nullptr && row->mode != cmt::Mode::Driver)
		{
			return true;
		}
	}
	return false;
}

enum class Ends : std::uint8_t
{
	Usable,
	Refused,
	Foreign
};

bool IsHole(const PointerFacts& facts)
{
	return facts.memory_type == 0 && facts.managed == 0 && facts.range_size != 0;
}

Ends TypedEnds(const CopyRequest& request, bool* pageable)
{
	CUdevice device = -1;
	PointerFacts host_end;
	PointerFacts device_end;
	*pageable = false;
	if (request.device == 0 || request.host == nullptr ||
		!QueryPointer(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(request.host)), &host_end) ||
		!QueryPointer(request.device, &device_end))
	{
		return Ends::Foreign;
	}
	if (g_api.cuCtxGetDevice == nullptr || g_api.cuCtxGetDevice(&device) != kSuccess)
	{
		device = device_end.ordinal;
	}
	*pageable = IsPageable(host_end);
	if (IsHole(host_end) || IsHole(device_end))
	{
		return Ends::Refused;
	}
	if ((!*pageable && host_end.memory_type != kMemoryTypeHost) ||
		(!IsPageable(device_end) && !IsDeviceMemoryOf(device_end, device)))
	{
		return Ends::Foreign;
	}
	return IsPageable(device_end) || IsDeviceRange(device_end, request.device, request.bytes, device)
			   ? Ends::Usable
			   : Ends::Refused;
}

CUresult CallGeneric(CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream, bool async)
{
	if (async)
	{
		return g_api.cuMemcpyAsync != nullptr ? g_api.cuMemcpyAsync(dst, src, bytes, stream) : kErrorNotFound;
	}
	return g_api.cuMemcpy != nullptr ? g_api.cuMemcpy(dst, src, bytes) : kErrorNotFound;
}

CUresult CopyOriginal(const CopyRequest& request, bool asynchronous, Origin origin)
{
	if (origin == Origin::Generic)
	{
		const auto host = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(request.host));
		const bool to_device = request.direction == cmt::Dir::HtoD;
		return CallGeneric(to_device ? request.device : host, to_device ? host : request.device,
						   request.bytes, request.stream, asynchronous);
	}
	return cmt::copying::CopyDirect(cmt::copying::DirectRequest<cmt::gpu::nv::CopyApi>{
		request.direction, request.device, request.host, request.bytes, request.stream, asynchronous});
}

CUresult CopyThroughHook(const CopyRequest& request, bool async, Origin origin)
{
	const bool to_device = request.direction == cmt::Dir::HtoD;
	if (ForkedChild() || CopiesDisabled())
	{
		tune_audit::Reason(tune_audit::Outcome::Fork);
		return CopyOriginal(request, async, origin);
	}
	NoteUsed();
	if (!pinning::Enabled() && !pinning::Active() && !GetConfig().tune_table && !EnvironmentStagingEnabled())
	{
		tune_audit::Reason(tune_audit::Outcome::Driver);
		return CopyOriginal(request, async, origin);
	}
	pinning::CopyUse copy_use(request.host, request.bytes);
	if (IsCapturing(request.stream))
	{
		tune_audit::Reason(tune_audit::Outcome::Capture);
		if (copy_use.HasRegistrationOverlap())
		{
			return kErrorStreamCaptureUnsupported;
		}
		return CopyOriginal(request, async, origin);
	}
	bool host_pageable = origin == Origin::Generic;
	bool obligations_only = false;
	if (origin == Origin::Typed && request.bytes != 0 && (MayLeaveDriver(request.bytes) || pinning::Active()))
	{
		const Ends ends = TypedEnds(request, &host_pageable);
		if (ends == Ends::Refused || (ends == Ends::Foreign && !copy_use.HasRegistrationOverlap()))
		{
			tune_audit::Reason(tune_audit::Outcome::Driver);
			return CopyOriginal(request, async, origin);
		}
		obligations_only = ends == Ends::Foreign;
		host_pageable = host_pageable && !obligations_only;
	}
	const bool machine_matches = MatchesTuningMachine();
	const cmt::Row* row = machine_matches ? TableRow(request.direction, request.bytes) : nullptr;
	if (row != nullptr)
	{
		tune_audit::Row(*row);
	}
	const auto& config = GetConfig();
	std::optional<PinningSettings> manual;
	const PinningSettings* selected_pinning = nullptr;
	if (config.fixed_benchmark)
	{
		selected_pinning = config.benchmark_pinning ? &*config.benchmark_pinning : nullptr;
	}
	else if (config.progressive_pinning == ProgressivePinning::On)
	{
		manual = pinning::ManualSettings();
		selected_pinning = &*manual;
	}
	else if (config.progressive_pinning == ProgressivePinning::Auto && row != nullptr &&
			 row->mode == Mode::ProgressivePinning)
	{
		selected_pinning = &row->pinning;
	}
	if (obligations_only)
	{
		selected_pinning = nullptr;
	}
	CUresult pinning_result = kSuccess;
	if (pinning::TryCopy(to_device ? pinning::Dir::H2D : pinning::Dir::D2H, request.device, request.host,
						 request.bytes, request.stream, &pinning_result, selected_pinning,
						 tune_audit::PinningObservation()))
	{
		tune_audit::Reason(tune_audit::Outcome::ProgressivePinning);
		return pinning_result;
	}
	if (!machine_matches)
	{
		tune_audit::Reason(tune_audit::Outcome::Driver);
		return CopyOriginal(request, async, origin);
	}
	if (row != nullptr)
	{
		if (row->mode == cmt::Mode::Driver || request.bytes == 0 || !host_pageable)
		{
			tune_audit::Reason(row->mode == cmt::Mode::Driver ? tune_audit::Outcome::DriverRow
															  : tune_audit::Outcome::NonPageable);
			return CopyOriginal(request, async, origin);
		}
		if (row->mode == Mode::ProgressivePinning)
		{
			tune_audit::Reason(tune_audit::Outcome::NotStarted);
			return CopyOriginal(request, async, origin);
		}
		const CUresult result = CopyMeasured(request, *row);
		return result != kNotStarted ? result : CopyOriginal(request, async, origin);
	}
	if (EnvironmentStagingEnabled() && request.bytes != 0 && host_pageable)
	{
		if (async && to_device)
		{
			LogDebug("pipeline h2d async: {} bytes", request.bytes);
		}
		tune_audit::Otherwise(tune_audit::Outcome::Env);
		const CUresult result = CopyEnvironment(request);
		return result != kNotStarted ? result : CopyOriginal(request, async, origin);
	}
	tune_audit::Otherwise(tune_audit::Outcome::Driver);
	return CopyOriginal(request, async, origin);
}
} // namespace

CUresult CopyHtoD(CUdeviceptr destination, const void* source, std::size_t bytes, CUstream stream, bool async)
{
	return CopyThroughHook({cmt::Dir::HtoD, destination,
							const_cast<uint8_t*>(static_cast<const uint8_t*>(source)), bytes, stream},
						   async, Origin::Typed);
}

CUresult CopyDtoH(void* destination, CUdeviceptr source, std::size_t bytes, CUstream stream, bool async)
{
	return CopyThroughHook({cmt::Dir::DtoH, source, static_cast<uint8_t*>(destination), bytes, stream}, async,
						   Origin::Typed);
}
} // namespace ring
} // namespace cmt

namespace cmt {
namespace ring {

using namespace cmt::cu;

namespace {

const cmt::Row* TableRow(cmt::Dir dir, std::size_t bytes)
{
	const std::shared_ptr<const cmt::Table>& table = GetConfig().tune_table;
	return table != nullptr ? table->Find(dir, bytes) : nullptr;
}

} // namespace

bool EnvironmentStagingEnabled()
{
	const auto& config = GetConfig();
	return !config.tune_table && config.num_buffers >= 1;
}

namespace {

bool IsPageableRange(const PointerFacts& first, CUdeviceptr address, std::size_t bytes)
{
	if (!IsPageable(first) || bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address)
	{
		return false;
	}
	PointerFacts last;
	return QueryPointer(address + (bytes - 1), &last) && IsPageable(last);
}

std::optional<cmt::Dir> RouteGeneric(CUdeviceptr dst, CUdeviceptr src, std::size_t bytes)
{
	CUdevice device = -1;
	if (dst == 0 || src == 0 || g_api.cuCtxGetDevice == nullptr || g_api.cuCtxGetDevice(&device) != kSuccess)
	{
		return std::nullopt;
	}
	PointerFacts to;
	PointerFacts from;
	if (!QueryPointer(dst, &to) || !QueryPointer(src, &from))
	{
		return std::nullopt;
	}
	if (IsDeviceRange(to, dst, bytes, device) && IsPageableRange(from, src, bytes))
	{
		return cmt::Dir::HtoD;
	}
	if (IsDeviceRange(from, src, bytes, device) && IsPageableRange(to, dst, bytes))
	{
		return cmt::Dir::DtoH;
	}
	return std::nullopt;
}

CUresult CopyGeneric(CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream, bool async)
{
	const std::optional<cmt::Dir> route =
		bytes == 0 || !MayLeaveDriver(bytes) ? std::nullopt : RouteGeneric(dst, src, bytes);
	if (!route)
	{
		return CallGeneric(dst, src, bytes, stream, async);
	}
	const cmt::Dir direction = *route;
	const bool to_device = direction == cmt::Dir::HtoD;
	const CUdeviceptr device = to_device ? dst : src;
	auto* host = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(to_device ? src : dst));
	tune_audit::Call observed(direction, bytes, async);
	return observed.Finish(
		CopyThroughHook({direction, device, host, bytes, async ? stream : nullptr}, async, Origin::Generic));
}

} // namespace

} // namespace ring
} // namespace cmt

namespace cmt {
namespace ring {

using namespace cmt::cu;

namespace {

template <class Fn>
Fn RealNow(const char* name)
{
	return reinterpret_cast<Fn>(CudaSymbol(name));
}

template <class Fn>
Fn Original(Fn DriverApi::* member, const char* name)
{
	return DriverApiResolved() ? g_api.*member : RealNow<Fn>(name);
}

CUresult MemcpyHtoDEntry(CUdeviceptr dst, const void* src, std::size_t bytes) noexcept
{
	tune_audit::Call observed(cmt::Dir::HtoD, bytes, false);
	if (!DriverApiReady())
	{
		tune_audit::Reason(tune_audit::Outcome::NotReady);
		auto fn = Original(&DriverApi::cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");
		return observed.Finish(fn != nullptr ? fn(dst, src, bytes) : kErrorNotFound);
	}
	return observed.Finish(CopyHtoD(dst, src, bytes, nullptr, false));
}

CUresult MemcpyDtoHEntry(void* dst, CUdeviceptr src, std::size_t bytes) noexcept
{
	tune_audit::Call observed(cmt::Dir::DtoH, bytes, false);
	if (!DriverApiReady())
	{
		tune_audit::Reason(tune_audit::Outcome::NotReady);
		auto fn = Original(&DriverApi::cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");
		return observed.Finish(fn != nullptr ? fn(dst, src, bytes) : kErrorNotFound);
	}
	return observed.Finish(CopyDtoH(dst, src, bytes, nullptr, false));
}

CUresult MemcpyHtoDAsyncEntry(CUdeviceptr dst, const void* src, std::size_t bytes, CUstream stream) noexcept
{
	tune_audit::Call observed(cmt::Dir::HtoD, bytes, true);
	if (!DriverApiReady())
	{
		tune_audit::Reason(tune_audit::Outcome::NotReady);
		auto fn = Original(&DriverApi::cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync_v2");
		return observed.Finish(fn != nullptr ? fn(dst, src, bytes, stream) : kErrorNotFound);
	}
	return observed.Finish(CopyHtoD(dst, src, bytes, stream, true));
}

CUresult MemcpyDtoHAsyncEntry(void* dst, CUdeviceptr src, std::size_t bytes, CUstream stream) noexcept
{
	tune_audit::Call observed(cmt::Dir::DtoH, bytes, true);
	if (!DriverApiReady())
	{
		tune_audit::Reason(tune_audit::Outcome::NotReady);
		auto fn = Original(&DriverApi::cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync_v2");
		return observed.Finish(fn != nullptr ? fn(dst, src, bytes, stream) : kErrorNotFound);
	}
	return observed.Finish(CopyDtoH(dst, src, bytes, stream, true));
}

CUresult MemcpyEntry(CUdeviceptr dst, CUdeviceptr src, std::size_t bytes) noexcept
{
	if (!DriverApiReady())
	{
		auto fn = Original(&DriverApi::cuMemcpy, "cuMemcpy");
		return fn != nullptr ? fn(dst, src, bytes) : kErrorNotFound;
	}
	return CopyGeneric(dst, src, bytes, nullptr, false);
}

CUresult MemcpyAsyncEntry(CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream) noexcept
{
	if (!DriverApiReady())
	{
		auto fn = Original(&DriverApi::cuMemcpyAsync, "cuMemcpyAsync");
		return fn != nullptr ? fn(dst, src, bytes, stream) : kErrorNotFound;
	}
	return CopyGeneric(dst, src, bytes, stream, true);
}

using ContextFn = CUresult (*)(CUcontext);

CUresult CallContextDestroy(ContextFn real, CUcontext ctx) noexcept
{
	if (DriverApiReady() && !ForkedChild())
	{
		NoteUsed();
		ResetContext(ctx);
	}
	const CUresult rc = real != nullptr ? real(ctx) : kErrorNotFound;
	host_alloc::AfterContextTeardown();
	return rc;
}

CUresult CtxDestroyV2Entry(CUcontext ctx) noexcept
{
	const auto real = Original(&DriverApi::cuCtxDestroy_v2, "cuCtxDestroy_v2");
	return CallContextDestroy(real, ctx);
}

CUresult CtxDestroyEntry(CUcontext ctx) noexcept
{
	const auto real = Original(&DriverApi::cuCtxDestroy, "cuCtxDestroy");
	return CallContextDestroy(real, ctx);
}

CUresult CallPrimaryContext(CUresult (*real)(CUdevice), CUdevice dev) noexcept
{
	if (DriverApiReady() && !ForkedChild())
	{
		NoteUsed();
		ResetDevice(dev);
	}
	const CUresult rc = real != nullptr ? real(dev) : kErrorNotFound;
	host_alloc::AfterContextTeardown();
	return rc;
}

using PrimaryFn = CUresult (*)(CUdevice);

PrimaryFn OriginalPrimary(PrimaryFn DriverApi::* member, const char* name)
{
	return Original(member, name);
}

CUresult DevicePrimaryCtxResetEntry(CUdevice dev) noexcept
{
	return CallPrimaryContext(OriginalPrimary(&DriverApi::cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset"),
							  dev);
}

CUresult DevicePrimaryCtxResetV2Entry(CUdevice dev) noexcept
{
	return CallPrimaryContext(
		OriginalPrimary(&DriverApi::cuDevicePrimaryCtxReset_v2, "cuDevicePrimaryCtxReset_v2"), dev);
}

CUresult DevicePrimaryCtxReleaseEntry(CUdevice dev) noexcept
{
	return CallPrimaryContext(
		OriginalPrimary(&DriverApi::cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease"), dev);
}

CUresult DevicePrimaryCtxReleaseV2Entry(CUdevice dev) noexcept
{
	return CallPrimaryContext(
		OriginalPrimary(&DriverApi::cuDevicePrimaryCtxRelease_v2, "cuDevicePrimaryCtxRelease_v2"), dev);
}

constexpr std::size_t kUnknownPrimarySlots = 8;
std::atomic<PrimaryFn> g_unknown_primary[kUnknownPrimarySlots] = {};

template <std::size_t K>
CUresult UnknownPrimary(CUdevice dev) noexcept
{
	return CallPrimaryContext(g_unknown_primary[K].load(std::memory_order_acquire), dev);
}

template <std::size_t... K>
constexpr std::array<PrimaryFn, sizeof...(K)> UnknownPrimaryTable(std::index_sequence<K...>)
{
	return {&UnknownPrimary<K>...};
}

constexpr std::size_t kUnknownContextSlots = 8;
std::atomic<ContextFn> g_unknown_context[kUnknownContextSlots] = {};

template <std::size_t K>
CUresult UnknownContext(CUcontext ctx) noexcept
{
	return CallContextDestroy(g_unknown_context[K].load(std::memory_order_acquire), ctx);
}

template <std::size_t... K>
constexpr std::array<ContextFn, sizeof...(K)> UnknownContextTable(std::index_sequence<K...>)
{
	return {&UnknownContext<K>...};
}

constexpr char kCopyExportPrefix[] = "cuMemcpy";
using AnyFn = void (*)();

struct Redirect
{
	const char* symbol;
	AnyFn target;
};

const Redirect* Redirects(std::size_t* count)
{
	static const Redirect table[] = {
		{"cuMemcpyHtoD_v2", reinterpret_cast<AnyFn>(&MemcpyHtoDEntry)},
		{"cuMemcpyDtoH_v2", reinterpret_cast<AnyFn>(&MemcpyDtoHEntry)},
		{"cuMemcpyHtoDAsync_v2", reinterpret_cast<AnyFn>(&MemcpyHtoDAsyncEntry)},
		{"cuMemcpyDtoHAsync_v2", reinterpret_cast<AnyFn>(&MemcpyDtoHAsyncEntry)},
		{"cuMemcpy", reinterpret_cast<AnyFn>(&MemcpyEntry)},
		{"cuMemcpyAsync", reinterpret_cast<AnyFn>(&MemcpyAsyncEntry)},
		{"cuCtxDestroy_v2", reinterpret_cast<AnyFn>(&CtxDestroyV2Entry)},
		{"cuCtxDestroy", reinterpret_cast<AnyFn>(&CtxDestroyEntry)},
	};
	*count = sizeof(table) / sizeof(table[0]);
	return table;
}

} // namespace

void* RedirectLookup(const char* symbol, bool per_thread_stream)
{
	std::size_t count = 0;
	const Redirect* table = Redirects(&count);
	for (std::size_t i = 0; i < count; i++)
	{
		if (std::strcmp(symbol, table[i].symbol) == 0)
		{
			if (per_thread_stream &&
				std::strncmp(table[i].symbol, kCopyExportPrefix, sizeof(kCopyExportPrefix) - 1) == 0)
			{
				return nullptr;
			}
			return reinterpret_cast<void*>(table[i].target);
		}
	}
	return nullptr;
}

void* RedirectForAddress(void* real)
{
	if (real == nullptr)
	{
		return nullptr;
	}
	const bool ready = DriverApiResolved();

	struct Pair
	{
		PrimaryFn DriverApi::* member;
		const char* name;
		PrimaryFn wrapper;
	};

	static const Pair pairs[] = {
		{&DriverApi::cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset", &DevicePrimaryCtxResetEntry},
		{&DriverApi::cuDevicePrimaryCtxReset_v2, "cuDevicePrimaryCtxReset_v2", &DevicePrimaryCtxResetV2Entry},
		{&DriverApi::cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease", &DevicePrimaryCtxReleaseEntry},
		{&DriverApi::cuDevicePrimaryCtxRelease_v2, "cuDevicePrimaryCtxRelease_v2",
		 &DevicePrimaryCtxReleaseV2Entry},
	};
	for (const Pair& p : pairs)
	{
		const PrimaryFn fn = ready ? g_api.*(p.member) : RealNow<PrimaryFn>(p.name);
		if (fn != nullptr && real == reinterpret_cast<void*>(fn))
		{
			return reinterpret_cast<void*>(p.wrapper);
		}
	}
	return nullptr;
}

void* WrapPrimaryByName(void* real)
{
	static constexpr auto wrappers = UnknownPrimaryTable(std::make_index_sequence<kUnknownPrimarySlots>());
	const auto fn = reinterpret_cast<PrimaryFn>(real);
	for (std::size_t k = 0; k < kUnknownPrimarySlots; k++)
	{
		PrimaryFn expected = nullptr;
		if (g_unknown_primary[k].compare_exchange_strong(expected, fn, std::memory_order_acq_rel) ||
			expected == fn)
		{
			return reinterpret_cast<void*>(wrappers[k]);
		}
	}
	return nullptr;
}

void* WrapContextByName(void* real)
{
	static constexpr auto wrappers = UnknownContextTable(std::make_index_sequence<kUnknownContextSlots>());
	const auto fn = reinterpret_cast<ContextFn>(real);
	for (std::size_t k = 0; k < kUnknownContextSlots; ++k)
	{
		ContextFn expected = nullptr;
		if (g_unknown_context[k].compare_exchange_strong(expected, fn, std::memory_order_acq_rel) ||
			expected == fn)
		{
			return reinterpret_cast<void*>(wrappers[k]);
		}
	}
	if (!DisableCopiesForUnwrappedTeardown("context-destroy"))
	{
		Panic("context-destroy wrapper capacity exhausted; cannot preserve context teardown");
	}
	return nullptr;
}

bool DisableCopiesForUnwrappedTeardown(const char* kind)
{
	if (pinning::Active())
	{
		return false;
	}
	DisableCopies();
	static std::atomic<bool> said{false};
	if (!said.exchange(true, std::memory_order_relaxed))
	{
		LogLine("error: {} wrapper capacity exhausted; an unwrapped driver teardown function is handed out, so "
				"the hook stops staging: from now on every copy uses the original driver",
				kind);
	}
	return true;
}

} // namespace ring
} // namespace cmt

#endif
