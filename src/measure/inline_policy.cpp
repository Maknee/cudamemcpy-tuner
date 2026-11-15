#include "measure/inline_policy.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

#include "arch/disasm/plan.h"
#include "common/error.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "measure/metrics.h"
#include "cuda/versions.h"

namespace cmt {
namespace bench {
namespace {
bool enabled = false;
void* driver = nullptr;

enum class Policy : std::uint8_t
{
	Stock,
	Inline,
	Dma
};

struct Range
{
	uintptr_t base;
	uintptr_t gate;
	bool found = false;
};

int FindGate(dl_phdr_info* object, std::size_t, void* opaque)
{
	auto& range = *static_cast<Range*>(opaque);
	if (object->dlpi_addr != range.base)
	{
		return 0;
	}
	for (unsigned i = 0; i < object->dlpi_phnum; i++)
	{
		const auto& segment = object->dlpi_phdr[i];
		if (segment.p_type != PT_LOAD || (segment.p_flags & (PF_R | PF_W | PF_X)) != (PF_R | PF_X))
		{
			continue;
		}
		const uintptr_t start = object->dlpi_addr + segment.p_vaddr;
		if (range.gate >= start && segment.p_filesz >= 33 && range.gate - start <= segment.p_filesz - 33)
		{
			range.found = true;
		}
	}
	return 1;
}

std::string HexBytes(const uint8_t* bytes, std::size_t count)
{
	constexpr char hex[] = "0123456789abcdef";
	std::string out;
	for (std::size_t i = 0; i < count; i++)
	{
		out += hex[bytes[i] >> 4];
		out += hex[bytes[i] & 15];
	}
	return out;
}
} // namespace

cmt::Result<std::map<std::string, std::string>> ConfigureInlinePolicy(const cmt::Config& cfg)
{
	const char* requested = std::getenv("CMT_INLINE_POLICY");
	if (!requested)
	{

		return {};
	}
	Policy policy;
	if (std::strcmp(requested, "stock") == 0)
	{
		policy = Policy::Stock;
	}
	else if (std::strcmp(requested, "inline") == 0)
	{
		policy = Policy::Inline;
	}
	else if (std::strcmp(requested, "dma") == 0)
	{
		policy = Policy::Dma;
	}
	else
	{
		return cmt::UsageError("CMT_INLINE_POLICY must be stock, inline or dma");
	}
	if (cmt::kNativeArch != cmt::Arch::X86_64)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported, "inline threshold experiment requires x86-64");
	}
	if (cfg.path->name != "pageable" && cfg.path->name != "pinned_ce")
	{
		return cmt::UsageError("inline threshold experiment requires --path pageable or pinned_ce");
	}
	if (!cfg.verify)
	{
		return cmt::UsageError("inline threshold experiment requires correctness verification");
	}
	if ((std::getenv("LD_PRELOAD") && *std::getenv("LD_PRELOAD")) || std::getenv("CMT_TUNE_AUDIT"))
	{
		return cmt::UsageError("inline threshold experiment requires no preload or tuning settings");
	}
	if (enabled)
	{
		return cmt::UsageError("inline threshold policy may be installed only once per process");
	}
	const bool pageable = cfg.path->name == "pageable";
	const uint64_t paged = policy == Policy::Inline && pageable ? INT_MAX : 0;
	const uint64_t pinned = policy == Policy::Inline && !pageable ? UINT64_MAX : 0;
	driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!driver)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported, "inline experiment cannot load libcuda.so.1");
	}
	void* init = dlsym(driver, "cuInit");
	void* get_context = dlsym(driver, "cuCtxGetCurrent");
	Dl_info library{};
	if (!init || !get_context || dladdr(init, &library) == 0)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported, "inline experiment cannot identify libcuda");
	}
	int (*current_context)(void**) = nullptr;
	static_assert(sizeof(current_context) == sizeof(get_context));
	std::memcpy(&current_context, &get_context, sizeof(current_context));
	void* context = nullptr;
	if (current_context(&context) != 3)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported,
							"inline experiment must run before cuInit; refusing a live driver patch");
	}
	std::error_code path_error;
	const auto path = std::filesystem::canonical(library.dli_fname, path_error);
	if (path_error)
	{
		return cmt::Failure(cmt::Error::Kind::Os, "inline experiment cannot resolve libcuda",
							path_error.value());
	}
	const auto filename = path.filename().string();
	const std::string prefix = "libcuda.so.";
	if (filename.rfind(prefix, 0) != 0)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported, "unrecognized libcuda filename");
	}
	const auto version = filename.substr(prefix.size());
	const auto* row = cmt::patch::FindVersion(cmt::patch::NativeVersionTable(), version.c_str());
	if (!row || row->inline_paged_check.offset == 0)
	{
		return cmt::Failure(cmt::Error::Kind::Unsupported, "no scanned inline gate for this libcuda version");
	}
	const auto base = reinterpret_cast<uintptr_t>(library.dli_fbase);
	if (base > UINTPTR_MAX - row->inline_paged_check.offset)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "inline gate address overflows");
	}
	Range range{base, base + row->inline_paged_check.offset};
	dl_iterate_phdr(FindGate, &range);
	if (!range.found)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "inline gate lies outside executable libcuda bytes");
	}
	const auto* current = reinterpret_cast<const uint8_t*>(range.gate);
	const auto original_gate = HexBytes(current, cmt::patch::kInlineGateBytes);
	const auto plan = cmt::patch::PlanInlineThresholds(cmt::kNativeArch, *row, current, 33, paged, pinned);
	if (!plan.ok)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "inline experiment patch plan is invalid");
	}
	if (policy != Policy::Stock)
	{
		const long system_page_size = sysconf(_SC_PAGESIZE);
		if (system_page_size <= 0 || (system_page_size & (system_page_size - 1)) != 0)
		{
			return cmt::OsError("inline experiment: invalid system page size");
		}
		const uintptr_t page_size = static_cast<uintptr_t>(system_page_size);
		const uintptr_t start = range.gate & ~(page_size - 1);
		const std::size_t length = range.gate - start + 33;
		auto* page = reinterpret_cast<void*>(start);
		if (mprotect(page, length, PROT_READ | PROT_WRITE) != 0)
		{
			return cmt::OsError("inline experiment: make gate writable");
		}
		auto* gate = reinterpret_cast<uint8_t*>(range.gate);
		std::memcpy(gate + (plan.paged_limit_va - row->inline_paged_check.offset), plan.paged_limit,
					sizeof(plan.paged_limit));
		std::memcpy(gate + (plan.pinned_fetch_va - row->inline_paged_check.offset), plan.pinned_fetch,
					sizeof(plan.pinned_fetch));
		__builtin___clear_cache(reinterpret_cast<char*>(gate), reinterpret_cast<char*>(gate + 33));
		if (mprotect(page, length, PROT_READ | PROT_EXEC) != 0)
		{
			return cmt::OsError("inline experiment: restore executable protection");
		}
	}
	enabled = true;
	return std::map<std::string, std::string>{
		{"inline_experiment_schema", "1"},
		{"inline_requested_policy", requested},
		{"inline_driver_version", version},
		{"inline_driver_path", path.string()},
		{"inline_gate_offset", std::to_string(row->inline_paged_check.offset)},
		{"inline_original_gate", original_gate},
		{"inline_effective_gate", HexBytes(current, 33)},
		{"inline_paged_limit", policy == Policy::Stock ? "driver" : std::to_string(paged)},
		{"inline_pinned_limit", policy == Policy::Stock ? "driver" : std::to_string(pinned)},
		{"inline_path_verified", "false"}};
}
} // namespace bench
} // namespace cmt
