#include "copying/cpu_copy.h"
#include "common/log.h"

namespace cmt {
namespace {

CopyLoop ResolveConfiguredLoop(const char* setting, const std::string& name, const HookConfig& config,
							   CopyCtx context)
{
	const auto selected = hwcopy::Select(name, context, config.copy_isa, hwcopy::GetCpu());
	if (!selected || selected->fell_back)
	{
		Panic("validated copy loop cannot execute");
	}
	LogDebug("{}={} runs {}", setting, name, hwcopy::Describe(selected->loop));
	return selected->loop;
}

std::optional<CopyLoop> ResolveOverride(const char* setting, const std::optional<std::string>& name,
										const HookConfig& config, CopyCtx context)
{
	if (!name)
	{
		return std::nullopt;
	}
	return ResolveConfiguredLoop(setting, *name, config, context);
}

const HostCopyLoops& ConfiguredCopyLoops()
{
	static const HostCopyLoops loops = SelectCopyLoops(GetConfig());
	return loops;
}

} // namespace

HostCopyLoops SelectCopyLoops(const HookConfig& cfg)
{
	CopyCtx ctx;
	ctx.flush_batch = cfg.copy_flush_bytes;
	HostCopyLoops loops{hwcopy::DefaultLoop(ctx, cfg.copy_isa, hwcopy::GetCpu()), std::nullopt, std::nullopt};
	loops.h2d = ResolveOverride("STAGING_COPY_NT_IMPL_H2D", cfg.impl_h2d, cfg, ctx);
	loops.d2h = ResolveOverride("STAGING_COPY_NT_IMPL_D2H", cfg.impl_d2h, cfg, ctx);
	if (cfg.impl)
	{
		loops.global = ResolveConfiguredLoop("STAGING_COPY_NT_IMPL", *cfg.impl, cfg, ctx);
	}
	return loops;
}

CopyLoop CopyLoopFor(Direction dir)
{
	const HostCopyLoops& loops = ConfiguredCopyLoops();
	if (dir == Direction::ToStaging && loops.h2d)
	{
		return *loops.h2d;
	}
	if (dir == Direction::FromStaging && loops.d2h)
	{
		return *loops.d2h;
	}
	return loops.global;
}

void CopySerial(Direction dir, void* dst, const void* src, std::size_t bytes)
{
	hwcopy::Run(CopyLoopFor(dir), dst, src, bytes);
}

void Copy2D(Direction dir, const Copy2DView& copy)
{
	Run2DSerial(CopyLoopFor(dir), copy);
	StoreFence();
}

} // namespace cmt
