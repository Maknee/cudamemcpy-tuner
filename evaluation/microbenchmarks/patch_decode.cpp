#include <benchmark/benchmark.h>
#include <cstdint>
#include "arch/disasm/plan.h"
#include "common/error.h"
#include "cuda/versions.h"

// Plan a trampoline for fixed prologues. Includes Capstone session setup and
// relocation, but never writes executable memory or installs a live patch.

namespace {
bool failed = false;

void decode(benchmark::State& state, cmt::Arch arch)
{
	constexpr uint8_t x86[] = {0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20,
							   0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
	constexpr uint8_t arm[] = {0xfd, 0x7b, 0xbf, 0xa9, 0xfd, 0x03, 0x00, 0x91,
							   0x1f, 0x20, 0x03, 0xd5, 0x1f, 0x20, 0x03, 0xd5};
	for (auto iteration : state)
	{
		(void)iteration;
		auto plan =
			cmt::patch::PlanWrap(arch, arch == cmt::Arch::X86_64 ? x86 : arm, 16, 0x1000, 0x4000, 0x8000);
		if (!plan.ok)
		{
			state.SkipWithError(plan.why);
			failed = true;
			break;
		}
		benchmark::DoNotOptimize(plan);
	}
}

void lookup(benchmark::State& state)
{
	const auto& table = cmt::patch::NativeVersionTable();
	for (auto iteration : state)
	{
		(void)iteration;
		const auto* row = cmt::patch::FindVersion(table, "615.71.09");
		if (!row)
		{
			state.SkipWithError("expected version table row is absent");
			failed = true;
			break;
		}
		benchmark::DoNotOptimize(row);
	}
}
} // namespace

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 2;
	}
	benchmark::RegisterBenchmark("decode/x86_64", decode, cmt::Arch::X86_64);
	benchmark::RegisterBenchmark("decode/aarch64", decode, cmt::Arch::Aarch64);
	benchmark::RegisterBenchmark("table_lookup", lookup);
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return failed ? 1 : 0;
}
