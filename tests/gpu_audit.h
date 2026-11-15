// The hook's per-thread copy audit (cmt_tune_audit_begin / _end, copying/chunk_pipeline.h) for the real-GPU
// tests, which nvcc compiles without the hook's headers. gpu_audit.cpp is built by the host compiler against
// the real Snapshot, so a change of the audit ABI fails there (begin refuses a wrong version or size), not
// silently here.
#pragma once

#include <cstdint>

struct HookAudit
{
	std::uint64_t calls = 0;  // copies that entered the hook's routing (CopyThroughHook) on this thread
	std::uint64_t staged = 0; // of them, copies the chunks pipeline completed (the first call's job)
	std::uint64_t errors = 0; // of them, calls that returned an error
};

// The hook is preloaded and exports the audit; false for a stock run.
bool HookAuditPresent();
// Starts an observation on this thread. Exits the process (status 1) when the hook refuses it.
void HookAuditBegin();
// Ends it. Exits the process (status 1) when the hook refuses it.
HookAudit HookAuditEnd();
