#include "gpu_audit.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "copying/chunk_pipeline.h"

namespace {

template <class Fn>
Fn Symbol(const char* name)
{
	void* p = dlsym(RTLD_DEFAULT, name);
	Fn fn = nullptr;
	static_assert(sizeof(fn) == sizeof(p), "function pointer ABI");
	std::memcpy(&fn, &p, sizeof(fn));
	return fn;
}

} // namespace

bool HookAuditPresent()
{
	return Symbol<cmt::tune_audit::Begin>("cmt_tune_audit_begin") != nullptr &&
		   Symbol<cmt::tune_audit::End>("cmt_tune_audit_end") != nullptr;
}

void HookAuditBegin()
{
	const auto begin = Symbol<cmt::tune_audit::Begin>("cmt_tune_audit_begin");
	if (begin == nullptr || begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) != 1)
	{
		std::fprintf(stderr, "hook audit: begin refused (version %u, size %zu)\n", cmt::tune_audit::kVersion,
					 sizeof(cmt::tune_audit::Snapshot));
		std::exit(1);
	}
}

HookAudit HookAuditEnd()
{
	const auto end = Symbol<cmt::tune_audit::End>("cmt_tune_audit_end");
	cmt::tune_audit::Snapshot s;
	if (end == nullptr || end(&s, sizeof(s)) != 1)
	{
		std::fprintf(stderr, "hook audit: end refused\n");
		std::exit(1);
	}
	HookAudit a;
	a.calls = s.calls;
	a.staged = s.job.completed;
	a.errors = s.errors;
	return a;
}
