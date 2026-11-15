#pragma once

namespace cmt {
namespace host_alloc {

void* RedirectLookup(const char* symbol);

void* RedirectQuery(const char* symbol, void* real);

void AfterForkInChild();

void AfterContextTeardown() noexcept;

} // namespace host_alloc
} // namespace cmt
