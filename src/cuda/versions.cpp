#include "cuda/versions.h"

namespace cmt {
namespace patch {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

namespace x86_64_rows {
#include "gen/versions_db.inc"
} // namespace x86_64_rows

namespace aarch64_rows {
#include "gen/versions_aarch64.inc"
} // namespace aarch64_rows

#pragma GCC diagnostic pop

namespace {

const VersionTable kX86Table{Arch::X86_64, x86_64_rows::g_versions, x86_64_rows::g_versions_count, true};
const VersionTable kA64Table{Arch::Aarch64, aarch64_rows::g_versions, aarch64_rows::g_versions_count, false};

} // namespace

const VersionTable& VersionTableFor(Arch arch)
{
	return arch == Arch::X86_64 ? kX86Table : kA64Table;
}

const DriverVersion* FindVersion(const VersionTable& table, const char* version)
{
	if (version == nullptr)
	{
		return nullptr;
	}
	for (std::size_t i = 0; i < table.count; i++)
	{
		if (std::strcmp(table.rows[i].version, version) == 0)
		{
			return &table.rows[i];
		}
	}
	return nullptr;
}

const DriverVersion* SelectRow(const VersionTable& table, const char* version, bool allow_unverified,
							   const char** why)
{
	const DriverVersion* row = FindVersion(table, version);
	if (row == nullptr)
	{
		*why = "no row for this version";
		return nullptr;
	}
	const bool verified = table.enabled_by_default && !row->scanned_only;
	if (!verified && !allow_unverified)
	{
		*why = "the row is unverified (MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS=1 uses it)";
		return nullptr;
	}
	*why = verified ? "" : "UNVERIFIED row, used because MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS=1";
	return row;
}

} // namespace patch
} // namespace cmt
