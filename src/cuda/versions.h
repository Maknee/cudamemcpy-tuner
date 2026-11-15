#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include "arch/copy.h"

namespace cmt {
namespace patch {

template <std::size_t N>
struct InlineSite
{
	uint32_t offset;
	uint8_t original[N];
};

constexpr std::size_t kInlineGateBytes = 33;

struct DriverVersion
{
	const char* version;
	uint32_t h2h_va;
	uint8_t h2h_prologue[16];
	uint32_t buffer_init_va;
	uint8_t buffer_init_prologue[16];
	uint32_t internal_memcpy_va;
	uint8_t internal_memcpy_prologue[16];
	uint16_t op_stride;
	uint16_t op_kind;
	uint16_t op_some_pointer1;
	uint16_t op_some_pointer1_off;
	uint16_t op_some_pointer1_width;
	uint16_t op_some_pointer1_height;
	uint16_t op_some_pointer2;
	uint16_t op_some_pointer2_width;
	uint16_t op_some_pointer2_height;
	uint16_t op_some_pointer3;
	uint16_t op_some_pointer3_width;
	uint16_t op_some_pointer3_height;
	uint16_t op_unkx;
	uint16_t op_unky;
	uint16_t op_unkz;
	uint16_t copying_width_bytes;
	uint16_t copying_height_bytes;
	uint16_t copying_depth_bytes;
	uint16_t op_some_pointer1_address;
	uint8_t op_kind_1;
	uint8_t op_kind_2;
	uint8_t op_kind_3;
	InlineSite<4> inline_paged_value;
	InlineSite<13> inline_paged_check;
	InlineSite<11> inline_pinned_value;
	InlineSite<9> inline_pinned_check;
	bool scanned_only;
};

struct VersionTable
{
	Arch arch;
	const DriverVersion* rows;
	std::size_t count;
	bool enabled_by_default;
};

const VersionTable& VersionTableFor(Arch arch);

inline const VersionTable& NativeVersionTable()
{
	return VersionTableFor(kNativeArch);
}

const DriverVersion* FindVersion(const VersionTable& table, const char* version);

const DriverVersion* SelectRow(const VersionTable& table, const char* version, bool allow_unverified,
							   const char** why);

} // namespace patch
} // namespace cmt
