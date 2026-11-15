#pragma once

#include <atomic>
#include <capstone/capstone.h>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "arch/copy.h"

// Static storage avoids allocator reentry under the loader lock.

namespace cmt {
namespace patch {
namespace disasm {

constexpr std::size_t kArenaBytes = 512 * 1024;

class Session
{
public:
	explicit Session(Arch arch);
	~Session();
	Session(const Session&) = delete;
	Session& operator=(const Session&) = delete;

	bool IsValid() const
	{
		return instruction != nullptr;
	}

	const cs_insn* Next(const uint8_t** code, std::size_t* avail, uint64_t* addr);

	bool InGroup(const cs_insn* insn, cs_group_type group) const;

private:
	csh handle = 0;
	cs_insn* instruction = nullptr;
	bool owns_arena = false;
};

std::size_t ArenaUsed();
std::size_t ArenaPeak();

} // namespace disasm
} // namespace patch
} // namespace cmt
