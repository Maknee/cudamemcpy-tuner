#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include "common/aliases.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "config.h"
#include "cuda/driver.h"

namespace cmt {
namespace pinning {

void PrepareInterposition();

bool Enabled();

PinningSettings ManualSettings();

struct Observation
{
	uint64_t completed = 0;
	uint64_t registered_bytes = 0;
	uint64_t released_bytes = 0;
	uint64_t registered_dma_bytes = 0;
	uint64_t pageable_dma_bytes = 0;
	uint64_t borrowed_bytes = 0;
	uint64_t chunks = 0;
	uint64_t first_chunk_bytes = 0;
	uint64_t max_chunk_bytes = 0;
	uint64_t max_registered_bytes = 0;
	uint64_t read_only = 0;
};

enum class Dir : std::uint8_t
{
	H2D,
	D2H
};

namespace detail {
struct CopyUseRegistry;
} // namespace detail

class CopyUse
{
public:
	CopyUse(const void* host, std::size_t bytes);
	~CopyUse();

	CopyUse(const CopyUse&) = delete;
	CopyUse& operator=(const CopyUse&) = delete;

	bool HasRegistrationOverlap() const;

private:
	friend struct detail::CopyUseRegistry;
	uintptr_t start = 0;
	uintptr_t end = 0;
	CopyUse* next = nullptr;
	CopyUse* previous = nullptr;
	bool tracked = false;
	bool valid = false;
};

bool TryCopy(Dir dir, cu::CUdeviceptr dev, void* host, std::size_t n, cu::CUstream stream, cu::CUresult* rc,
			 const PinningSettings* settings, Observation* observation = nullptr);

bool OnFree(void* p, std::size_t usable);
bool Holds(const void* p, std::size_t usable);
void OnUnmap(const void* p, std::size_t n);

void ReleaseContext(cu::CUcontext ctx);
void ReleaseAllRegistrations();
bool Active();

namespace detail {
inline thread_local int t_no_driver = 0;
} // namespace detail

class NoDriverScope
{
public:
	NoDriverScope()
	{
		++detail::t_no_driver;
	}

	~NoDriverScope()
	{
		--detail::t_no_driver;
	}

	NoDriverScope(const NoDriverScope&) = delete;
	NoDriverScope& operator=(const NoDriverScope&) = delete;
};

inline void MarkInternalThread()
{
	detail::t_no_driver = 1;
}

void MarkDriverThread();

void AfterForkInChild();

void Shutdown();

} // namespace pinning
} // namespace cmt

namespace cmt {
namespace copying {
namespace services {
void ReleaseAllocation(void* pointer);
} // namespace services
} // namespace copying
} // namespace cmt
