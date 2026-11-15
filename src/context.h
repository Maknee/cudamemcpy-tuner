#pragma once

#include <atomic>
#include <cstdio>
#include <memory>
#include <type_traits>

#include <unistd.h>

#include "common/aliases.h"
#include "config.h"
#include "copying/chunk_pipeline.h"
#include "copying/cpu_copy.h"
#include "copying/progressive_pinning.h"
#include "cuda/driver.h"

namespace cmt {
namespace ring {

struct CopyRequest
{
	cmt::Dir direction;
	cu::CUdeviceptr device;
	uint8_t* host;
	std::size_t bytes;
	cu::CUstream stream;
};

bool MatchesTuningMachine();
cu::CUresult CopyMeasured(const CopyRequest& request, const cmt::Row& row);
cu::CUresult CopyEnvironment(const CopyRequest& request);

bool ForkedChild();

bool CopiesDisabled();
void DisableCopies();

void NoteUsed();

bool UsedInProcess();

void ResetContext(cu::CUcontext ctx);

void ResetDevice(cu::CUdevice dev);

void AfterForkInChild();

} // namespace ring

namespace tune_audit {
class Call
{
public:
	Call(cmt::Dir direction, std::size_t requested_bytes, bool asynchronous);
	cu::CUresult Finish(cu::CUresult status) noexcept;

private:
	cmt::Dir direction;
	std::size_t bytes;
	bool asynchronous;
	bool owns_audit = false;
};

void Reason(Outcome outcome) noexcept;
void Row(const cmt::Row& row) noexcept;
void Otherwise(Outcome outcome) noexcept;
ring::CopyObservation* Observation() noexcept;
pinning::Observation* PinningObservation() noexcept;
} // namespace tune_audit
} // namespace cmt
