// CudaDeviceCount's reading of cudaGetDeviceCount (cuda/cuda.h, DeviceCountFrom): no device and no driver are
// zero devices, so the GPU tests skip on a driverless runner; an outdated driver stays an error. Makes no CUDA call.
#include "check.h"
#include "cuda/cuda.h"

using namespace cmt;

TEST(no_device_or_no_driver_is_zero_and_an_old_driver_an_error)
{
	CHECK(check::Require(DeviceCountFrom(cudaSuccess, 2, true)) == 2);
	CHECK(check::Require(DeviceCountFrom(cudaErrorNoDevice, 0, true)) == 0);
	CHECK(check::Require(DeviceCountFrom(cudaErrorStubLibrary, 0, true)) == 0);
	CHECK(check::Require(DeviceCountFrom(cudaErrorInsufficientDriver, 0, false)) == 0);
	// libcuda.so.1 is installed and older than the runtime: the user must update it, not read "no device".
	const auto old = DeviceCountFrom(cudaErrorInsufficientDriver, 0, true);
	CHECK(!old && old.error().GetKind() == Error::Kind::Cuda &&
		  old.error().GetCode() == static_cast<int>(cudaErrorInsufficientDriver));
	// Other failures are not an absent driver, whatever libcuda's state.
	for (const auto status : {cudaErrorInitializationError, cudaErrorSystemDriverMismatch, cudaErrorUnknown})
	{
		CHECK(!DeviceCountFrom(status, 0, false) && !DeviceCountFrom(status, 0, true));
	}
}

TEST_MAIN()
