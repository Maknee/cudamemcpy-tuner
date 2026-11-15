// The shared dispatcher must preserve direction, API completion mode, caller
// stream and backend errors. A byte check prevents a no-op from passing.
#include "copying/direct.h"
#include "tests/check.h"
#include <array>
#include <cstring>

namespace {
struct Api
{
	using DevicePtr = unsigned char*;
	using Stream = void*;
	static inline Stream seen_stream = nullptr;
	static inline bool seen_async = false;
	static inline cmt::Dir seen_dir = cmt::Dir::HtoD;
	static inline int status = 0;

	static int H2D(DevicePtr dst, const void* src, std::size_t bytes, Stream stream, bool async)
	{
		seen_dir = cmt::Dir::HtoD;
		return move(dst, src, bytes, stream, async);
	}

	static int DtoH(void* dst, DevicePtr src, std::size_t bytes, Stream stream, bool async)
	{
		seen_dir = cmt::Dir::DtoH;
		return move(dst, src, bytes, stream, async);
	}

	static int move(void* dst, const void* src, std::size_t bytes, Stream stream, bool async)
	{
		seen_stream = stream;
		seen_async = async;
		// SAFETY: test callers supply distinct arrays with at least bytes storage.
		if (status == 0)
		{
			std::memcpy(dst, src, bytes);
		}
		return status;
	}
};
} // namespace

TEST(dispatch_preserves_bytes_stream_mode_and_error)
{
	int stream;
	for (const auto dir : {cmt::Dir::HtoD, cmt::Dir::DtoH})
	{
		for (const bool async : {false, true})
		{
			std::array<unsigned char, 31> host{}, device{};
			auto& source = dir == cmt::Dir::HtoD ? host : device;
			source.fill(0x5a);
			const cmt::copying::DirectRequest<Api> request{
				dir, std::data(device), std::data(host), std::size(host), &stream, async};
			Api::status = 713;
			CHECK_EQ(cmt::copying::CopyDirect(request), 713);
			CHECK(host != device);
			Api::status = 0;
			CHECK_EQ(cmt::copying::CopyDirect(request), 0);
			CHECK(host == device);
			CHECK(Api::seen_dir == dir);
			CHECK(Api::seen_stream == &stream);
			CHECK(Api::seen_async == async);
		}
	}
}
TEST_MAIN()
