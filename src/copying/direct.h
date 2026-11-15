#pragma once

#include <cstddef>

#include "common/defines.h"

namespace cmt {
namespace copying {

template <class Backend>
struct DirectRequest
{
	Dir direction;
	typename Backend::DevicePtr device;
	void* host;
	std::size_t bytes;
	typename Backend::Stream stream = nullptr;
	bool asynchronous = false;
};

template <class Backend>
auto CopyDirect(const DirectRequest<Backend>& request)
{
	return request.direction == Dir::HtoD ? Backend::H2D(request.device, request.host, request.bytes,
														 request.stream, request.asynchronous)
										  : Backend::DtoH(request.host, request.device, request.bytes,
														  request.stream, request.asynchronous);
}
} // namespace copying
} // namespace cmt
