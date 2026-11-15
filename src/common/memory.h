#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "error.h"

namespace cmt {

enum class Pages : std::uint8_t
{
	Small,
	Thp,
	Hugetlb
};

const char* PagesName(Pages p);
Result<Pages> ParsePages(const std::string& s);

std::optional<std::size_t> ParseMemInfoValue(const std::string& meminfo, const char* name,
											 bool page_count = false);

std::size_t BasePageBytes();
std::size_t ThpBytes();
std::optional<std::size_t> HugetlbPageBytes();
std::optional<std::size_t> HostMemoryAvailable(Pages p, std::size_t pending_ordinary_bytes = 0);

class HostBuffer
{
public:
	HostBuffer() = default;
	[[nodiscard]] static Result<HostBuffer> Create(std::size_t requested_bytes, Pages requested_pages);

	~HostBuffer()
	{
		Reset();
	}

	HostBuffer(HostBuffer&& o) noexcept;
	HostBuffer& operator=(HostBuffer&& o) noexcept;
	HostBuffer(const HostBuffer&) = delete;
	HostBuffer& operator=(const HostBuffer&) = delete;

	unsigned char* data() const noexcept
	{
		return data_pointer;
	}

	std::size_t size() const noexcept
	{
		return bytes;
	}

	Pages PagePolicy() const noexcept
	{
		return page_policy;
	}

	std::optional<long> HugeKiB() const;

	static long Live() noexcept;

private:
	void Reset() noexcept;

	unsigned char* data_pointer = nullptr;
	void* mapping = nullptr;
	std::size_t bytes = 0;
	std::size_t mapping_bytes = 0;
	Pages page_policy = Pages::Small;
};

std::optional<long> SmapsKiB(const void* p, std::size_t n, const char* field);

} // namespace cmt
