#include "memory.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

#include "defines.h"
#include "error.h"
#include "helpers.h"

#include "log.h"

namespace cmt {

std::optional<std::size_t> ParseMemInfoValue(const std::string& meminfo, const char* name, bool page_count)
{
	constexpr std::size_t kBytesPerKiB = 1024;
	std::istringstream input(meminfo);
	const std::string key = std::string(name) + ":";
	for (std::string line; std::getline(input, line);)
	{
		if (line.rfind(key, 0) == 0)
		{
			std::istringstream fields(line.substr(key.size()));
			std::string number;
			std::string unit;
			if (!(fields >> number))
			{
				return std::nullopt;
			}
			std::size_t value = 0;
			const auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
			if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size())
			{
				return std::nullopt;
			}
			if (fields >> unit)
			{
				std::string extra;
				if (page_count || unit != "kB" || fields >> extra ||
					value > std::numeric_limits<std::size_t>::max() / kBytesPerKiB)
				{
					return std::nullopt;
				}
				value *= kBytesPerKiB;
			}
			else if (!page_count)
			{
				return std::nullopt;
			}
			return value;
		}
	}
	return std::nullopt;
}

namespace {
std::atomic<long> g_live{0};

std::size_t RoundUp(std::size_t x, std::size_t a)
{
	if (x > kMaxBytes || a == 0 || a > kMaxBytes)
	{
		Panic("round_up out of range");
	}
	return (x + a - 1) / a * a;
}

std::optional<std::size_t> ReadMemInfoValue(const char* name, bool page_count = false)
{
	const auto text = ReadTextFile("/proc/meminfo");
	return text ? ParseMemInfoValue(*text, name, page_count) : std::nullopt;
}

bool HasController(const std::string& controllers, const std::string& name)
{
	std::istringstream words(controllers);
	for (std::string word; words >> word;)
	{
		if (word == name)
		{
			return true;
		}
	}
	return false;
}

std::optional<std::string> MountPath(const std::string& encoded)
{
	std::string path;
	for (std::size_t i = 0; i < encoded.size(); ++i)
	{
		if (encoded[i] != '\\')
		{
			path += encoded[i];
			continue;
		}
		unsigned byte = 0;
		if (encoded.size() - i <= 3)
		{
			return std::nullopt;
		}
		const auto parsed = std::from_chars(encoded.data() + i + 1, encoded.data() + i + 4, byte, 8);
		if (parsed.ec != std::errc{} || parsed.ptr != encoded.data() + i + 4 || byte == 0 || byte > UCHAR_MAX)
		{
			return std::nullopt;
		}
		path += static_cast<char>(byte);
		i += 3;
	}
	for (const auto& component : std::filesystem::path(path))
	{
		if (component == ".." || component == ".")
		{
			return std::nullopt;
		}
	}
	return !path.empty() && path.front() == '/' ? std::optional<std::string>(std::move(path)) : std::nullopt;
}

std::optional<std::size_t> CgroupNumber(std::string_view text)
{
	const auto last = text.find_last_not_of(" \t\r\n");
	text = last == std::string_view::npos ? std::string_view{} : text.substr(0, last + 1);
	if (text.empty())
	{
		return std::nullopt;
	}
	std::size_t value = 0;
	const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
	return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()
			   ? std::optional<std::size_t>(value)
			   : std::nullopt;
}

bool CapCgroupMemory(const std::filesystem::path& group, const std::string& controllers,
					 const std::string& controller, const std::string& stem, bool hierarchy_root,
					 std::size_t& bytes, std::size_t pending_charge = 0)
{
	auto limit_text = ReadTextFile((group / (stem + ".max")).string());
	if (!limit_text)
	{
		return limit_text.error().GetCode() == ENOENT &&
			   (hierarchy_root || !HasController(controllers, controller));
	}
	const auto last = limit_text->find_last_not_of(" \t\r\n");
	if (last != std::string::npos && limit_text->substr(0, last + 1) == "max")
	{
		return true;
	}
	const auto limit = CgroupNumber(*limit_text);
	if (!limit)
	{
		return false;
	}
	const auto used_text = ReadTextFile((group / (stem + ".current")).string());
	const auto used = used_text ? CgroupNumber(*used_text) : std::nullopt;
	if (!used)
	{
		return false;
	}
	const auto remaining = *limit > *used ? *limit - *used : 0;
	bytes = std::min(bytes, remaining > pending_charge ? remaining - pending_charge : 0);
	return true;
}

std::optional<std::size_t> CgroupMemoryBound(std::size_t bytes, std::optional<std::size_t> huge_page_bytes,
											 std::size_t pending_ordinary_bytes = 0)
{
	const auto membership = ReadTextFile("/proc/self/cgroup");
	const auto mounts = ReadTextFile("/proc/self/mountinfo");
	if (!membership || !mounts)
	{
		return std::nullopt;
	}
	std::string process_path;
	std::istringstream groups(*membership);
	for (std::string line; std::getline(groups, line);)
	{
		const auto first = line.find(':');
		const auto second = first == std::string::npos ? first : line.find(':', first + 1);
		if (second == std::string::npos)
		{
			return std::nullopt;
		}
		std::string controllers = line.substr(first + 1, second - first - 1);
		std::replace(controllers.begin(), controllers.end(), ',', ' ');
		if (HasController(controllers, "memory") ||
			(huge_page_bytes && HasController(controllers, "hugetlb")))
		{
			return std::nullopt;
		}
		if (line.rfind("0::", 0) == 0)
		{
			if (!process_path.empty() || second + 1 == line.size())
			{
				return std::nullopt;
			}
			process_path = line.substr(second + 1);
		}
	}
	if (process_path.empty())
	{
		return bytes;
	}
	if (process_path.front() != '/')
	{
		return std::nullopt;
	}
	for (const auto& component : std::filesystem::path(process_path))
	{
		if (component == ".." || component == ".")
		{
			return std::nullopt;
		}
	}

	std::string mount_root;
	std::string mount_point;
	bool account_hugetlb_memory = false;
	std::istringstream mount_lines(*mounts);
	for (std::string line; std::getline(mount_lines, line);)
	{
		std::istringstream fields(line);
		std::string ignored;
		std::string root;
		std::string point;
		std::string type;
		std::string super_options;
		if (!(fields >> ignored >> ignored >> ignored >> root >> point >> ignored))
		{
			return std::nullopt;
		}
		while (fields >> ignored && ignored != "-")
		{
		}
		if (!(fields >> type >> ignored >> super_options) || type != "cgroup2")
		{
			continue;
		}
		const auto decoded_root = MountPath(root);
		const auto decoded_point = MountPath(point);
		if (!decoded_root || !decoded_point ||
			!(*decoded_root == "/" || process_path == *decoded_root ||
			  process_path.rfind(*decoded_root + "/", 0) == 0))
		{
			continue;
		}
		if (mount_point.empty() || decoded_root->size() < mount_root.size())
		{
			mount_root = *decoded_root;
			mount_point = *decoded_point;
			account_hugetlb_memory =
				("," + super_options + ",").find(",memory_hugetlb_accounting,") != std::string::npos;
		}
	}
	if (mount_point.empty())
	{
		return std::nullopt;
	}
	std::string huge_stem;
	if (huge_page_bytes)
	{
		std::size_t count = *huge_page_bytes;
		const char* unit = "B";
		for (const char* next_unit : {"KB", "MB", "GB"})
		{
			if (count % 1024 != 0)
			{
				break;
			}
			count /= 1024;
			unit = next_unit;
		}
		huge_stem = "hugetlb." + std::to_string(count) + unit;
	}
	const std::filesystem::path visible_root(mount_point);
	const auto relative = process_path.substr(mount_root == "/" ? 1 : mount_root.size());
	auto group = visible_root;
	if (!relative.empty())
	{
		group /= std::filesystem::path(relative).relative_path();
	}
	for (;; group = group.parent_path())
	{
		const auto controllers = ReadTextFile((group / "cgroup.controllers").string());
		const bool at_root = group == visible_root;
		bool hierarchy_root = false;
		if (at_root)
		{
			const auto events = ReadTextFile((group / "cgroup.events").string());
			if (!events && events.error().GetCode() != ENOENT)
			{
				return std::nullopt;
			}
			hierarchy_root = !events;
		}
		if (!controllers)
		{
			return std::nullopt;
		}
		if ((!huge_page_bytes || account_hugetlb_memory) &&
			!CapCgroupMemory(group, *controllers, "memory", "memory", hierarchy_root, bytes,
							 account_hugetlb_memory ? pending_ordinary_bytes : 0))
		{
			return std::nullopt;
		}
		if (huge_page_bytes &&
			(!CapCgroupMemory(group, *controllers, "hugetlb", huge_stem, hierarchy_root, bytes) ||
			 !CapCgroupMemory(group, *controllers, "hugetlb", huge_stem + ".rsvd", hierarchy_root, bytes)))
		{
			return std::nullopt;
		}
		if (at_root)
		{
			return bytes;
		}
	}
}
} // namespace

const char* PagesName(Pages p)
{
	switch (p)
	{
	case Pages::Small:
		return BasePageBytes() == 4 * 1024 ? "4k" : "small";
	case Pages::Thp:
		return "thp";
	case Pages::Hugetlb:
		return "hugetlb";
	}
	return "?";
}

Result<Pages> ParsePages(const std::string& s)
{
	if (s == "4k" || s == "small")
	{
		return Pages::Small;
	}
	if (s == "thp")
	{
		return Pages::Thp;
	}
	if (s == "hugetlb")
	{
		return Pages::Hugetlb;
	}
	return Failure(Error::Kind::Usage, "unknown page kind (4k|thp|hugetlb)");
}

std::size_t BasePageBytes()
{
	static const std::size_t page_bytes = []
	{
		const long reported_bytes = sysconf(_SC_PAGESIZE);
		if (reported_bytes <= 0)
		{
			Panic("Cannot determine the OS page size");
		}
		return static_cast<std::size_t>(reported_bytes);
	}();
	return page_bytes;
}

std::size_t ThpBytes()
{
	static const std::size_t alignment_bytes = []
	{
		std::ifstream size_file("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");
		std::size_t huge_page_bytes = 0;
		const std::size_t base_page_bytes = BasePageBytes();
		if (!(size_file >> huge_page_bytes) || huge_page_bytes < base_page_bytes ||
			huge_page_bytes % base_page_bytes != 0 || (huge_page_bytes & (huge_page_bytes - 1)) != 0)
		{
			return base_page_bytes;
		}
		return huge_page_bytes;
	}();
	return alignment_bytes;
}

std::optional<std::size_t> HugetlbPageBytes()
{
	const auto page_bytes = ReadMemInfoValue("Hugepagesize");
	const std::size_t base_page_bytes = BasePageBytes();
	if (!page_bytes || *page_bytes < base_page_bytes || *page_bytes > kMaxBytes ||
		*page_bytes % base_page_bytes != 0 || (*page_bytes & (*page_bytes - 1)) != 0)
	{
		return std::nullopt;
	}
	return page_bytes;
}

std::optional<std::size_t> HostMemoryAvailable(Pages p, std::size_t pending_ordinary_bytes)
{
	if (p == Pages::Hugetlb)
	{
		const auto free_pages = ReadMemInfoValue("HugePages_Free", true);
		const auto reserved_pages = ReadMemInfoValue("HugePages_Rsvd", true);
		const auto size = HugetlbPageBytes();
		if (!free_pages || !reserved_pages || !size ||
			*free_pages > std::numeric_limits<std::size_t>::max() / *size)
		{
			return std::nullopt;
		}
		const auto unreserved_pages = *free_pages > *reserved_pages ? *free_pages - *reserved_pages : 0;
		return CgroupMemoryBound(unreserved_pages * *size, size, pending_ordinary_bytes);
	}
	const auto available = ReadMemInfoValue("MemAvailable");
	return available ? CgroupMemoryBound(*available, std::nullopt) : std::nullopt;
}

Result<HostBuffer> HostBuffer::Create(std::size_t requested_bytes, Pages requested_pages)
{
	HostBuffer buffer;
	buffer.page_policy = requested_pages;
	if (requested_bytes == 0)
	{
		Panic("HostBuffer of zero bytes");
	}
	if (requested_bytes > kMaxBytes)
	{
		return Failure(Error::Kind::Usage, "buffer is above the 1 TiB limit");
	}
	if (requested_pages == Pages::Hugetlb)
	{
		const auto huge_page_bytes = HugetlbPageBytes();
		if (!huge_page_bytes)
		{
			return Failure(Error::Kind::Unsupported, "cannot determine the default hugetlb page size");
		}
		buffer.mapping_bytes = RoundUp(requested_bytes, *huge_page_bytes);
		buffer.mapping = mmap(nullptr, buffer.mapping_bytes, PROT_READ | PROT_WRITE,
							  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
		if (buffer.mapping == MAP_FAILED)
		{
			const int code = errno;
			buffer.mapping = nullptr;
			return Failure(Error::Kind::Unsupported,
						   "hugetlb mmap failed; reserve pages with vm.nr_hugepages", code);
		}
		buffer.data_pointer = static_cast<unsigned char*>(buffer.mapping);
	}
	else
	{
		const std::size_t huge = ThpBytes();
		buffer.mapping_bytes = RoundUp(requested_bytes, huge) + huge;
		buffer.mapping =
			mmap(nullptr, buffer.mapping_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (buffer.mapping == MAP_FAILED)
		{
			buffer.mapping = nullptr;
			const int code = errno;
			return Failure(Error::Kind::Os, "mmap failed", code);
		}
		buffer.data_pointer = reinterpret_cast<unsigned char*>(
			(reinterpret_cast<std::uintptr_t>(buffer.mapping) + huge - 1) & ~(huge - 1));
		if (madvise(buffer.data_pointer, RoundUp(requested_bytes, huge),
					requested_pages == Pages::Thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0)
		{
			const int e = errno;
			if (!(e == EINVAL && requested_pages == Pages::Small))
			{
				if (e == EINVAL)
				{
					return Failure(Error::Kind::Unsupported,
								   "transparent huge pages are not available on this kernel", e);
				}
				return Failure(Error::Kind::Os, "madvise failed", e);
			}
		}
	}
	buffer.bytes = requested_bytes;
	for (std::size_t i = 0; i < requested_bytes; i += BasePageBytes())
	{
		buffer.data_pointer[i] = 0;
	}
	g_live++;
	return buffer;
}

void HostBuffer::Reset() noexcept
{
	if (mapping)
	{
		if (munmap(mapping, mapping_bytes) != 0)
		{
			Panic("Cannot release host mapping");
		}
		if (bytes != 0)
		{
			--g_live;
		}
	}
	mapping = nullptr;
	data_pointer = nullptr;
	bytes = 0;
	mapping_bytes = 0;
}

HostBuffer::HostBuffer(HostBuffer&& o) noexcept
	: data_pointer(std::exchange(o.data_pointer, nullptr)), mapping(std::exchange(o.mapping, nullptr)),
	  bytes(std::exchange(o.bytes, 0)), mapping_bytes(std::exchange(o.mapping_bytes, 0)),
	  page_policy(o.page_policy)
{
}

HostBuffer& HostBuffer::operator=(HostBuffer&& o) noexcept
{
	if (this != &o)
	{
		Reset();
		data_pointer = std::exchange(o.data_pointer, nullptr);
		mapping = std::exchange(o.mapping, nullptr);
		bytes = std::exchange(o.bytes, 0);
		mapping_bytes = std::exchange(o.mapping_bytes, 0);
		page_policy = o.page_policy;
	}
	return *this;
}

std::optional<long> HostBuffer::HugeKiB() const
{
	if (!data_pointer)
	{
		return 0L;
	}
	return SmapsKiB(data_pointer, bytes, page_policy == Pages::Hugetlb ? "Private_Hugetlb" : "AnonHugePages");
}

long HostBuffer::Live() noexcept
{
	return g_live.load();
}

std::optional<long> SmapsKiB(const void* p, std::size_t n, const char* field)
{
	std::ifstream f("/proc/self/smaps");
	if (!f)
	{
		return std::nullopt;
	}
	bool in = false;
	long sum = 0;
	const auto a = reinterpret_cast<std::uintptr_t>(p), b = a + n;
	const std::string key = std::string(field) + ":";
	for (std::string line; std::getline(f, line);)
	{
		const std::size_t dash = line.find('-'), space = line.find(' ');
		if (dash != std::string::npos && space != std::string::npos && dash < space &&
			line.find_first_not_of("0123456789abcdef") == dash)
		{
			unsigned long long lo = 0, hi = 0;
			const auto lower = std::from_chars(line.data(), line.data() + dash, lo, 16);
			const auto upper = std::from_chars(line.data() + dash + 1, line.data() + space, hi, 16);
			if (lower.ec != std::errc{} || upper.ec != std::errc{} || lower.ptr != line.data() + dash ||
				upper.ptr != line.data() + space)
			{
				return std::nullopt;
			}
			in = lo < b && hi > a;
			continue;
		}
		if (in && line.rfind(key, 0) == 0)
		{
			sum += std::atol(line.c_str() + std::size(key));
		}
	}
	return sum;
}

} // namespace cmt
