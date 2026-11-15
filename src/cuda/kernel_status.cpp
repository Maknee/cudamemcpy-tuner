#include "kernel_status.h"
#include "common/helpers.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cmt {

namespace nv {
namespace {

std::string RegistrationPath(const std::string& root, const char* path)
{
	return root.empty() || root == "/" ? std::string(path)
									   : (root.back() == '/' ? root.substr(0, root.size() - 1) : root) + path;
}

std::string TrimRegistrationText(const std::string& text)
{
	const auto first = text.find_first_not_of(" \t\r\n");
	const auto last = text.find_last_not_of(" \t\r\n");
	return first == std::string::npos ? std::string{} : text.substr(first, last - first + 1);
}

std::optional<std::string> RegistrationText(const std::string& path)
{
	auto text = ReadTextFile(path, 4 * 1024 * 1024);
	return text ? std::optional<std::string>(std::move(*text)) : std::nullopt;
}

std::optional<std::string> RegistrationLine(const std::string& path)
{
	auto text = RegistrationText(path);
	if (!text)
	{
		return std::nullopt;
	}
	auto line = TrimRegistrationText(*text);
	return line.empty() ? std::nullopt : std::optional<std::string>(std::move(line));
}

std::optional<std::vector<std::string>> RegistrationDirectory(const std::string& path)
{
	auto entries = ListDirectory(path);
	return entries ? std::optional<std::vector<std::string>>(std::move(*entries)) : std::nullopt;
}

std::optional<std::string> ReadRegistrationJournal(const std::string& root)
{
	int directory = open(root.empty() ? "/" : root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
	if (directory < 0)
	{
		return std::nullopt;
	}
	for (const char* component : {"var", "lib", "cmt", "kernel", "state.json"})
	{
		struct stat info{};
		if (fstat(directory, &info) != 0 || info.st_uid != 0 || (info.st_mode & 0022) != 0)
		{
			close(directory);
			return std::nullopt;
		}
		const bool is_file = std::strcmp(component, "state.json") == 0;
		const int next = openat(directory, component,
								O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (is_file ? O_NONBLOCK : O_DIRECTORY));
		close(directory);
		if (next < 0)
		{
			return std::nullopt;
		}
		directory = next;
	}
	struct stat info{};
	constexpr std::size_t limit = 4 * 1024 * 1024;
	if (fstat(directory, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != 0 ||
		(info.st_mode & 0022) != 0 || info.st_size < 0 || static_cast<std::uintmax_t>(info.st_size) > limit)
	{
		close(directory);
		return std::nullopt;
	}
	std::string text;
	char buffer[4096];
	for (;;)
	{
		const auto count = read(directory, buffer, sizeof buffer);
		if (count < 0 && errno == EINTR)
		{
			continue;
		}
		if (count < 0 || (count > 0 && static_cast<std::size_t>(count) > limit - text.size()))
		{
			close(directory);
			return std::nullopt;
		}
		if (count == 0)
		{
			break;
		}
		text.append(buffer, static_cast<std::size_t>(count));
	}
	if (close(directory) != 0)
	{
		return std::nullopt;
	}
	return text;
}

bool IsHexDigest(std::string_view text, std::size_t length)
{
	return text.size() == length &&
		   text.find_first_not_of("0123456789abcdefABCDEF") == std::string_view::npos;
}

bool HasRegistrationBundle(Json manifest, const std::string& release, const std::string& kernel)
{
	std::uint64_t version = 0;
	std::string_view schema;
	std::string_view recorded_kernel;
	std::string_view commit;
	simdjson::dom::object patches;
	if (manifest["schema"].get(schema) || schema != "cmt.kernel-set" || manifest["version"].get(version) ||
		version != 1 || manifest["kernel"].get(recorded_kernel) || recorded_kernel != kernel ||
		manifest["source_commit"].get(commit) || !IsHexDigest(commit, 40) ||
		manifest["patches"].get(patches) || patches.size() != 3)
	{
		return false;
	}
	for (const char* filename : {"a_pin_user_pages_fast_batch512.patch",
								 "b_unpin_user_pages_dirty_lock.patch", "d_rm_2mib_osdesc.patch"})
	{
		std::string_view digest;
		if (patches[release + "/" + filename].get(digest) || !IsHexDigest(digest, 64))
		{
			return false;
		}
	}
	return true;
}

bool MatchesBuildId(Json manifest, std::string_view note)
{
	if (note.size() < 16)
	{
		return false;
	}
	std::uint32_t name_size = 0, value_size = 0, type = 0;
	std::memcpy(&name_size, note.data(), 4);
	std::memcpy(&value_size, note.data() + 4, 4);
	std::memcpy(&type, note.data() + 8, 4);
	if (name_size != 4 || type != 3 || note.substr(12, 4) != std::string_view("GNU\0", 4) ||
		value_size == 0 || value_size > note.size() - 16)
	{
		return false;
	}
	std::string_view expected;
	if (manifest["build_id"].get(expected) || expected.size() != std::size_t(value_size) * 2)
	{
		return false;
	}
	constexpr std::string_view hex = "0123456789abcdef";
	for (std::size_t i = 0; i < value_size; ++i)
	{
		const auto byte = static_cast<unsigned char>(note[16 + i]);
		if (expected[2 * i] != hex[byte >> 4] || expected[2 * i + 1] != hex[byte & 15])
		{
			return false;
		}
	}
	return true;
}

bool JournalStringEquals(Json object, const char* name, const std::string& value)
{
	std::string_view recorded;
	return !object[name].get(recorded) && recorded == value;
}

bool MatchesLoadedModule(Json manifest, Json observed, const std::string& root, const char* name)
{
	const std::string directory = RegistrationPath(root, "/sys/module/") + name;
	const auto srcversion = RegistrationLine(directory + "/srcversion");
	const auto version = RegistrationLine(directory + "/version");
	if (!srcversion || !JournalStringEquals(manifest, "srcversion", *srcversion) ||
		!JournalStringEquals(observed, "srcversion", *srcversion) ||
		!JournalStringEquals(manifest, "name", name))
	{
		return false;
	}
	if (version && (!JournalStringEquals(manifest, "version", *version) ||
					!JournalStringEquals(observed, "version", *version)))
	{
		return false;
	}
	const auto note = RegistrationText(directory + "/notes/.note.gnu.build-id");
	if (!note || !MatchesBuildId(manifest, *note))
	{
		return false;
	}
	std::string_view recorded_note;
	if (observed["build_id_note"].get(recorded_note) || recorded_note.size() != note->size() * 2)
	{
		return false;
	}
	constexpr std::string_view hex = "0123456789abcdef";
	for (std::size_t i = 0; i < note->size(); ++i)
	{
		const auto byte = static_cast<unsigned char>((*note)[i]);
		if (recorded_note[2 * i] != hex[byte >> 4] || recorded_note[2 * i + 1] != hex[byte & 15])
		{
			return false;
		}
	}
	simdjson::dom::object parameters;
	const auto names = RegistrationDirectory(directory + "/parameters");
	if (!names || observed["parameters"].get(parameters) || parameters.size() != names->size())
	{
		return false;
	}
	for (const auto& key : *names)
	{
		if (key.empty() || key.find('/') != std::string_view::npos ||
			key.find('\0') != std::string_view::npos)
		{
			return false;
		}
		auto value = RegistrationText(directory + "/parameters/" + std::string(key));
		std::string_view recorded;
		if (!value || parameters[key].get(recorded))
		{
			return false;
		}
		while (!value->empty() && value->back() == '\n')
		{
			value->pop_back();
		}
		if (*value != recorded)
		{
			return false;
		}
	}
	return true;
}

} // namespace

HostRegistrationStatus ReadHostRegistrationStatus(const std::string& root)
{
	HostRegistrationStatus result;
	const auto text = ReadRegistrationJournal(root);
	if (!text)
	{
		return result;
	}
	JsonParser parser;
	Json state;
	std::uint64_t version = 0;
	const auto parsed = parser.parse(*text).get(state);
	if (parsed == simdjson::MEMALLOC)
	{
		Panic("kernel state JSON allocation failed");
	}
	const auto boot_id = RegistrationLine(RegistrationPath(root, "/proc/sys/kernel/random/boot_id"));
	const auto kernel = RegistrationLine(RegistrationPath(root, "/proc/sys/kernel/osrelease"));
	if (parsed || state["version"].get(version) || version != 1 || !boot_id || !kernel ||
		!JournalStringEquals(state, "schema", "cmt.kernel-state") ||
		!JournalStringEquals(state, "status", "active") ||
		!JournalStringEquals(state, "bundle", "host-registration") ||
		!JournalStringEquals(state, "boot_id", *boot_id) || !JournalStringEquals(state, "kernel", *kernel))
	{
		result.evidence = "Kernel journal is not an active host-registration bundle for this boot";
		return result;
	}
	Json manifest;
	Json observed;
	const auto release = RegistrationLine(RegistrationPath(root, "/sys/module/nvidia/version"));
	if (state["requested_manifest"].get(manifest) || state["observed"].get(observed) || !release ||
		!HasRegistrationBundle(manifest, *release, *kernel) ||
		!JournalStringEquals(manifest, "variant", "host-registration") ||
		!JournalStringEquals(manifest, "release", *release) ||
		!JournalStringEquals(state, "release", *release))
	{
		result.evidence = "Kernel journal does not identify the loaded NVIDIA driver release";
		return result;
	}
	for (const char* name : {"nvidia", "nvidia_uvm"})
	{
		Json expected_module;
		Json recorded_module;
		if (manifest["modules"][name].get(expected_module) || observed[name].get(recorded_module) ||
			!MatchesLoadedModule(expected_module, recorded_module, root, name))
		{
			result.evidence = "Loaded NVIDIA module identity or parameters differ from the patch journal";
			return result;
		}
	}
	result.large_system_pages = false;
	const auto registry =
		RegistrationLine(RegistrationPath(root, "/sys/module/nvidia/parameters/NVreg_RegistryDwords"));
	if (registry)
	{
		bool found = false;
		std::istringstream settings(*registry);
		for (std::string setting; std::getline(settings, setting, ';');)
		{
			const auto equals = setting.find('=');
			if (equals != std::string::npos &&
				TrimRegistrationText(setting.substr(0, equals)) == "RMAllowSysmemLargePages")
			{
				if (found)
				{
					result.evidence = "Large-system-page setting is ambiguous";
					return result;
				}
				found = true;
				result.large_system_pages = TrimRegistrationText(setting.substr(equals + 1)) == "1";
			}
		}
	}
	const auto current_text = ReadRegistrationJournal(root);
	if (!current_text || *current_text != *text)
	{
		result.evidence = "Kernel patch journal changed during inspection";
		return result;
	}
	result.active = true;
	result.evidence = "Both loaded modules and their parameters match the active patch journal";
	return result;
}

} // namespace nv

bool ProgressivePinningAvailable()
{
	static const bool available = []
	{
		const auto status = nv::ReadHostRegistrationStatus();
		return status.active && status.large_system_pages == true;
	}();
	return available;
}

} // namespace cmt
