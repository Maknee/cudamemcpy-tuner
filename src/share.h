#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/aliases.h"
#include "common/error.h"

namespace cmt {
namespace share {

inline constexpr char kShareSchema[] = "cmt.tune-share/1";
inline constexpr std::size_t kMaxShareBytes = std::size_t{2} << 20;

Result<std::string> ShareableMachine(Json machine);
std::vector<std::string> ShareableEnvironment(const std::vector<std::string>& env, std::string_view out_dir = {});
std::string ShareablePath(std::string_view path, std::string_view out_dir);
std::string WithoutValue(std::string text, std::string_view value);
Result<std::string> MachineFingerprint(Json machine, const std::vector<std::string>& env, std::string_view salt);
Result<std::string> NewSalt();

struct Finding
{
	std::string pointer;
	const char* kind;
	std::string value;
};
std::vector<Finding> FindSensitive(Json document);

struct MachineKey
{
	std::string arch;
	std::string gpu;
	std::string cpu;
	std::string driver;
};
Result<MachineKey> KeyOf(Json machine);
std::string Slug(std::string_view text);

inline bool PrintableField(std::string_view text)
{
	return std::size(text) <= 128 && std::all_of(std::begin(text), std::end(text), [](char c) { return c >= ' ' && c < 0x7f; });
}

Result<std::string> Export(const std::string& tune_json);

struct ShareInfo
{
	MachineKey key;
	std::string tier;
};
inline bool IsShare(const std::string& text)
{
	JsonParser parser;
	std::string_view schema;
	return !parser.parse(text)["schema"].get(schema) && schema == kShareSchema;
}
std::string RepositoryPath(const ShareInfo& info, std::string_view text);
Result<ShareInfo> Validate(const std::string& text);

bool IsSelectionOnly(const std::string& text);
struct SelectionOnlyInfo
{
	MachineKey key;
	std::string source;
};
Result<SelectionOnlyInfo> ValidateSelectionOnly(const std::string& text);

} // namespace share
} // namespace cmt
