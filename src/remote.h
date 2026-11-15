#pragma once

#include <string>
#include <string_view>

#include "common/error.h"

namespace cmt {

bool IsUrl(std::string_view text);

Result<std::string> DownloadShare(const std::string& url);

struct ClosestTune
{
	std::string path;
	std::string arch;
	std::string cpu;
	std::string gpu;
};

Result<ClosestTune> DownloadClosest(std::string base, std::string_view cpu, std::string_view gpu);

std::string UrlEncode(std::string_view text);

} // namespace cmt
