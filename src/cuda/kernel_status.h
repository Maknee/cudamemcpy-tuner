#pragma once

#include <optional>
#include <string>

namespace cmt {

namespace nv {
struct HostRegistrationStatus
{
	bool active = false;
	std::optional<bool> large_system_pages;
	const char* evidence = "No active cmt patch journal identifies the loaded modules";
};

HostRegistrationStatus ReadHostRegistrationStatus(const std::string& root = {});
} // namespace nv

bool ProgressivePinningAvailable();

} // namespace cmt
