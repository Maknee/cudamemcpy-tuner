#pragma once

#include <memory>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <utility>

namespace cmt {

inline bool InstallStderrLogger()
{
	auto logger = std::make_shared<spdlog::logger>("", std::make_shared<spdlog::sinks::stderr_sink_mt>());
	logger->set_pattern("%v");
	spdlog::set_default_logger(std::move(logger));
	return true;
}

inline const bool kStderrLogger = InstallStderrLogger();

} // namespace cmt
