#pragma once
#include "common/error.h"

#include <map>
#include <string>

namespace cmt {
struct Config;
}

namespace cmt {
namespace bench {

cmt::Result<std::map<std::string, std::string>> ConfigureInlinePolicy(const cmt::Config& cfg);

} // namespace bench
} // namespace cmt
