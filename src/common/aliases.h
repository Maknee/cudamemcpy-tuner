#pragma once

#include <atomic>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

#include <absl/synchronization/mutex.h>
#include <ankerl/unordered_dense.h>
#include <function2/function2.hpp>
#include <simdjson.h>

template <typename Signature>
using Function = fu2::function<Signature>;

using Json = simdjson::dom::element;
using JsonParser = simdjson::dom::parser;
using JsonWriter = simdjson::builder::string_builder;

class Mutex : public absl::Mutex
{
public:
	constexpr Mutex() : absl::Mutex(absl::kConstInit)
	{
	}
};

using ConditionVariable = absl::CondVar;

namespace cmt {

template <typename K, typename V>
using HashMap = ankerl::unordered_dense::map<K, V>;

} // namespace cmt
