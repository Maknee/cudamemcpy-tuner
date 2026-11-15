# All dependencies are fetched before our directory include paths are set.
include(FetchContent)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(add_dependency dependency_name endpoint_url endpoint_ref do_build_with_cmake)
  if("${endpoint_url}" MATCHES "\\.(tar\\.gz|tar\\.xz|tgz|zip)$")
    FetchContent_Declare(${dependency_name} URL ${endpoint_url} URL_HASH SHA256=${endpoint_ref}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE ${ARGN})
  else()
    FetchContent_Declare(${dependency_name} GIT_REPOSITORY ${endpoint_url} GIT_TAG ${endpoint_ref} ${ARGN})
  endif()
  if(${do_build_with_cmake})
    FetchContent_MakeAvailable(${dependency_name})
  else()
    FetchContent_GetProperties(${dependency_name})
    if(NOT ${dependency_name}_POPULATED)
      FetchContent_Populate(${dependency_name})
    endif()
  endif()
  list(APPEND CMT_BUILT_DEPENDENCIES ${dependency_name})
  message(STATUS "cmt: ${dependency_name} ${endpoint_url}")
endmacro()

set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
add_dependency(fmt
  https://github.com/fmtlib/fmt/archive/1be298e1bd68957e4cd352e1f676f00e07dcfb57.tar.gz
  d041f836d9b2a3c1c0fe3cce81ffe6a4a257028faa2f4af0dd20d7e58b493766 TRUE)  # 12.2.0
target_compile_definitions(fmt PUBLIC FMT_USE_EXCEPTIONS=0)
target_compile_definitions(fmt-header-only INTERFACE FMT_USE_EXCEPTIONS=0)

if(NOT CMT_HOOK_ONLY)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
  set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
  set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
  add_dependency(spdlog
    https://github.com/gabime/spdlog/archive/79524ddd08a4ec981b7fea76afd08ee05f83755d.tar.gz
    c6dee844e35a54cc80e3ab76aae2310763e0b49c7397549dfa4d4657d9737138 TRUE)  # v1.17.0
endif()

set(SIMDJSON_DEVELOPER_MODE OFF CACHE BOOL "" FORCE)
set(SIMDJSON_EXCEPTIONS OFF CACHE BOOL "" FORCE)
add_dependency(simdjson
  https://github.com/simdjson/simdjson/archive/f5de14f09256982933af2849beb43778bd421ca7.tar.gz
  9157ef4acb368d727873b4dd592f7b1d67d19fcb9201e8bc5a6b093e54e5171a TRUE)  # v4.6.11

if(NOT CMT_HOOK_ONLY)
  set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(CLI11_INSTALL OFF CACHE BOOL "" FORCE)
  add_dependency(cli11
    https://github.com/CLIUtils/CLI11/archive/6c7b07a878ad834957b98d0f9ce1dbe0cb204fc9.tar.gz
    9ac528f302eb1a775ce790e59fe639b13040d8d58183d90f429a845ea5f6bfc2 TRUE)  # v2.4.2
endif()

if(NOT CMT_HOOK_ONLY)
  set(PICOSHA2_TEST OFF CACHE BOOL "" FORCE)
  set(PICOSHA2_EXAMPLE OFF CACHE BOOL "" FORCE)
  add_dependency(picosha2
    https://github.com/okdshin/PicoSHA2/archive/161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29.tar.gz
    6cf473a00c98298d3ddee0aed853e3c799791f49dbc01c996ea46cb248e85802 TRUE)  # v1.0.1, MIT
endif()

if(NOT CMT_HOOK_ONLY)
  set(INDICATORS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(INDICATORS_SAMPLES OFF CACHE BOOL "" FORCE)
  set(INDICATORS_DEMO OFF CACHE BOOL "" FORCE)
  add_dependency(indicators
    https://github.com/p-ranav/indicators/archive/26d39ad8fb438f5f50fb16427f554045d8431030.tar.gz
    cc79414257dc9d0c47c0a33a9d005b8484613ba2f36473709e285d227bedaddc TRUE)  # v2.3, MIT
endif()

add_dependency(function2
  https://github.com/Naios/function2/archive/496c20fede5378190fff471fcd7ee3d8705e5096.tar.gz
  18d6f465b6d17f145318835b8f95f705119fac16388bd65ba9bc88e42826b79a TRUE)  # 4.2.5
target_compile_definitions(function2 INTERFACE FU2_WITH_DISABLED_EXCEPTIONS=1)

add_dependency(unordered_dense
  https://github.com/martinus/unordered_dense/archive/7c58a8b7a48e790fe72f6508eb3b46317cbba9d5.tar.gz
  88d2d30d63b69259a3e4b0b6ec48a41f19ab9a6b30422986f6f3d4bbc57b6950 TRUE)  # v5.3.3

set(CAPSTONE_ARCHITECTURE_DEFAULT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_X86_SUPPORT ON CACHE BOOL "" FORCE)
set(CAPSTONE_ARM64_SUPPORT ON CACHE BOOL "" FORCE)
set(CAPSTONE_X86_REDUCE OFF CACHE BOOL "" FORCE)   # would drop SSE/AVX; the internal memcpy matcher looks for movntps
set(CAPSTONE_BUILD_DIET OFF CACHE BOOL "" FORCE)   # diet mode has no operand details; every user needs them
set(CAPSTONE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_CSTOOL OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_CSTEST OFF CACHE BOOL "" FORCE)
set(CAPSTONE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)  # a normal variable: binds Capstone's option() (CMP0077) without a project-wide change
set(BUILD_STATIC_LIBS ON)
add_dependency(capstone
  https://github.com/capstone-engine/capstone/archive/022575848782a4801fd150fdbc927effcbca0864.tar.gz
  1b0d7efa450e766c120a8dfcf9b539f1e120925fc1478fbe125d6d52bae0c506 TRUE)  # 5.0.9
add_library(cmt_capstone INTERFACE)  # the one name the build links
target_link_libraries(cmt_capstone INTERFACE capstone_static)

# Build only in the native generator subproject, not for the hook's target ABI.
if(NOT CMT_PATCH_ASSEMBLER)
  add_dependency(keystone
    https://github.com/keystone-engine/keystone/archive/0d9567f08c0c23e8f604b2cad3d49450c93cfb40.tar.gz
    0995127fc656680f94b6e29fd28a02b56dd15d83f3f27366bb81155ced594914 FALSE)  # untagged
endif()

# Header-only. SOURCE_SUBDIR include (no CMakeLists.txt there) only downloads it: its own CMakeLists.txt would add
# include/ and share/ to every `cmake --install`. The target it would define is defined here.
add_dependency(expected
  https://github.com/TartanLlama/expected/archive/1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4.tar.gz
  8e6655be615ab8dd3a52745d54c8e88258e58a5c0efe4c4133072abe82414885 TRUE
  SOURCE_SUBDIR include)  # v1.3.1, CC0
add_library(expected INTERFACE)
add_library(tl::expected ALIAS expected)
target_include_directories(expected INTERFACE "$<BUILD_INTERFACE:${expected_SOURCE_DIR}/include>")
# Keep Result definitions identical in the exception-enabled CLI and no-exception
# hook. Upstream 1.3.1 has no opt-out; patch a build-local copy, never the cache.
file(READ "${expected_SOURCE_DIR}/include/tl/expected.hpp" _expected_header)
set(_expected_guard "#if defined(__EXCEPTIONS) || defined(_CPPUNWIND)")
string(FIND "${_expected_header}" "${_expected_guard}" _expected_guard_at)
if(_expected_guard_at EQUAL -1)
  message(FATAL_ERROR "Pinned expected exception guard changed; review the compatibility patch")
endif()
string(REPLACE "${_expected_guard}"
  "#if !defined(TL_EXPECTED_NO_EXCEPTIONS) && (defined(__EXCEPTIONS) || defined(_CPPUNWIND))"
  _expected_header "${_expected_header}")
set(_expected_include "${CMAKE_CURRENT_BINARY_DIR}/expected-no-exceptions")
file(MAKE_DIRECTORY "${_expected_include}/tl")
set(_expected_previous "")
if(EXISTS "${_expected_include}/tl/expected.hpp")
  file(READ "${_expected_include}/tl/expected.hpp" _expected_previous)
endif()
if(NOT _expected_previous STREQUAL _expected_header)
  file(WRITE "${_expected_include}/tl/expected.hpp" "${_expected_header}")
endif()
target_include_directories(expected BEFORE INTERFACE "$<BUILD_INTERFACE:${_expected_include}>")
target_compile_definitions(expected INTERFACE TL_EXPECTED_NO_EXCEPTIONS=1)

if(NOT CMT_HOOK_ONLY)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  add_dependency(benchmark
    https://github.com/google/benchmark/archive/eddb0241389718a23a42db6af5f0164b6e0139af.tar.gz
    5580831930bafe41ecb74f2af47d9adcec25d5bb02c30228d37b7032e0aa6122 TRUE)  # v1.9.4
endif()

add_library(cmt_aliases INTERFACE)
target_link_libraries(cmt_aliases INTERFACE Threads::Threads unordered_dense::unordered_dense
  simdjson::simdjson function2::function2)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
add_dependency(abseil
  https://github.com/abseil/abseil-cpp/archive/2065f4ded0558c6f89fee67c8e5228feb4eb960e.tar.gz
  7f4240fe135c0b0dcdd2efa664f1393b1da7e25031e17560515f452182aa0c5e TRUE)  # 20260817.0
target_link_libraries(cmt_aliases INTERFACE absl::synchronization)
