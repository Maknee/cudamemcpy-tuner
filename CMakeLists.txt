cmake_minimum_required(VERSION 3.24)

# Apply to our targets only. Third-party build directories retain their own flags.
# tl::expected detects -fno-exceptions through __EXCEPTIONS; defining its internal
# exception macro to zero would incorrectly enable its #ifdef exception branches.
function(cmt_exception_free target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-exceptions>)
  target_compile_definitions(${target} PRIVATE FMT_USE_EXCEPTIONS=0)
endfunction()

# Native build tool, including when the hook itself is cross-compiled. Keystone
# allocates internally, so never run it under the application's loader lock.
if(CMT_PATCH_ASSEMBLER_ONLY)
  project(cmt_patch_assembler LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_LIBS_ONLY ON CACHE BOOL "" FORCE)
  set(LLVM_TARGETS_TO_BUILD "AArch64;X86" CACHE STRING "" FORCE)
  set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
  set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CMT_KEYSTONE_SOURCE}" keystone EXCLUDE_FROM_ALL)
  add_executable(cmt_patch_assembler src/gen/assemble.cpp)
  cmt_exception_free(cmt_patch_assembler)
  target_include_directories(cmt_patch_assembler PRIVATE src "${CMT_EXPECTED_INCLUDE}" "${CMT_KEYSTONE_SOURCE}/include")
  target_link_libraries(cmt_patch_assembler PRIVATE keystone)
  set_target_properties(cmt_patch_assembler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  return()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Generate.cmake)
if(CMT_SCRIPT)
  return()
endif()

# Build options and compiler settings.
option(CMT_HOOK_ONLY "Build the preload hook and CPU tests without CUDA" OFF)
project(cudamemcpy_tuner LANGUAGES CXX VERSION 0.1.0)
# The shipped targets have no device code: they use the CUDA headers and the static
# runtime (CUDA::cudart_static), so one build runs on every NVIDIA GPU the driver supports.
# Only tests/test_gpu_semantics.cu has kernels; it needs nvcc and is opt-in.
option(CMT_GPU_SEMANTICS_TESTS "Build the real-GPU semantics tests (needs nvcc)" OFF)
if(CMT_GPU_SEMANTICS_TESTS AND NOT CMT_HOOK_ONLY)
  enable_language(CUDA)
endif()
set(CMT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
get_property(CMT_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT CMAKE_BUILD_TYPE AND NOT CMT_MULTI_CONFIG)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

option(CMT_BUILD_TESTS "Build component correctness checks (GPU checks skip without a GPU)" OFF)
option(CMT_BUILD_BENCHMARKS "Build optional CPU and component benchmarks (gpu_copy is always built with cmt)" ON)
option(CMT_BUILD_EVALUATION "Build and install the Rust evaluation runner" OFF)
if(CMT_BUILD_TESTS)
  enable_testing()
endif()
option(CMT_WERROR "treat warnings as errors" OFF)

if(NOT CMT_HOOK_ONLY)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads REQUIRED)

include(cmake/Dependencies.cmake)

set(CMT_WARN -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion)
if(CMT_WERROR)
  list(APPEND CMT_WARN -Werror)
endif()
string(REPLACE ";" "," _cmt_warn_x "${CMT_WARN}")
list(APPEND CMT_WARN -Wpedantic)
function(cmt_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMT_WARN}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_cmt_warn_x}>)
endfunction()

# The hook carries its own libstdc++ and libgcc, so an application with an older system libstdc++ can preload it.
# Their symbols stay local: the version scripts below export only the hook's interface (tests/check_exports.cmake).
option(MEMCPY_HOOK_STATIC_LIBSTDCXX "Link C++ runtime statically into the hook" ON)
option(MEMCPY_HOOK_WERROR "Treat hook warnings as errors (CMT_WERROR=ON implies it)" OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMT_HOOK_FLAGS -O2 -g -fPIC -Wall -Wextra -Wpedantic -Wno-unused-parameter)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fcf-protection CMT_HAS_CF_PROTECTION)
if(CMT_HAS_CF_PROTECTION)
  list(APPEND CMT_HOOK_FLAGS -fcf-protection)
endif()
if(MEMCPY_HOOK_WERROR OR CMT_WERROR)
  list(APPEND CMT_HOOK_FLAGS -Werror)
endif()

# First-party targets and tests share these paths; dependencies were added above.
include_directories(${CMT_ROOT} ${CMT_ROOT}/src)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(CMT_ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(CMT_ARCH aarch64)
else()
  message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(cmt_arch STATIC ${CMT_ROOT}/src/arch/copy.cpp ${CMT_ROOT}/src/arch/${CMT_ARCH}/copy.cpp)

target_link_libraries(cmt_arch PUBLIC tl::expected)
set_target_properties(cmt_arch PROPERTIES POSITION_INDEPENDENT_CODE ON)
cmt_warnings(cmt_arch)

add_library(cmt_nv_driver STATIC ${CMT_ROOT}/src/cuda/driver.cpp)

target_link_libraries(cmt_nv_driver PRIVATE fmt::fmt-header-only)
set_target_properties(cmt_nv_driver PROPERTIES POSITION_INDEPENDENT_CODE ON)
cmt_warnings(cmt_nv_driver)

add_library(cmt_patch_decode STATIC
  ${CMT_ROOT}/src/arch/disasm/disasm.cpp ${CMT_ROOT}/src/arch/disasm/plan.cpp
  ${CMT_ROOT}/src/arch/disasm/x86_64/relocate.cpp ${CMT_ROOT}/src/arch/disasm/aarch64/relocate.cpp)
target_link_libraries(cmt_patch_decode PUBLIC cmt_arch cmt_capstone)
target_compile_options(cmt_patch_decode PRIVATE ${CMT_HOOK_FLAGS})

# Fixed instruction templates are assembled once at build time; runtime patch
# planning only copies them and writes address literals. Offline source overrides
# use the usual FETCHCONTENT_SOURCE_DIR_KEYSTONE setting.
set(CMT_PATCH_ASSEMBLER "" CACHE FILEPATH "Native patch assembler executable (optional cross-build reuse)")
if(NOT CMT_PATCH_ASSEMBLER)
  find_program(CMT_HOST_CC NAMES cc gcc clang REQUIRED NO_CMAKE_FIND_ROOT_PATH)
  find_program(CMT_HOST_CXX NAMES c++ g++ clang++ REQUIRED NO_CMAKE_FIND_ROOT_PATH)
  include(ExternalProject)
  set(_patch_host "${CMAKE_BINARY_DIR}/patch-assembler-host")
  ExternalProject_Add(cmt_patch_assembler_host
    SOURCE_DIR "${CMT_ROOT}" BINARY_DIR "${_patch_host}"
    CMAKE_ARGS -DCMT_PATCH_ASSEMBLER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMT_HOST_CC} -DCMAKE_CXX_COMPILER=${CMT_HOST_CXX}
      -DCMAKE_TOOLCHAIN_FILE= -DCMAKE_SYSROOT= -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS=
      -DCMT_KEYSTONE_SOURCE=${keystone_SOURCE_DIR} -DCMT_EXPECTED_INCLUDE=${expected_SOURCE_DIR}/include
    # The default build step builds `all`, which is cmt_patch_assembler alone (Keystone is EXCLUDE_FROM_ALL). With
    # a Makefile generator it is $(MAKE), which shares the parent's jobserver; otherwise `cmake --build`, which
    # takes its parallelism from CMAKE_BUILD_PARALLEL_LEVEL or the native tool's default.
    BUILD_ALWAYS TRUE INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_patch_host}/cmt_patch_assembler")
  set(_patch_assembler "${_patch_host}/cmt_patch_assembler")
  set(_patch_dep cmt_patch_assembler_host)
else()
  set(_patch_assembler "${CMT_PATCH_ASSEMBLER}")
  set(_patch_dep "${CMT_PATCH_ASSEMBLER}")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/generated/patch_instructions.h"
  COMMAND "${_patch_assembler}" "${CMAKE_BINARY_DIR}/generated/patch_instructions.h"
  DEPENDS ${_patch_dep} "${CMT_ROOT}/src/gen/assemble.cpp" VERBATIM)
add_custom_target(cmt_patch_instructions DEPENDS "${CMAKE_BINARY_DIR}/generated/patch_instructions.h")
add_dependencies(cmt_patch_decode cmt_patch_instructions)
target_include_directories(cmt_patch_decode PRIVATE "${CMAKE_BINARY_DIR}/generated")

add_library(cmt_base STATIC ${CMT_ROOT}/src/common/helpers.cpp)

target_link_libraries(cmt_base PUBLIC tl::expected cmt_aliases)
set_target_properties(cmt_base PROPERTIES POSITION_INDEPENDENT_CODE ON)
cmt_warnings(cmt_base)

add_library(cmt_linux STATIC ${CMT_ROOT}/src/common/memory.cpp ${CMT_ROOT}/src/common/cpu.cpp
  ${CMT_ROOT}/src/cuda/kernel_status.cpp)

target_link_libraries(cmt_linux PUBLIC cmt_base fmt::fmt-header-only tl::expected Threads::Threads ${CMAKE_DL_LIBS})
# NUMA CPU topology comes from Linux sysfs; no extra runtime library is needed.
target_compile_options(cmt_linux PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(cmt_nv_driver PUBLIC cmt_linux)

add_library(cmt_nv_versions STATIC ${CMT_ROOT}/src/cuda/versions.cpp)

target_link_libraries(cmt_nv_versions PUBLIC cmt_arch)
target_compile_options(cmt_nv_versions PRIVATE ${CMT_HOOK_FLAGS})

add_library(cmt_copying_config STATIC ${CMT_ROOT}/src/config.cpp ${CMT_ROOT}/src/tune_table.cpp)

target_link_libraries(cmt_copying_config PUBLIC cmt_arch cmt_base cmt_linux fmt::fmt-header-only)
target_compile_options(cmt_copying_config PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(cmt_patch_decode PRIVATE cmt_copying_config fmt::fmt-header-only ${CMAKE_DL_LIBS})

add_library(cmt_copying_cpu STATIC ${CMT_ROOT}/src/copying/cpu_copy.cpp)
target_link_libraries(cmt_copying_cpu PUBLIC cmt_copying_config cmt_nv_driver cmt_aliases)
target_compile_options(cmt_copying_cpu PRIVATE ${CMT_HOOK_FLAGS})

# Optional split-copy diagnostics; production ChunkPipeline does not use this pool.
if(CMT_BUILD_TESTS OR
   (CMT_BUILD_BENCHMARKS AND NOT CMT_HOOK_ONLY))
  add_library(cmt_pool_diagnostics STATIC ${CMT_ROOT}/evaluation/microbenchmarks/thread_pool.cpp)
  target_link_libraries(cmt_pool_diagnostics PUBLIC cmt_copying_cpu)
  target_compile_options(cmt_pool_diagnostics PRIVATE ${CMT_HOOK_FLAGS})
endif()

# Production engine, reusable by the preload and native benchmark adapters.
add_library(cmt_chunk STATIC ${CMT_ROOT}/src/copying/chunk_pipeline.cpp)
target_link_libraries(cmt_chunk PUBLIC cmt_copying_cpu cmt_nv_driver cmt_linux)
target_compile_options(cmt_chunk PRIVATE ${CMT_HOOK_FLAGS})

add_library(cmt_copying OBJECT ${CMT_ROOT}/src/copying/progressive_pinning.cpp)
target_link_libraries(cmt_copying PUBLIC cmt_copying_cpu cmt_nv_driver)
target_compile_options(cmt_copying PRIVATE ${CMT_HOOK_FLAGS})

add_library(cmt_context OBJECT ${CMT_ROOT}/src/context.cpp)

target_link_libraries(cmt_context PUBLIC cmt_copying_cpu cmt_nv_driver)
target_compile_options(cmt_context PRIVATE ${CMT_HOOK_FLAGS})

add_library(cmt_hook_core STATIC ${CMT_ROOT}/src/hook.cpp)

target_compile_definitions(cmt_hook_core PRIVATE CMT_HOOK_CORE_ONLY)
target_compile_options(cmt_hook_core PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(cmt_hook_core PUBLIC cmt_copying_cpu cmt_chunk cmt_copying_config cmt_nv_driver cmt_patch_decode cmt_linux cmt_nv_versions cmt_capstone fmt::fmt-header-only Threads::Threads ${CMAKE_DL_LIBS})

add_library(cmt_preload OBJECT ${CMT_ROOT}/src/hook.cpp ${CMT_ROOT}/src/loader.cpp ${CMT_ROOT}/src/trace.cpp
  ${CMT_ROOT}/src/copying/progressive_pinning.cpp ${CMT_ROOT}/src/copying/host_alloc.cpp)
target_compile_definitions(cmt_preload PRIVATE CMT_HOOK_PRELOAD_ONLY)
target_compile_options(cmt_preload PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(cmt_preload PUBLIC cmt_hook_core)

add_library(cmt_trace_objects OBJECT ${CMT_ROOT}/src/hook.cpp ${CMT_ROOT}/src/loader.cpp ${CMT_ROOT}/src/trace.cpp
  ${CMT_ROOT}/src/copying/progressive_pinning.cpp ${CMT_ROOT}/src/copying/host_alloc.cpp)
target_compile_definitions(cmt_trace_objects PRIVATE CMT_HOOK_PRELOAD_ONLY CMT_TRACE_ONLY)
target_compile_options(cmt_trace_objects PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(cmt_trace_objects PUBLIC cmt_hook_core)
set_target_properties(cmt_trace_objects PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

set(CMT_TRACE_EXPORTS_MAP ${CMAKE_CURRENT_BINARY_DIR}/trace.exports)
file(GENERATE OUTPUT ${CMT_TRACE_EXPORTS_MAP} CONTENT [=[
{
  global:
    dlsym;
    cudaMemcpy;
    cudaMemcpyAsync;
    cudaMemcpy_ptds;
    cudaMemcpyAsync_ptsz;
    cuMemcpyHtoD_v2;
    cuMemcpyHtoD_v2_ptds;
    cuMemcpyHtoDAsync_v2;
    cuMemcpyHtoDAsync_v2_ptsz;
    cuMemcpyDtoH_v2;
    cuMemcpyDtoH_v2_ptds;
    cuMemcpyDtoHAsync_v2;
    cuMemcpyDtoHAsync_v2_ptsz;
    cuMemcpyDtoD_v2;
    cuMemcpyDtoD_v2_ptds;
    cuMemcpyDtoDAsync_v2;
    cuMemcpyDtoDAsync_v2_ptsz;
    cuMemcpy;
    cuMemcpy_ptds;
    cuMemcpyAsync;
    cuMemcpyAsync_ptsz;
    cudaMallocHost;
    cudaHostAlloc;
    cudaHostRegister;
    cudaFreeHost;
    cudaHostUnregister;
    cuMemAllocHost_v2;
    cuMemHostAlloc;
    cuMemHostRegister_v2;
    cuMemFreeHost;
    cuMemHostUnregister;
    cuGetProcAddress;
    cuGetProcAddress_v2;
  local: *;
};
]=])

set(CMT_HOOK_EXPORTS_MAP ${CMAKE_CURRENT_BINARY_DIR}/hook.exports)
file(GENERATE OUTPUT ${CMT_HOOK_EXPORTS_MAP} CONTENT [=[
/* Interposed libc symbols and the opt-in benchmark observation ABI. */
{
  global:
    dlsym;
    free;
    realloc;
    munmap;
    mremap;
    madvise;
    pthread_create;
    cmt_tune_audit_begin;
    cmt_tune_audit_end;
  local:
    *;
};
]=])
function(cmt_add_hook_library target)
  set(preload cmt_preload)
  set(exports ${CMT_HOOK_EXPORTS_MAP})
  if(ARGC GREATER 1 AND ARGV1)
    set(preload cmt_trace_objects)
    set(exports ${CMT_TRACE_EXPORTS_MAP})
  endif()
  add_library(${target} SHARED $<TARGET_OBJECTS:${preload}> $<TARGET_OBJECTS:cmt_copying>
    $<TARGET_OBJECTS:cmt_context>)
  set_target_properties(${target} PROPERTIES LINK_DEPENDS ${exports})
  target_link_libraries(${target} PRIVATE cmt_hook_core
    -Wl,--version-script=${exports}
    -Wl,--no-undefined)
  if(MEMCPY_HOOK_STATIC_LIBSTDCXX)
    target_link_libraries(${target} PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endfunction()

cmt_add_hook_library(cmt_hook)
cmt_add_hook_library(cmt_trace TRUE)
set_target_properties(cmt_hook PROPERTIES OUTPUT_NAME cmt_hook)

foreach(target cmt_hook_core cmt_preload cmt_copying cmt_context)
  set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endforeach()
# Measurement, tuning and inspection stay outside the preload dependency graph.
if(NOT CMT_HOOK_ONLY)
  # The shared records and reporter are CPU-only; their compiled code stays outside the preload hook.

  add_library(cmt_metrics STATIC ${CMT_ROOT}/src/measure/metrics.cpp)
  target_link_libraries(cmt_metrics PUBLIC cmt_base cmt_arch cmt_aliases fmt::fmt Threads::Threads benchmark::benchmark PRIVATE cmt_linux)
  cmt_warnings(cmt_metrics)

  add_library(cmt_log INTERFACE)

  target_include_directories(cmt_log SYSTEM INTERFACE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(cmt_log INTERFACE spdlog::spdlog)


  add_library(cmt_machine STATIC ${CMT_ROOT}/src/machine.cpp)

  target_link_libraries(cmt_machine PUBLIC cmt_base cmt_linux cmt_aliases fmt::fmt tl::expected)
  cmt_warnings(cmt_machine)
  cmt_build_info(cmt_machine cuda "CUDA ${CUDAToolkit_VERSION}" "")

  # Only the CLI parser catches exceptions. No hook or copy-engine source enables them.
  set_source_files_properties(${CMT_ROOT}/src/cli.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")

  add_library(cmt_runner STATIC ${CMT_ROOT}/src/runner.cpp)
  cmt_warnings(cmt_runner)
  target_link_libraries(cmt_runner PUBLIC Threads::Threads cmt_base PRIVATE fmt::fmt cmt_copying_config)

  # Shared tunes, CPU-only: what tune.json records of the machine, `cmt export`, and tunes downloaded by `cmt run`
  # (share.h, remote.h).
  add_library(cmt_share STATIC ${CMT_ROOT}/src/share.cpp ${CMT_ROOT}/src/remote.cpp)
  target_link_libraries(cmt_share PUBLIC cmt_copying_config cmt_runner cmt_machine PRIVATE picosha2 fmt::fmt)
  cmt_warnings(cmt_share)

  add_library(cmt_tuner STATIC ${CMT_ROOT}/src/tuner.cpp ${CMT_ROOT}/src/search.cpp ${CMT_ROOT}/src/tune_io.cpp)
  target_link_libraries(cmt_tuner PUBLIC cmt_copying_config cmt_runner cmt_metrics cmt_machine cmt_linux cmt_share)
  target_link_libraries(cmt_tuner PRIVATE picosha2 indicators::indicators)
  target_include_directories(cmt_tuner SYSTEM PRIVATE $<TARGET_PROPERTY:indicators::indicators,INTERFACE_INCLUDE_DIRECTORIES>)
  cmt_warnings(cmt_tuner)

  add_library(cmt_gpu STATIC ${CMT_ROOT}/src/cuda/cuda.cpp ${CMT_ROOT}/src/cuda/device_info.cpp)

  target_link_libraries(cmt_gpu PUBLIC CUDA::cudart_static cmt_metrics cmt_machine cmt_linux PRIVATE fmt::fmt cmt_aliases)
  set_target_properties(cmt_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  cmt_warnings(cmt_gpu)

  add_library(cmt_copying_runtime STATIC ${CMT_ROOT}/src/measure/pin_pipeline.cpp)
  target_link_libraries(cmt_copying_runtime PUBLIC cmt_gpu cmt_chunk cmt_arch cmt_linux cmt_aliases)
  set_target_properties(cmt_copying_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
  cmt_warnings(cmt_copying_runtime)

  add_library(cmt_frontend STATIC ${CMT_ROOT}/src/measure/copy.cpp ${CMT_ROOT}/src/cli.cpp)

  cmt_warnings(cmt_frontend)
  target_link_libraries(cmt_frontend PUBLIC cmt_runner cmt_tuner cmt_copying_runtime PRIVATE CLI11::CLI11 fmt::fmt cmt_aliases cmt_log)
  set_target_properties(cmt_frontend PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_executable(cmt ${CMT_ROOT}/src/main.cpp)
  cmt_warnings(cmt)
  target_link_libraries(cmt PRIVATE cmt_frontend fmt::fmt cmt_log)

  # Required by cmt tune, including when optional diagnostics are disabled.
  add_executable(gpu_copy ${CMT_ROOT}/src/measure/gpu_copy.cpp ${CMT_ROOT}/src/measure/inline_policy.cpp)
  target_link_libraries(gpu_copy PRIVATE cmt_frontend cmt_metrics cmt_copying_config cmt_patch_decode cmt_nv_versions fmt::fmt cmt_log ${CMAKE_DL_LIBS})
  cmt_warnings(gpu_copy)

  if(CMT_BUILD_BENCHMARKS OR CMT_BUILD_EVALUATION)
    add_executable(cpu_loops ${CMT_ROOT}/evaluation/cpu_loops.cpp)
    target_link_libraries(cpu_loops PRIVATE cmt_gpu cmt_metrics cmt_arch cmt_nv_versions benchmark::benchmark Threads::Threads fmt::fmt ${CMAKE_DL_LIBS})
    cmt_warnings(cpu_loops)
  endif()

  if(CMT_BUILD_BENCHMARKS)
    add_executable(handoff ${CMT_ROOT}/evaluation/microbenchmarks/handoff.cpp)
    target_link_libraries(handoff PRIVATE cmt_pool_diagnostics benchmark::benchmark)
    cmt_warnings(handoff)

    # Measure the actual persistent hook pipeline, without LD_PRELOAD or patch installation.
    add_executable(chunk_pipeline_bench ${CMT_ROOT}/evaluation/microbenchmarks/chunk_pipeline.cpp
      $<TARGET_OBJECTS:cmt_copying> $<TARGET_OBJECTS:cmt_context>)
    target_link_libraries(chunk_pipeline_bench PRIVATE cmt_hook_core cmt_gpu benchmark::benchmark)
    cmt_warnings(chunk_pipeline_bench)

    add_executable(patch_decode ${CMT_ROOT}/evaluation/microbenchmarks/patch_decode.cpp)
    target_link_libraries(patch_decode PRIVATE cmt_patch_decode cmt_nv_versions benchmark::benchmark)
    cmt_warnings(patch_decode)

  endif()
endif()
include(cmake/Tools.cmake)

include(cmake/Install.cmake)

# All tests register below the top-level tests directory.
if(CMT_BUILD_TESTS)
  enable_testing()
  add_test(NAME patch_assembler COMMAND "${_patch_assembler}" --self-test)
  add_subdirectory(tests)
endif()

# Root targets and first-party tests are checked independently of the backend,
# hook-only mode, warning settings, or optional benchmark targets.
function(cmt_check_directory directory)
  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(kind ${target} TYPE)
    if(NOT kind STREQUAL "INTERFACE_LIBRARY" AND NOT kind STREQUAL "UTILITY")
      cmt_exception_free(${target})
    endif()
  endforeach()
  get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS children)
    if(child MATCHES "^${CMT_ROOT}/tests(/|$)")
      cmt_check_directory("${child}")
    endif()
  endforeach()
endfunction()
cmt_check_directory("${CMT_ROOT}")
