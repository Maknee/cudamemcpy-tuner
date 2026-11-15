# toolchain.cmake - cross-compiling the preload hook for aarch64 (Grace, GH200) on an x86-64 host.
#
#   mamba env create -f tests/fixtures/aarch64/environment.yml -n cmt-aarch64
#   cmake -S . -B tmp/build-aarch64 -DCMAKE_TOOLCHAIN_FILE=tests/fixtures/aarch64/toolchain.cmake \
#         -DCMT_HOOK_ONLY=ON -DCMT_BUILD_TESTS=ON -DMEMCPY_HOOK_WERROR=ON
#   cmake --build tmp/build-aarch64 -j4
#   ctest --test-dir tmp/build-aarch64          # under qemu user-mode emulation; see below for what that proves
#
# Use the root build with CMT_HOOK_ONLY=ON. GPU benchmarks require their native backend toolchain.
#
# The compiler, binutils and the glibc 2.28 sysroot come from the conda environment of environment.yml (pinned there),
# found in this order: -DCMT_AARCH64_ENV=<prefix>, $CMT_AARCH64_ENV, the active conda environment if it has the
# compiler, then an environment named cmt-aarch64 under $MAMBA_ROOT_PREFIX, $CONDA_ROOT or the usual install roots.
#
# Tests: when the environment has qemu-aarch64 (environment.yml installs it), it becomes the ctest emulator through
# tests/fixtures/aarch64/run.cmake, so the cross-compiled unit and end-to-end tests run on this host. qemu runs aarch64 code
# with the host's coherent instruction cache and x86-64 memory ordering: a passing test shows the code computes the
# right bytes and patches the right instructions; it cannot show that a barrier or an instruction-cache flush is
# sufficient on real hardware. Without qemu the tests are built (compile check) and registered DISABLED; on a native
# aarch64 build (no toolchain file) they run normally.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_cmt_triple aarch64-conda-linux-gnu)
set(_cmt_candidates)
if(DEFINED CMT_AARCH64_ENV)
  list(APPEND _cmt_candidates "${CMT_AARCH64_ENV}")
endif()
if(DEFINED ENV{CMT_AARCH64_ENV})
  list(APPEND _cmt_candidates "$ENV{CMT_AARCH64_ENV}")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _cmt_candidates "$ENV{CONDA_PREFIX}")
endif()
foreach(_root "$ENV{MAMBA_ROOT_PREFIX}" "$ENV{CONDA_ROOT}" "$ENV{HOME}/miniforge3" "$ENV{HOME}/mambaforge"
              "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3" "/opt/conda")
  if(NOT _root STREQUAL "")
    list(APPEND _cmt_candidates "${_root}/envs/cmt-aarch64")
  endif()
endforeach()
set(_cmt_env "")
foreach(_c IN LISTS _cmt_candidates)
  if(EXISTS "${_c}/bin/${_cmt_triple}-g++")
    set(_cmt_env "${_c}")
    break()
  endif()
endforeach()
if(_cmt_env STREQUAL "")
  message(FATAL_ERROR "aarch64 toolchain: no ${_cmt_triple}-g++ found. Create the environment with\n"
                      "  mamba env create -f tests/fixtures/aarch64/environment.yml -n cmt-aarch64\n"
                      "or pass -DCMT_AARCH64_ENV=<its prefix>. Looked in: ${_cmt_candidates}")
endif()
# try_compile projects re-read this file; pass them the same answer.
set(CMT_AARCH64_ENV "${_cmt_env}" CACHE PATH "conda environment with the aarch64 cross compiler (environment.yml)")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMT_AARCH64_ENV)

set(CMAKE_C_COMPILER "${_cmt_env}/bin/${_cmt_triple}-gcc")
set(CMAKE_CXX_COMPILER "${_cmt_env}/bin/${_cmt_triple}-g++")
set(CMAKE_AR "${_cmt_env}/bin/${_cmt_triple}-ar" CACHE FILEPATH "")
set(CMAKE_RANLIB "${_cmt_env}/bin/${_cmt_triple}-ranlib" CACHE FILEPATH "")
set(CMAKE_NM "${_cmt_env}/bin/${_cmt_triple}-nm" CACHE FILEPATH "")
set(CMAKE_OBJDUMP "${_cmt_env}/bin/${_cmt_triple}-objdump" CACHE FILEPATH "")
set(CMAKE_READELF "${_cmt_env}/bin/${_cmt_triple}-readelf" CACHE FILEPATH "")
set(CMAKE_SYSROOT "${_cmt_env}/${_cmt_triple}/sysroot")

# conda-forge's GCC appends "-rpath <environment>/lib" to every non-static link: its specs file (next to the compiler,
# read after the built-in specs) ends the link command with that clause, so that programs built inside an
# environment find its libraries. A preload library must not carry a path of the build machine, so every link here
# also reads (-specs=, which comes last and wins) a copy of that file with the clause removed.
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=specs OUTPUT_VARIABLE _cmt_specs_path
                OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _cmt_rc)
if(_cmt_rc EQUAL 0 AND IS_ABSOLUTE "${_cmt_specs_path}" AND EXISTS "${_cmt_specs_path}")
  file(READ "${_cmt_specs_path}" _cmt_specs)
  string(REGEX REPLACE "%{!static:-rpath [^}]*}" "" _cmt_specs_norpath "${_cmt_specs}")
  if(NOT _cmt_specs STREQUAL _cmt_specs_norpath)
    set(_cmt_specs_file "${CMAKE_BINARY_DIR}/aarch64-conda-norpath.specs")
    file(WRITE "${_cmt_specs_file}" "${_cmt_specs_norpath}")
    foreach(_kind EXE SHARED MODULE)
      string(APPEND CMAKE_${_kind}_LINKER_FLAGS_INIT " -specs=${_cmt_specs_file}")
    endforeach()
  endif()
endif()

# Headers and libraries only from the sysroot; programs (nvcc, nm, ...) only from the host.
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# The guest needs the cross compiler's aarch64 libstdc++ and libgcc_s, which the glibc sysroot does not have; they
# sit next to the compiler's other target libraries (lib/gcc/<triple>/<version>/).
file(GLOB _cmt_runtime "${_cmt_env}/lib/gcc/${_cmt_triple}/*/libstdc++.so.6")
if(EXISTS "${_cmt_env}/bin/qemu-aarch64" AND _cmt_runtime)
  list(GET _cmt_runtime 0 _cmt_runtime)
  get_filename_component(_cmt_runtime "${_cmt_runtime}" DIRECTORY)
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "${CMAKE_COMMAND};-DQEMU=${_cmt_env}/bin/qemu-aarch64;-DSYSROOT=${CMAKE_SYSROOT};-DRUNTIME=${_cmt_runtime}"
      "-P;${CMAKE_CURRENT_LIST_DIR}/run.cmake;--")
endif()
