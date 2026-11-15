# tests/fixtures/aarch64/run.cmake - ctest's emulator for the aarch64 cross build (toolchain.cmake sets it):
# runs one aarch64 test program under qemu user-mode emulation.
#
#   cmake -DQEMU=<qemu-aarch64> -DSYSROOT=<glibc sysroot> -DRUNTIME=<dir of the aarch64 libstdc++> \
#         -P tests/fixtures/aarch64/run.cmake -- <program> [args...]
#
# Why a script and not qemu itself as the emulator: many hook tests set LD_PRELOAD and LD_LIBRARY_PATH with
# `cmake -E env` for the program under test. Those are meant for the aarch64 program, but the x86-64 qemu process
# would see them first (its own dynamic loader would try to preload an aarch64 library). This script moves them to
# the guest (qemu -E) and removes them from qemu's own environment; every other variable reaches the guest as is.
# The guest's library path gets the cross compiler's runtime (libstdc++, libgcc_s) appended.
cmake_minimum_required(VERSION 3.29)  # cmake_language(EXIT): the test's exit status must reach ctest unchanged

set(program_args)
set(after_separator FALSE)
math(EXPR last "${CMAKE_ARGC} - 1")
foreach(i RANGE 1 ${last})
  if(after_separator)
    list(APPEND program_args "${CMAKE_ARGV${i}}")
  elseif(CMAKE_ARGV${i} STREQUAL "--")
    set(after_separator TRUE)
  endif()
endforeach()
if(NOT program_args)
  message(FATAL_ERROR "tests/fixtures/aarch64/run.cmake: no program after --")
endif()

if("$ENV{LD_LIBRARY_PATH}" STREQUAL "")
  set(guest_env -E "LD_LIBRARY_PATH=${RUNTIME}")
else()
  set(guest_env -E "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${RUNTIME}")
endif()
if(DEFINED ENV{LD_PRELOAD})
  list(APPEND guest_env -E "LD_PRELOAD=$ENV{LD_PRELOAD}")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env --unset=LD_PRELOAD --unset=LD_LIBRARY_PATH
          ${QEMU} -L ${SYSROOT} ${guest_env} ${program_args}
  RESULT_VARIABLE rc)
if(NOT rc MATCHES "^[0-9]+$")
  message("tests/fixtures/aarch64/run.cmake: ${rc}")  # killed by a signal: execute_process reports it as text
  set(rc 1)
endif()
cmake_language(EXIT ${rc})
