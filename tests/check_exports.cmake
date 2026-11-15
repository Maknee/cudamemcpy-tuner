# check_exports.cmake - the hook must export exactly dlsym and the memory-release interposers of the
# registration cache (free, realloc, munmap, mremap, madvise), pthread_create (threads libcuda creates),
# and the two opt-in tune observation functions.
# Nothing else may leave: not Capstone (linked in statically for the patcher), not libstdc++, not other hook internals.
# Usage: cmake -DLIB=<path to .so> [-DNM=<nm for the library's architecture>] -P check_exports.cmake
if(NOT NM)
  find_program(NM nm)
endif()
if(NOT NM)
  message(STATUS "nm not found; skipping")
  return()
endif()
execute_process(COMMAND ${NM} -D --defined-only ${LIB} OUTPUT_VARIABLE out RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nm failed on ${LIB}")
endif()
string(REGEX MATCHALL "[^\n]+" lines "${out}")
set(names "")
foreach(line ${lines})
  string(REGEX REPLACE "^.* ([^ ]+)$" "\\1" name "${line}")
  list(APPEND names ${name})
endforeach()
list(SORT names)
message(STATUS "exports: ${names}")
if(NOT names STREQUAL "cmt_tune_audit_begin;cmt_tune_audit_end;dlsym;free;madvise;mremap;munmap;pthread_create;realloc")
  message(FATAL_ERROR "unexpected exports: ${names}")
endif()
