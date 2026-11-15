# Both frontends must fail after a correct copy if final resource cleanup fails.
# CMT and GPU_COPY are the --wrap twins (tests/CMakeLists.txt, fixtures/test_cleanup_fault.cpp); the
# uninjected control runs the same twin and distinguishes a working fault test from a broken host.
cmake_minimum_required(VERSION 3.18)
foreach(frontend IN ITEMS CMT GPU_COPY)
  set(prefix bench)
  if(frontend STREQUAL "GPU_COPY")
    set(prefix --)
  endif()
  execute_process(COMMAND "${${frontend}}" ${prefix} --path pinned_ce --alloc hostalloc
    --sizes 4K --samples 1 --min-bytes 1 RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(rc EQUAL 77)
    message("SKIP: no CUDA device")
    return()
  endif()
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${frontend} control failed (${rc}): ${out}\n${err}")
  endif()
  foreach(call IN ITEMS cudaFreeHost cudaHostUnregister cudaFree cudaStreamDestroy cudaEventDestroy)
    set(path pinned_ce)
    set(alloc hostalloc)
    if(call STREQUAL "cudaHostUnregister")
      set(alloc register)
    elseif(call STREQUAL "cudaEventDestroy")
      set(path bidir_ce)
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
      "CMT_TEST_FAIL_CLEANUP=${call}" "${${frontend}}" ${prefix}
      --path "${path}" --alloc "${alloc}" --sizes 4K --samples 1 --min-bytes 1
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(rc EQUAL 0 OR NOT err MATCHES "INJECTED cleanup failure" OR
       NOT err MATCHES "cmt panic: CUDA cleanup failed; resource lifetime cannot be established" OR
       NOT err MATCHES "Subprocess aborted")
      message(FATAL_ERROR "${frontend} ${call}: expected fail-stop after injected cleanup failure, got ${rc}: ${out}\n${err}")
    endif()
  endforeach()
endforeach()
