# A successful measurement/report must not conceal a failed output sink.
cmake_minimum_required(VERSION 3.18)
foreach(frontend IN ITEMS CMT GPU_COPY)
  set(prefix bench)
  if(frontend STREQUAL "GPU_COPY")
    set(prefix --)
  endif()
  # Header checks need no GPU and catch failures delayed until stdout flush.
  execute_process(COMMAND "${${frontend}}" ${prefix} --header
    OUTPUT_FILE /dev/full ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 4)
    message(FATAL_ERROR "${frontend} header to /dev/full: expected exit 4, got ${rc}: ${err}")
  endif()
endforeach()

# A pinned loop this CPU cannot run is final (exit 3); 77 means "no GPU now" and the runner retries it.
# Exit 0: this CPU runs it. Usage exit 2: the loop belongs to the other architecture.
foreach(loop IN ITEMS cached@avx2 cached@avx512 cached@sve)
  execute_process(COMMAND "${GPU_COPY}" -- --path staged --fill ${loop} --header
    OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT (rc EQUAL 0 OR rc EQUAL 3 OR (rc EQUAL 2 AND err MATCHES "unknown CPU copy loop")))
    message(FATAL_ERROR "staged --fill ${loop}: expected exit 0, 2 or 3, got ${rc}: ${err}")
  endif()
endforeach()

execute_process(COMMAND "${GPU_COPY}" -- --info
  OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
if(rc EQUAL 77)
  message("SKIP: no CUDA device; both CPU-only header checks passed")
  return()
endif()
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "GPU control failed (${rc}): ${err}")
endif()
foreach(frontend IN ITEMS CMT GPU_COPY)
  set(prefix bench)
  if(frontend STREQUAL "GPU_COPY")
    set(prefix --rows --)
  endif()
  execute_process(COMMAND "${${frontend}}" ${prefix} --path pinned_ce --sizes 4K --samples 1 --min-bytes 1
    OUTPUT_FILE /dev/full ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 4)
    message(FATAL_ERROR "${frontend} CSV to /dev/full: expected exit 4, got ${rc}: ${err}")
  endif()
endforeach()
execute_process(COMMAND "${GPU_COPY}" --csv=/dev/full -- --path pinned_ce --sizes 4K --samples 1 --min-bytes 1
  OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
if(NOT rc EQUAL 4)
  message(FATAL_ERROR "GPU_COPY CSV file /dev/full: expected exit 4, got ${rc}: ${err}")
endif()
