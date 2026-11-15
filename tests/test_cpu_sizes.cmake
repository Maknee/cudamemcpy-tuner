# cpu_loops reads --cpu-sizes with gpu_copy's parser (ParseSizes): LO..HI is every power of two, an
# explicit list (bench cpu's default) keeps its sizes and order, and a repeated size is refused.
# --cpu-preview prints the measured points without measuring or initializing CUDA.
cmake_minimum_required(VERSION 3.18)
function(preview_bytes sizes out)
  execute_process(COMMAND "${CPU_LOOPS}" --cpu-preview --cpu-loops=memcpy --cpu-workers=1 --cpu-cache=cold
    "--cpu-sizes=${sizes}" OUTPUT_VARIABLE json ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--cpu-sizes=${sizes}: expected exit 0, got ${rc}: ${err}")
  endif()
  string(REGEX MATCHALL "\"bytes\": *[0-9]+" fields "${json}")
  list(TRANSFORM fields REPLACE "\"bytes\": *" "")
  set(${out} "${fields}" PARENT_SCOPE)
endfunction()

set(expected)
foreach(shift RANGE 12 30)
  math(EXPR bytes "1 << ${shift}")
  list(APPEND expected ${bytes})
endforeach()
preview_bytes("4K..1G" got)
if(NOT got STREQUAL expected)
  message(FATAL_ERROR "4K..1G: expected ${expected}, got ${got}")
endif()

preview_bytes("4K,16K,64K,128K,512K,1M,2M,4M,8M,16M,64M,128M,512M,1G" got)
set(expected 4096 16384 65536 131072 524288 1048576 2097152 4194304 8388608 16777216 67108864 134217728
  536870912 1073741824)
if(NOT got STREQUAL expected)
  message(FATAL_ERROR "bench cpu's list: expected ${expected}, got ${got}")
endif()

execute_process(COMMAND "${CPU_LOOPS}" --cpu-preview --cpu-loops=memcpy --cpu-workers=1 --cpu-cache=cold
  --cpu-sizes=4K,1M,4K OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
if(NOT rc EQUAL 2 OR NOT err MATCHES "unique")
  message(FATAL_ERROR "repeated size: expected the uniqueness usage error, got ${rc}: ${err}")
endif()
