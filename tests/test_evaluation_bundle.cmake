# An installed bundle holds every file cmt-evaluate finds at run time (evaluation_file):
# `report` runs evaluation/plot.py, `workload` the catalog.
cmake_minimum_required(VERSION 3.18)
file(REMOVE_RECURSE "${PREFIX}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD}" --prefix "${PREFIX}"
  OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "install into ${PREFIX} failed (${rc}): ${err}")
endif()
foreach(file cmt-evaluate gpu_copy cpu_loops evaluation/plot.py evaluation/workloads/appenv.py licenses/rust/index.txt)
  if(NOT EXISTS "${PREFIX}/${file}")
    message(FATAL_ERROR "the installed bundle has no ${file}")
  endif()
endforeach()
