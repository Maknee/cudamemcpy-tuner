# A relocatable bundle keeps the launcher and its companions together. Kernel
# sources/modules remain explicit inputs; ship only the workflow and patch set.
# Two install components: `cmt` is the release package (what README.md lists), `evaluation` adds cmt-evaluate and
# its files. `cmake --install build` installs both; `--component cmt` only the first.
install(TARGETS cmt_hook cmt_trace LIBRARY DESTINATION . COMPONENT cmt)
install(FILES ${CMT_ROOT}/LICENSE ${CMT_ROOT}/THIRD_PARTY_NOTICES DESTINATION . COMPONENT cmt)
# licenses/<library>/: the license files of every fetched library compiled into an installed binary
# Keystone only builds the patch assembler, which is not installed.
set(CMT_LICENSE_FILES_abseil LICENSE)
set(CMT_LICENSE_FILES_benchmark LICENSE)
set(CMT_LICENSE_FILES_capstone LICENSE.TXT LICENSE_LLVM.TXT)
set(CMT_LICENSE_FILES_cli11 LICENSE)
set(CMT_LICENSE_FILES_expected COPYING)
set(CMT_LICENSE_FILES_fmt LICENSE)
set(CMT_LICENSE_FILES_function2 LICENSE.txt)
set(CMT_LICENSE_FILES_indicators LICENSE LICENSE.termcolor)
set(CMT_LICENSE_FILES_picosha2 LICENSE)
set(CMT_LICENSE_FILES_simdjson LICENSE LICENSE-MIT)
set(CMT_LICENSE_FILES_spdlog LICENSE)
set(CMT_LICENSE_FILES_unordered_dense LICENSE)
foreach(dep IN LISTS CMT_BUILT_DEPENDENCIES)
  if(dep STREQUAL "keystone")
    continue()
  endif()
  if(NOT DEFINED CMT_LICENSE_FILES_${dep})
    message(FATAL_ERROR "No license files listed for ${dep}: add CMT_LICENSE_FILES_${dep}")
  endif()
  foreach(file IN LISTS CMT_LICENSE_FILES_${dep})
    if(NOT EXISTS "${${dep}_SOURCE_DIR}/${file}")
      message(FATAL_ERROR "${dep}: no license file ${${dep}_SOURCE_DIR}/${file}")
    endif()
    install(FILES "${${dep}_SOURCE_DIR}/${file}" DESTINATION licenses/${dep} COMPONENT cmt)
  endforeach()
endforeach()
if(TARGET cmt)
  # Building the launcher also builds the companions used by its normal actions.
  add_dependencies(cmt cmt_hook cmt_trace)
  foreach(companion gpu_copy cpu_loops)
    if(TARGET ${companion})
      add_dependencies(cmt ${companion})
    endif()
  endforeach()
  install(TARGETS cmt RUNTIME DESTINATION . COMPONENT cmt)
  # Keep the launcher's workflow/catalog in sync after source edits as well as install.
  file(GLOB_RECURSE CMT_KERNEL_ASSETS CONFIGURE_DEPENDS
    "${CMT_ROOT}/kernel_patches/nvidia-open/*.sh"
    "${CMT_ROOT}/kernel_patches/nvidia-open/*.json"
    "${CMT_ROOT}/kernel_patches/nvidia-open/*.patch"
    "${CMT_ROOT}/kernel_patches/nvidia-open/*.md")
  foreach(asset IN LISTS CMT_KERNEL_ASSETS)
    file(RELATIVE_PATH relative_asset "${CMT_ROOT}" "${asset}")
    configure_file("${asset}" "${CMAKE_CURRENT_BINARY_DIR}/${relative_asset}" COPYONLY)
  endforeach()
  # Include COPYING for the GPL-2.0-only patches.
  install(DIRECTORY "${CMT_ROOT}/kernel_patches/nvidia-open" DESTINATION kernel_patches COMPONENT cmt
    PATTERN "__pycache__" EXCLUDE)
endif()
foreach(tool gpu_copy cpu_loops)
  if(TARGET ${tool})
    install(TARGETS ${tool} RUNTIME DESTINATION . COMPONENT cmt)
  endif()
endforeach()
