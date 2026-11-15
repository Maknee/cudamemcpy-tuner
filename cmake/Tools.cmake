# Optional Rust evaluation companion. The preload library does not depend on it.

if(CMT_BUILD_EVALUATION)
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "CMT_BUILD_EVALUATION builds for the current host; cross-build the Rust runner separately")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_program(CMT_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
  find_program(CMT_RUSTC rustc HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
  execute_process(COMMAND "${CMT_RUSTC}" -vV
    WORKING_DIRECTORY "${CMT_ROOT}/evaluation"
    RESULT_VARIABLE CMT_RUST_INFO_RESULT OUTPUT_VARIABLE CMT_RUST_INFO
    ERROR_VARIABLE CMT_RUST_INFO_ERROR)
  if(NOT CMT_RUST_INFO_RESULT EQUAL 0 OR NOT CMT_RUST_INFO MATCHES "\nhost: ([^\r\n]+)")
    message(FATAL_ERROR "Cannot determine the Rust host target: ${CMT_RUST_INFO_ERROR}")
  endif()
  set(CMT_RUST_HOST "${CMAKE_MATCH_1}")
  set(CMT_EVALUATION_TARGET_DIR "${PROJECT_BINARY_DIR}/evaluation")
  set(CMT_EVALUATION_EXECUTABLE "${PROJECT_BINARY_DIR}/cmt-evaluate")
  # Pin the host explicitly so Cargo environment/config defaults cannot silently
  # put the executable under a different target directory or architecture.
  add_custom_target(cmt_evaluate ALL
    COMMAND "${CMAKE_COMMAND}" -E env "RUSTC=${CMT_RUSTC}"
      "${CMT_CARGO}" build --release --locked --bin cmt-evaluate
      --manifest-path "${CMT_ROOT}/evaluation/Cargo.toml"
      --target "${CMT_RUST_HOST}" --target-dir "${CMT_EVALUATION_TARGET_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMT_EVALUATION_TARGET_DIR}/${CMT_RUST_HOST}/release/cmt-evaluate"
      "${CMT_EVALUATION_EXECUTABLE}"
    COMMAND "${Python3_EXECUTABLE}" "${CMT_ROOT}/tools/crate_licenses.py"
      --cargo "${CMT_CARGO}" --manifest "${CMT_ROOT}/evaluation/Cargo.toml"
      --target "${CMT_RUST_HOST}" --out "${PROJECT_BINARY_DIR}/licenses/rust"
    WORKING_DIRECTORY "${CMT_ROOT}/evaluation"
    BYPRODUCTS "${CMT_EVALUATION_EXECUTABLE}"
    COMMENT "Building the Rust evaluation runner for ${CMT_RUST_HOST}"
    VERBATIM)
  # The workload catalog is the directory evaluation/workloads/ (catalog JSON and drivers).
  # build/cmt-evaluate reads the build tree's copy first, so the copy must follow the sources:
  # configure_file reconfigures when a file changes, and CONFIGURE_DEPENDS when one is added or
  # removed. The copy is rebuilt from scratch, so a removed or renamed catalog file cannot linger
  # (every *.json in the directory is loaded).
  file(GLOB_RECURSE CMT_WORKLOAD_FILES CONFIGURE_DEPENDS "${CMT_ROOT}/evaluation/workloads/*")
  list(FILTER CMT_WORKLOAD_FILES EXCLUDE REGEX "/__pycache__/")
  file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/evaluation/workloads")
  foreach(workload_file IN LISTS CMT_WORKLOAD_FILES)
    file(RELATIVE_PATH relative_workload_file "${CMT_ROOT}" "${workload_file}")
    configure_file("${workload_file}" "${PROJECT_BINARY_DIR}/${relative_workload_file}" COPYONLY)
  endforeach()
  configure_file("${CMT_ROOT}/evaluation/plot.py" "${PROJECT_BINARY_DIR}/evaluation/plot.py" COPYONLY)
  install(FILES "${CMT_ROOT}/evaluation/plot.py" DESTINATION evaluation COMPONENT evaluation)
  install(DIRECTORY "${PROJECT_BINARY_DIR}/licenses/rust" DESTINATION licenses COMPONENT evaluation)
  install(PROGRAMS "${CMT_EVALUATION_EXECUTABLE}" DESTINATION . COMPONENT evaluation)
  install(DIRECTORY "${CMT_ROOT}/evaluation/workloads" DESTINATION evaluation COMPONENT evaluation
    PATTERN "__pycache__" EXCLUDE)
endif()

# Explicit driver-table maintenance. Normal builds use the tracked tables.
option(CMT_BUILD_SIGSCAN "Build the Rust libcuda scanner and its tests" OFF)
set(CMT_SIGSCAN_DRIVERS_DIR "${PROJECT_SOURCE_DIR}/tmp/drivers" CACHE PATH "Verified libcuda cache")
if(CMT_BUILD_SIGSCAN)
  find_program(CMT_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
  set(CMT_SIGSCAN_MANIFEST "${CMT_ROOT}/tools/sigscan/Cargo.toml")
  set(CMT_SIGSCAN_TARGET_DIR "${PROJECT_BINARY_DIR}/sigscan")
  set(CMT_SIGSCAN_EXECUTABLE "${CMT_SIGSCAN_TARGET_DIR}/release/cmt-sigscan")
  add_custom_target(cmt-sigscan ALL
    COMMAND ${CMT_CARGO} build --release --locked
      --manifest-path ${CMT_SIGSCAN_MANIFEST} --target-dir ${CMT_SIGSCAN_TARGET_DIR}
    BYPRODUCTS ${CMT_SIGSCAN_EXECUTABLE}
    COMMENT "Building the Rust libcuda scanner"
    VERBATIM)
  add_custom_target(cmt_fetch_drivers
    COMMAND ${CMT_SIGSCAN_EXECUTABLE} --repo ${CMT_ROOT}
      --drivers ${CMT_SIGSCAN_DRIVERS_DIR} fetch
    DEPENDS cmt-sigscan
    COMMENT "Fetch SHA-256-pinned libcuda packages; extract without running installers"
    VERBATIM)
  foreach(mode generate check)
    # One command scans BOTH architectures before publishing either table.
    add_custom_target(cmt_${mode}_versions
      COMMAND ${CMT_SIGSCAN_EXECUTABLE} --repo ${CMT_ROOT}
        --drivers ${CMT_SIGSCAN_DRIVERS_DIR} ${mode}
      DEPENDS cmt-sigscan
      VERBATIM)
    foreach(arch x86_64 aarch64)
      add_custom_target(cmt_${mode}_versions_${arch}
        COMMAND ${CMT_SIGSCAN_EXECUTABLE} --repo ${CMT_ROOT}
          --drivers ${CMT_SIGSCAN_DRIVERS_DIR} ${mode} --arch ${arch}
        DEPENDS cmt-sigscan
        VERBATIM)
    endforeach()
  endforeach()
endif()
