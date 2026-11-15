# One entry point for component correctness. Performance runs live in
# evaluation/ (and evaluation/microbenchmarks/); they are not pass/fail speed tests.
include_directories(${CMT_ROOT}/tests)

function(cmt_component name)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE ${ARGN})
  cmt_warnings(test_${name})
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

cmt_component(sync cmt_aliases cmt_linux)
set_tests_properties(sync PROPERTIES TIMEOUT 30)

if(NOT CMT_HOOK_ONLY)
  cmt_component(io cmt_base cmt_linux)
  target_compile_definitions(test_io PRIVATE CMT_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}")
  cmt_component(logging cmt_base cmt_log fmt::fmt-header-only)
  target_compile_definitions(test_logging PRIVATE CMT_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}")
  cmt_component(metrics cmt_metrics fmt::fmt)
  cmt_component(machine cmt_machine cmt_linux)
  cmt_component(tuner cmt_tuner fmt::fmt)
  cmt_component(runner cmt_tuner fmt::fmt)
  cmt_component(selection cmt_copying_config cmt_share)
  cmt_component(share cmt_tuner fmt::fmt)
  target_link_libraries(test_share PRIVATE picosha2)
  target_compile_definitions(test_share PRIVATE CMT_TEST_FIXTURES="${CMT_ROOT}/tests/fixtures")
  cmt_component(device_count cmt_gpu)
  if(TARGET cmt)
    add_test(NAME cli_commands COMMAND ${CMAKE_COMMAND} -DCMT=$<TARGET_FILE:cmt>
      -P ${CMT_ROOT}/tests/test_cli_commands.cmake)
    add_test(NAME selection_check COMMAND test_selection --cmt $<TARGET_FILE:cmt>)
    # export, check and run through cmt; the hook loads the fake libcuda, never a GPU. HTTP: a local server.
    add_test(NAME share_cli COMMAND test_share --cmt $<TARGET_FILE:cmt> $<TARGET_FILE_DIR:fake_cuda>)
    set_tests_properties(share_cli PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    set_tests_properties(selection_check share_cli PROPERTIES ENVIRONMENT "CMT_INTERNAL_ENABLE=1")
    if(TARGET gpu_copy)
      add_test(NAME selection_resume COMMAND test_selection --resume $<TARGET_FILE:cmt>)
      add_test(NAME tune_instant COMMAND test_selection --instant $<TARGET_FILE:cmt>)
      set_tests_properties(selection_resume tune_instant PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu RUN_SERIAL TRUE
        ENVIRONMENT "CMT_INTERNAL_ENABLE=1")
    endif()
  endif()
  add_executable(test_direct test_direct.cpp)
  cmt_warnings(test_direct)
  add_test(NAME direct_copier COMMAND test_direct)
  if(TARGET cmt_frontend)
    cmt_component(cli cmt_frontend)
  endif()

  cmt_component(kernel_module cmt_machine)
  foreach(component gpu_paths gpu_resources measurement)
    cmt_component(${component} cmt_frontend)
    set_tests_properties(${component} PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu RUN_SERIAL TRUE)
  endforeach()
  if(TARGET gpu_copy)
    add_test(NAME benchmark_output_failures COMMAND ${CMAKE_COMMAND}
      -DCMT=$<TARGET_FILE:cmt> -DGPU_COPY=$<TARGET_FILE:gpu_copy>
      -P ${CMT_ROOT}/tests/test_output_failure.cmake)
    # cmt and gpu_copy link the CUDA runtime statically (CUDA::cudart_static), so their cudaFreeHost & co are
    # bound at link time and an LD_PRELOAD library never sees them. These test-only twins relink the same sources
    # and libraries with --wrap: each undefined reference to NAME becomes __wrap_NAME (the fault fixture), which
    # calls the runtime's NAME as __real_NAME. The shipped binaries are not changed.
    set(cleanup_wraps cudaFreeHost cudaHostUnregister cudaFree cudaStreamDestroy cudaEventDestroy)
    list(TRANSFORM cleanup_wraps PREPEND "LINKER:--wrap=")
    foreach(frontend cmt gpu_copy)
      get_target_property(frontend_sources ${frontend} SOURCES)
      get_target_property(frontend_libraries ${frontend} LINK_LIBRARIES)
      add_executable(${frontend}_cleanup_fault ${frontend_sources} fixtures/test_cleanup_fault.cpp)
      target_include_directories(${frontend}_cleanup_fault PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_link_libraries(${frontend}_cleanup_fault PRIVATE ${frontend_libraries})
      target_link_options(${frontend}_cleanup_fault PRIVATE ${cleanup_wraps})
      cmt_warnings(${frontend}_cleanup_fault)
    endforeach()
    add_test(NAME benchmark_cleanup_failures COMMAND ${CMAKE_COMMAND}
      -DCMT=$<TARGET_FILE:cmt_cleanup_fault> -DGPU_COPY=$<TARGET_FILE:gpu_copy_cleanup_fault>
      -P ${CMT_ROOT}/tests/test_cleanup.cmake)
    set_tests_properties(benchmark_output_failures benchmark_cleanup_failures
      PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP:" LABELS gpu RUN_SERIAL TRUE)
  endif()
  if(TARGET cpu_loops)
    add_test(NAME cpu_loops_sizes COMMAND ${CMAKE_COMMAND}
      -DCPU_LOOPS=$<TARGET_FILE:cpu_loops> -P ${CMT_ROOT}/tests/test_cpu_sizes.cmake)
  endif()
endif()

if(CMT_BUILD_SIGSCAN)
  add_test(NAME sigscan_units COMMAND ${CMT_CARGO} test --locked --offline
    --manifest-path ${CMT_SIGSCAN_MANIFEST} --target-dir ${CMT_SIGSCAN_TARGET_DIR})
  set_tests_properties(sigscan_units PROPERTIES LABELS sigscan)
  # The tables against the pinned libcuda files, which `cmake --build BUILD --target cmt_fetch_drivers` downloads:
  # 77 (skipped, never passed) while none of an architecture's files is there.
  foreach(arch x86_64 aarch64)
    add_test(NAME versions_db_${arch}_current COMMAND sh -c
      "ls \"$1\"/*/libcuda.so.* >/dev/null 2>&1 || { echo \"SKIP: no libcuda files in $1; build the cmt_fetch_drivers target\"; exit 77; }; shift; exec \"$@\""
      versions_db ${CMT_SIGSCAN_DRIVERS_DIR}/${arch}
      ${CMT_SIGSCAN_EXECUTABLE} --repo ${CMT_ROOT} --drivers ${CMT_SIGSCAN_DRIVERS_DIR} verify --arch ${arch})
    set_tests_properties(versions_db_${arch}_current PROPERTIES LABELS "sigscan;drivers" SKIP_RETURN_CODE 77)
  endforeach()
  add_test(NAME sigscan_cli_rejects_non_elf COMMAND ${CMT_SIGSCAN_EXECUTABLE}
    scan ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt)
  set_tests_properties(sigscan_cli_rejects_non_elf PROPERTIES WILL_FAIL ON LABELS "sigscan;e2e")
endif()

# Hook components and fake-driver integration. The fixture library has no GPU dependency.
# Unit tests (no CUDA needed) and the end-to-end pipeline test against a
# fake libcuda.so.1.
#
# Architectures. Every test builds for x86-64 and aarch64. In a cross build (tests/fixtures/aarch64/toolchain.cmake)
# CMAKE_CROSSCOMPILING_EMULATOR runs them: add_test inserts it for a target COMMAND, and MH_RUN puts it in front of
# the program in the `cmake -E env` commands. A cross build without an emulator still builds every test (a compile
# check) and registers it DISABLED (end of this file); a native aarch64 build runs them like any other.
set(MH_RUN ${CMAKE_CROSSCOMPILING_EMULATOR})
set(MH_CANNOT_RUN OFF)
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set(MH_CANNOT_RUN ON)
endif()

# These fixtures prove loader semantics independently of CUDA. The provider must
# not appear in LD_LIBRARY_PATH: only the caller DSO's RUNPATH may find it. The
# global-scope executable retains both DSOs for RTLD_NEXT; the local executable
# loads only the caller and lets its own lookups discover the provider.
add_library(mh_loader_provider SHARED ${CMT_ROOT}/tests/fixtures/loader_provider.cpp)
add_library(mh_loader_caller SHARED ${CMT_ROOT}/tests/fixtures/loader_caller.cpp)
foreach(target mh_loader_provider mh_loader_caller)
  target_compile_options(${target} PRIVATE ${CMT_HOOK_FLAGS})
  set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/loader
    BUILD_RPATH "$ORIGIN")
endforeach()
target_link_libraries(mh_loader_caller PRIVATE ${CMAKE_DL_LIBS})
add_executable(test_loader_next ${CMT_ROOT}/tests/test_loader_lookup.cpp)
target_compile_definitions(test_loader_next PRIVATE MH_LOADER_LINKED=1)
target_link_libraries(test_loader_next PRIVATE mh_loader_caller -Wl,--no-as-needed mh_loader_provider ${CMAKE_DL_LIBS})
add_executable(test_loader_local ${CMT_ROOT}/tests/test_loader_lookup.cpp)
target_link_libraries(test_loader_local PRIVATE ${CMAKE_DL_LIBS})
foreach(target test_loader_next test_loader_local)
  target_compile_options(${target} PRIVATE ${CMT_HOOK_FLAGS})
  add_dependencies(${target} cmt_hook mh_loader_caller mh_loader_provider)
endforeach()
foreach(mode next default sibling)
  if(mode STREQUAL "next")
    set(loader_command $<TARGET_FILE:test_loader_next> next)
  else()
    set(loader_command $<TARGET_FILE:test_loader_local> ${mode} $<TARGET_FILE:mh_loader_caller>)
  endif()
  foreach(hooked OFF ON)
    if(hooked)
      set(loader_preload "LD_PRELOAD=$<TARGET_FILE:cmt_hook>")
    else()
      set(loader_preload "LD_PRELOAD=")
    endif()
    add_test(NAME loader_${mode}_${hooked}
      COMMAND ${CMAKE_COMMAND} -E env --unset=LD_LIBRARY_PATH ${loader_preload}
      MEMCPY_HOOK_NO_EAGER=1 ${MH_RUN} ${loader_command})
    set_tests_properties(loader_${mode}_${hooked} PROPERTIES TIMEOUT 20)
  endforeach()
endforeach()

# Copy-loop names the end-to-end tests ask for, per architecture (src/arch/copy.h): an aarch64 hook
# rejects the x86 names, so a table naming one would be ignored whole and the tests would check the wrong path. The
# loop names below are the current METHOD@ISA spellings.
if(CMT_ARCH STREQUAL "x86_64")
  set(MH_LOOP_A movsb)
  set(MH_LOOP_B nt_store@avx2)
  set(MH_LOOP_C nt_store@sse2)
  set(MH_LOOP_NTLOAD nt_load@avx2)
  set(MH_LOOP_FLUSH movsb+flush)
  set(MH_LOOP_ALIAS nt_store@avx2)
else()
  set(MH_LOOP_A cached@neon)
  set(MH_LOOP_B nt_store@neon)
  set(MH_LOOP_C nt_load@neon)
  set(MH_LOOP_NTLOAD nt_load@neon)
  set(MH_LOOP_FLUSH cached@neon+flush)
  set(MH_LOOP_ALIAS nt_store@neon)
endif()

# The library the LD_PRELOAD tests load (mh_e2e, mh_tune, ...): the default one, or MH_PRELOAD set around a call.
set(MH_PRELOAD cmt_hook)

function(mh_test name)
  add_executable(${name} ${name}.cpp)
  target_compile_options(${name} PRIVATE ${CMT_HOOK_FLAGS})
  target_link_libraries(${name} PRIVATE cmt_hook_core)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

mh_test(test_cpu_copy)
# aarch64 under qemu: the same test on CPUs the emulator can pretend to be (QEMU_CPU reaches qemu through
# tests/fixtures/aarch64/run.cmake): no SVE, SVE as wide as NEON (Grace: NEON must stay), and 256-byte SVE. test_cpu_copy checks
# that detection saw exactly that (expect-sve=) and runs every variant the CPU has.
if(CMT_ARCH STREQUAL "aarch64" AND CMAKE_CROSSCOMPILING_EMULATOR)
  foreach(cpu "cortex-a57:0" "max,sve-default-vector-length=16:16" "max,sve-default-vector-length=256:256")
    string(REPLACE ":" ";" cpu_pair "${cpu}")
    list(GET cpu_pair 0 qemu_cpu)
    list(GET cpu_pair 1 sve_bytes)
    add_test(NAME test_cpu_copy_sve${sve_bytes}
      COMMAND ${CMAKE_COMMAND} -E env QEMU_CPU=${qemu_cpu} ${MH_RUN} $<TARGET_FILE:test_cpu_copy> expect-sve=${sve_bytes})
  endforeach()
endif()
mh_test(test_thread_pool)
target_link_libraries(test_thread_pool PRIVATE cmt_pool_diagnostics)
add_test(NAME test_thread_pool_shared_cpu COMMAND test_thread_pool shared_cpu)
set_tests_properties(test_thread_pool_shared_cpu PROPERTIES TIMEOUT 60)
if(TARGET chunk_pipeline_bench)
  # The component executable links AllocPinned itself. This mode returns before
  # benchmark setup or CUDA initialization and needs no GPU or preload library.
  add_test(NAME test_staging_allocation_overflow
    COMMAND chunk_pipeline_bench --check-allocation-overflow)
  set_tests_properties(test_staging_allocation_overflow PROPERTIES TIMEOUT 10 LABELS cpu)
endif()
mh_test(test_config)
mh_test(test_patch)
mh_test(test_patch_plan)
mh_test(test_h2h_engine)

# The in-driver CPU leg is serial; whole-slot workers belong to ChunkPipeline.
add_test(NAME test_h2h_engine_bench_dma
  COMMAND ${CMAKE_COMMAND} -E env STAGING_BENCH_DMA=1 ${MH_RUN} $<TARGET_FILE:test_h2h_engine>)

# Exactly the interposed libc symbols leave each library: no Capstone, libstdc++ or hook symbol. Read with the
# target's nm (CMAKE_NM), so the cross build checks its aarch64 libraries too.
add_test(NAME test_exports
  COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DLIB=$<TARGET_FILE:cmt_hook>
  -P ${CMT_ROOT}/tests/check_exports.cmake)
# Fake driver: <build>/tests/fakecuda/libcuda.so.1
add_library(fake_cuda SHARED ${CMT_ROOT}/tests/fixtures/fake_libcuda.cpp)
target_compile_options(fake_cuda PRIVATE ${CMT_HOOK_FLAGS})
set_target_properties(fake_cuda PROPERTIES
  OUTPUT_NAME cuda SUFFIX ".so.1"
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fakecuda)

add_dependencies(test_loader_local fake_cuda)
add_test(NAME loader_cuda_queries
  COMMAND ${CMAKE_COMMAND} -E env
  LD_PRELOAD=$<TARGET_FILE:cmt_hook> MEMCPY_HOOK_NO_EAGER=1
  STAGING_THREADS_COPY_NUM_BUFFERS=2
  ${MH_RUN} $<TARGET_FILE:test_loader_local> cuda $<TARGET_FILE:fake_cuda>)
set_tests_properties(loader_cuda_queries PROPERTIES TIMEOUT 20)

add_executable(test_pipeline_e2e ${CMT_ROOT}/tests/test_pipeline_e2e.cpp)
target_compile_options(test_pipeline_e2e PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_pipeline_e2e PRIVATE cmt_linux ${CMAKE_DL_LIBS})
add_dependencies(test_pipeline_e2e fake_cuda cmt_hook)

# name, then environment assignments.
function(mh_e2e name)
  add_test(NAME e2e_${name}
    COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    CMT_OVERRIDE=progressive_pinning=off
    ${ARGN}
    ${MH_RUN} $<TARGET_FILE:test_pipeline_e2e>)
endfunction()

mh_e2e(off)
mh_e2e(off_debug MEMCPY_HOOK_DEBUG=1 STAGING_COPY_NT_IMPL=${MH_LOOP_ALIAS})
mh_e2e(k1 STAGING_THREADS_COPY_NUM_BUFFERS=3 STAGING_BUFFER_SIZE=64)
mh_e2e(k1_one_slot STAGING_THREADS_COPY_NUM_BUFFERS=1 STAGING_BUFFER_SIZE=64)

add_test(NAME e2e_progressive_pinning
  COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=1024
    CMT_PINNING_MIN_KIB=1024 CMT_PINNING_FIRST_MIB=1 CMT_PINNING_STEP_MIB=1
    ${MH_RUN} $<TARGET_FILE:test_pipeline_e2e> progressive_pinning)
set_tests_properties(e2e_progressive_pinning PROPERTIES SKIP_RETURN_CODE 77)
mh_e2e(no_eager MEMCPY_HOOK_NO_EAGER=1 MEMCPY_HOOK_DEBUG=1 STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=128)

mh_e2e(flush_loops STAGING_THREADS_COPY_NUM_BUFFERS=3 STAGING_BUFFER_SIZE=1024 STAGING_COPY_NT_IMPL=${MH_LOOP_FLUSH}
  STAGING_COPY_NT_IMPL_D2H=memcpy STAGING_COPY_FLUSH_BYTES=1000 STAGING_PIPELINE_WORKERS=2)
mh_e2e(chunks_w1 STAGING_PIPELINE_WORKERS=1
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
mh_e2e(chunks_w2 STAGING_PIPELINE_WORKERS=2
  STAGING_THREADS_COPY_NUM_BUFFERS=3 STAGING_BUFFER_SIZE=64 MEMCPY_HOOK_DEBUG=1)
mh_e2e(chunks_w3 STAGING_PIPELINE_WORKERS=3
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=1024 STAGING_COPY_NT_IMPL=${MH_LOOP_A}
  STAGING_COPY_NT_IMPL_D2H=${MH_LOOP_B})
mh_e2e(chunks_w2_pinned STAGING_PIPELINE_WORKERS=2
  STAGING_THREADS_COPY_NUM_BUFFERS=1 STAGING_BUFFER_SIZE=128 STAGING_THREADS_COPY_CPUS=0,1
  STAGING_COPY_NT_IMPL=${MH_LOOP_FLUSH})

mh_e2e(chunks_error STAGING_PIPELINE_WORKERS=3
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64 FAKE_FAIL_PINNED_DMA_AT=4)

add_executable(test_registration_release ${CMT_ROOT}/tests/test_registration_release.cpp)
target_compile_options(test_registration_release PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_registration_release PRIVATE cmt_linux ${CMAKE_DL_LIBS})
add_dependencies(test_registration_release fake_cuda cmt_hook)
add_test(NAME registration_owned_pinned
  COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
    CMT_PINNING_MIN_KIB=0 CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1
    ${MH_RUN} $<TARGET_FILE:test_registration_release> pinned)
add_test(NAME registration_release_failure
  COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
    CMT_PINNING_MIN_KIB=0 CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1
    ${MH_RUN} $<TARGET_FILE:test_registration_release> failures)
set_tests_properties(registration_owned_pinned registration_release_failure PROPERTIES SKIP_RETURN_CODE 77)

# Opt-in pinned host allocation (src/copying/host_alloc.h): off unless CMT_CUDA_HOST_ALLOC_HOOK=1.
add_executable(test_host_alloc ${CMT_ROOT}/tests/test_host_alloc.cpp)
target_compile_options(test_host_alloc PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_host_alloc PRIVATE cmt_linux Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(test_host_alloc fake_cuda cmt_hook)
set(MH_HOST_ALLOC_ENV LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}> LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda)
add_test(NAME host_alloc_unset COMMAND ${CMAKE_COMMAND} -E env --unset=CMT_CUDA_HOST_ALLOC_HOOK ${MH_HOST_ALLOC_ENV}
  ${MH_RUN} $<TARGET_FILE:test_host_alloc> off)
add_test(NAME host_alloc_off COMMAND ${CMAKE_COMMAND} -E env ${MH_HOST_ALLOC_ENV} CMT_CUDA_HOST_ALLOC_HOOK=0
  ${MH_RUN} $<TARGET_FILE:test_host_alloc> off)
add_test(NAME host_alloc_on COMMAND ${CMAKE_COMMAND} -E env ${MH_HOST_ALLOC_ENV} CMT_CUDA_HOST_ALLOC_HOOK=1
  ${MH_RUN} $<TARGET_FILE:test_host_alloc> on)
# Private memory (cmt run --cuda_host_alloc_hook private), and the private switch alone, which changes nothing.
add_test(NAME host_alloc_private COMMAND ${CMAKE_COMMAND} -E env ${MH_HOST_ALLOC_ENV} CMT_CUDA_HOST_ALLOC_HOOK=1
  CMT_CUDA_HOST_ALLOC_PRIVATE=1 ${MH_RUN} $<TARGET_FILE:test_host_alloc> private)
add_test(NAME host_alloc_private_alone COMMAND ${CMAKE_COMMAND} -E env --unset=CMT_CUDA_HOST_ALLOC_HOOK
  ${MH_HOST_ALLOC_ENV} CMT_CUDA_HOST_ALLOC_PRIVATE=1 ${MH_RUN} $<TARGET_FILE:test_host_alloc> off)
set_tests_properties(host_alloc_unset host_alloc_off host_alloc_on host_alloc_private host_alloc_private_alone
  PROPERTIES TIMEOUT 60)

# Progressive pinning's registration limit (the GPU's memory unless set) and registration refusals mid-copy.
# Each sets its own CMT_OVERRIDE, so none is in the progressive_pinning=on list below.
add_executable(test_pinning_limits ${CMT_ROOT}/tests/test_pinning_limits.cpp)
target_compile_options(test_pinning_limits PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_pinning_limits PRIVATE cmt_copying_config cmt_linux ${CMAKE_DL_LIBS})
add_dependencies(test_pinning_limits fake_cuda cmt_hook)
set(MH_LIMIT_ENV LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}> LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  CMT_PINNING_MIN_KIB=0 CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1)
add_test(NAME registration_limit_default COMMAND ${CMAKE_COMMAND} -E env ${MH_LIMIT_ENV}
  CMT_OVERRIDE=progressive_pinning=on ${MH_RUN} $<TARGET_FILE:test_pinning_limits> default)
add_test(NAME registration_limit_env COMMAND ${CMAKE_COMMAND} -E env ${MH_LIMIT_ENV}
  CMT_OVERRIDE=progressive_pinning=on CMT_PINNING_MAX_MIB=8 ${MH_RUN} $<TARGET_FILE:test_pinning_limits> explicit)
add_test(NAME registration_limit_override COMMAND ${CMAKE_COMMAND} -E env ${MH_LIMIT_ENV}
  CMT_OVERRIDE=progressive_pinning=on,pinning_max_mib=8 CMT_PINNING_MAX_MIB=4096
  ${MH_RUN} $<TARGET_FILE:test_pinning_limits> explicit)
add_test(NAME registration_refusals COMMAND ${CMAKE_COMMAND} -E env ${MH_LIMIT_ENV}
  CMT_OVERRIDE=progressive_pinning=on ${MH_RUN} $<TARGET_FILE:test_pinning_limits> refusals)
set_tests_properties(registration_limit_default registration_limit_env registration_limit_override
  registration_refusals PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

# A real missing optional API export (not merely a zero support value).
add_library(fake_cuda_no_attribute SHARED ${CMT_ROOT}/tests/fixtures/fake_libcuda.cpp)
target_compile_options(fake_cuda_no_attribute PRIVATE ${CMT_HOOK_FLAGS})
target_compile_definitions(fake_cuda_no_attribute PRIVATE FAKE_OMIT_DEVICE_ATTRIBUTE=1)
set_target_properties(fake_cuda_no_attribute PROPERTIES OUTPUT_NAME cuda SUFFIX ".so.1"
  CXX_VISIBILITY_PRESET hidden LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fakecuda_no_attribute)
add_executable(test_registration_readonly ${CMT_ROOT}/tests/test_registration_readonly.cpp)
target_compile_options(test_registration_readonly PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_registration_readonly PRIVATE cmt_linux ${CMAKE_DL_LIBS})
add_dependencies(test_registration_readonly fake_cuda fake_cuda_no_attribute cmt_hook)
# An explicit registration limit: these check the registration flags, also when the context's device cannot be
# read, which without a limit declines pinning altogether (registration_limit_default).
function(mh_readonly name mode ro driver)
  add_test(NAME readonly_${name}
    COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}> LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/${driver}
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64 CMT_PINNING_MAX_MIB=64
    CMT_PINNING_MIN_KIB=0 CMT_PINNING_READ_ONLY=${ro}
    CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1
    ${MH_RUN} $<TARGET_FILE:test_registration_readonly> ${mode})
  set_tests_properties(readonly_${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()
mh_readonly(progressive enabled 1 fakecuda)
mh_readonly(default disabled 0 fakecuda)
mh_readonly(missing_api missing_api 1 fakecuda_no_attribute)
add_test(NAME readonly_pinned_bypass
  COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  CMT_PINNING_MIN_KIB=0 CMT_PINNING_READ_ONLY=1
  ${MH_RUN} $<TARGET_FILE:test_registration_release> pinned)
add_test(NAME readonly_release_reuse
  COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  CMT_PINNING_MIN_KIB=0 CMT_PINNING_READ_ONLY=1 CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1
  ${MH_RUN} $<TARGET_FILE:test_registration_release> failures)
set_tests_properties(readonly_pinned_bypass readonly_release_reuse PROPERTIES SKIP_RETURN_CODE 77)

# Canonical measured policy: each scenario gets a clean child process.
add_executable(test_tune_table_e2e ${CMT_ROOT}/tests/test_tune_table_e2e.cpp)
target_compile_options(test_tune_table_e2e PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_tune_table_e2e PRIVATE cmt_copying_config cmt_linux Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(test_tune_table_e2e fake_cuda cmt_hook)
if(TARGET cmt_share)
  # The "share" scenario: a `cmt export` file through the hook.
  target_link_libraries(test_tune_table_e2e PRIVATE cmt_share)
  target_compile_definitions(test_tune_table_e2e PRIVATE CMT_TEST_SHARE=1)
endif()
add_test(NAME tune_measured_policy
  COMMAND ${CMAKE_COMMAND} -E env --unset=LD_PRELOAD --unset=CMT_CONFIG
    $<TARGET_FILE:test_tune_table_e2e> $<TARGET_FILE:cmt_hook> $<TARGET_FILE_DIR:fake_cuda>)
set_tests_properties(tune_measured_policy PROPERTIES TIMEOUT 90)

# Ring lifecycle (test_ring_lifecycle): allocation failure, primary-context teardown, lock order, fork, early dlsym
add_executable(test_ring_lifecycle ${CMT_ROOT}/tests/test_ring_lifecycle.cpp)
target_compile_options(test_ring_lifecycle PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_ring_lifecycle PRIVATE cmt_linux Threads::Threads ${CMAKE_DL_LIBS})
# install_race: the executable's realpath() must interpose the one the hook calls (-rdynamic).
set_target_properties(test_ring_lifecycle PROPERTIES ENABLE_EXPORTS ON)
add_dependencies(test_ring_lifecycle fake_cuda cmt_hook early_dlsym fake_alloc)
# A malloc replacement preloaded after the hook, as tcmalloc/jemalloc would be.
add_library(fake_alloc SHARED ${CMT_ROOT}/tests/fixtures/fake_alloc.cpp)
target_compile_options(fake_alloc PRIVATE ${CMT_HOOK_FLAGS} -fno-builtin)
target_link_libraries(fake_alloc PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(fake_alloc PROPERTIES CXX_VISIBILITY_PRESET hidden)
add_library(early_dlsym SHARED ${CMT_ROOT}/tests/fixtures/early_dlsym.cpp)
target_compile_options(early_dlsym PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(early_dlsym PRIVATE ${CMAKE_DL_LIBS})

# name, mode, then environment assignments.
function(mh_lifecycle name mode)
  add_test(NAME ring_${name}
    COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    CMT_OVERRIDE=progressive_pinning=off
    ${ARGN}
    ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> ${mode})
  set_tests_properties(ring_${name} PROPERTIES TIMEOUT 60)
endfunction()

# C6: a table-only ring (no STAGING_THREADS_COPY_NUM_BUFFERS) whose pinned pool cannot be allocated.
mh_lifecycle(alloc_fail alloc_fail STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# A leftover trace gate must not disable the ordinary optimizing library.
mh_lifecycle(trace_gate_normal_hook alloc_fail STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  CUDAMEMCPY_TUNER_TRACE_ENABLE=1 CUDAMEMCPY_TUNER_TRACE_FD=-1)
# Staging allocation flags: The shared production pipeline allocates through
# alloc_pinned; each must pass the WC flag when asked and no flag by default.
foreach(pmode chunks)
  set(MH_SLOT_ENV STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
  mh_lifecycle(slot_alloc_wc_${pmode} "slot_alloc;wc" ${MH_SLOT_ENV} STAGING_SLOT_ALLOC=hostalloc_wc)
  mh_lifecycle(slot_alloc_cached_${pmode} "slot_alloc;cached" ${MH_SLOT_ENV})
endforeach()
# Context resources across primary-context reset and release.
mh_lifecycle(primary_reset primary_reset STAGING_THREADS_COPY_NUM_BUFFERS=2
  STAGING_BUFFER_SIZE=64)
mh_lifecycle(handoff_stress handoff_stress
  STAGING_PIPELINE_WORKERS=4 STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
add_test(NAME ring_handoff_shared_cpu_default
  COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    STAGING_PIPELINE_WORKERS=4
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
    ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> handoff_stress shared_cpu)
set_tests_properties(ring_handoff_shared_cpu_default PROPERTIES TIMEOUT 60)
# C9: the early library is preloaded after the hook, so its constructor runs before the hook's initializers.
add_test(NAME ring_early_dlsym
  COMMAND ${CMAKE_COMMAND} -E env
  "LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>:$<TARGET_FILE:early_dlsym>"
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
  STAGING_THREADS_COPY_NUM_BUFFERS=1
  ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> early_dlsym)
set_tests_properties(ring_early_dlsym PROPERTIES TIMEOUT 60)

# Additional context lifetime and loader failure cases.
# A: an allocator preloaded after the hook.
add_test(NAME ring_foreign_alloc
  COMMAND ${CMAKE_COMMAND} -E env
  "LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>:$<TARGET_FILE:fake_alloc>"
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> foreign_alloc)
set_tests_properties(ring_foreign_alloc PROPERTIES TIMEOUT 60)
# Stream capture preserves the original driver's semantics.
mh_lifecycle(capture_chunks capture STAGING_THREADS_COPY_NUM_BUFFERS=2
  STAGING_BUFFER_SIZE=64)
# C: per-thread default stream lookups.
mh_lifecycle(ptds ptds STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# A declined first registration must preserve the non-blocking caller stream.
# Without verified patched modules, this scenario explicitly skips (77).
add_test(NAME ring_stream_order_regfail COMMAND ${CMAKE_COMMAND} -E env
  LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}> LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
  FAKE_STRICT_LEGACY=1 CMT_PINNING_MIN_KIB=1024 CMT_PINNING_STEP_MIB=2 CMT_PINNING_FIRST_MIB=1
  ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> stream_order regfail)
set_tests_properties(ring_stream_order_regfail PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
# F: out of memory inside cuCtxDestroy and inside a primary-context reset.
foreach(variant destroy reset)
  add_test(NAME ring_teardown_oom_${variant} COMMAND ${CMAKE_COMMAND} -E env
    "LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>:$<TARGET_FILE:fake_alloc>"
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
    CMT_OVERRIDE=progressive_pinning=off
    ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> teardown_oom ${variant})
  set_tests_properties(ring_teardown_oom_${variant} PROPERTIES TIMEOUT 60)
endforeach()
# G: lookups racing install (no eager dlopen: the first explicit CUDA lookup installs), and an unmatched primary-context address.
mh_lifecycle(install_race install_race MEMCPY_HOOK_NO_EAGER=1 STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
mh_lifecycle(unknown_primary unknown_primary STAGING_THREADS_COPY_NUM_BUFFERS=2
  STAGING_BUFFER_SIZE=64)
# H: a failed slot record after a queued DMA, in the chunks pipeline and in the split ring (one 64 KiB slot each).
mh_lifecycle(record_fail_chunks record_fail STAGING_PIPELINE_WORKERS=1
  STAGING_THREADS_COPY_NUM_BUFFERS=1 STAGING_BUFFER_SIZE=64)

foreach(mode cleanup_success cleanup_drain_failure cleanup_unregister_failure)
  mh_lifecycle(${mode} ${mode} STAGING_PIPELINE_WORKERS=1
    STAGING_THREADS_COPY_NUM_BUFFERS=1 STAGING_BUFFER_SIZE=64 STAGING_SLOT_ALLOC=mmap_4k)
endforeach()
# I: fork before any use of the hook.
mh_lifecycle(fork_unused fork_unused STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# J: sticky errors, concurrent copies, capture modes and failed setup paths.
# A sticky CUDA error during a staged copy, then teardown and a new context.
mh_lifecycle(sticky sticky STAGING_PIPELINE_WORKERS=2
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# A second thread's copy does not wait for the context's busy staging.
mh_lifecycle(concurrent_copy concurrent_copy STAGING_PIPELINE_WORKERS=1
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# The staging build runs in relaxed capture mode while another thread captures globally.
mh_lifecycle(capture_global capture_global STAGING_THREADS_COPY_NUM_BUFFERS=2
  STAGING_BUFFER_SIZE=64)
# A helper that cannot select the context: the staging setup fails and copies use the driver.
mh_lifecycle(helper_ctx_fail helper_ctx_fail STAGING_PIPELINE_WORKERS=2
  STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64)
# A caller stream the fence cannot be recorded on: the copy goes to the driver.
mh_lifecycle(fence_fail fence_fail STAGING_THREADS_COPY_NUM_BUFFERS=2
  STAGING_BUFFER_SIZE=64)
# An install that cannot identify libcuda (no cuInit): every call still reaches the driver.
add_library(fake_cuda_no_init SHARED ${CMT_ROOT}/tests/fixtures/fake_libcuda.cpp)
target_compile_options(fake_cuda_no_init PRIVATE ${CMT_HOOK_FLAGS})
target_compile_definitions(fake_cuda_no_init PRIVATE FAKE_OMIT_CU_INIT=1)
set_target_properties(fake_cuda_no_init PROPERTIES OUTPUT_NAME cuda SUFFIX ".so.1"
  CXX_VISIBILITY_PRESET hidden LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fakecuda_no_init)
add_dependencies(test_ring_lifecycle fake_cuda_no_init)
add_test(NAME ring_install_failed COMMAND ${CMAKE_COMMAND} -E env
  LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}> LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda_no_init
  CMT_OVERRIDE=progressive_pinning=off STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  ${MH_RUN} $<TARGET_FILE:test_ring_lifecycle> install_failed)
set_tests_properties(ring_install_failed PROPERTIES TIMEOUT 60)

# Real-GPU semantics (${CMT_ROOT}/tests/test_gpu_semantics.cu): what the fake libcuda cannot show. Built only when the top-level
# project enabled CUDA (-DCMT_GPU_SEMANTICS_TESTS=ON; a standalone hook build needs no CUDA toolkit). Each binary runs twice: stock (the proof
# that every check is a property of CUDA itself) and hooked, under a table that sends EVERY size through the chunks
# pipeline. Exit 77 (skipped) without a GPU; RUN_SERIAL so they never share the GPU with another timing test.
# Requires CUDA enabled as a language, not merely found.
get_property(mh_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST mh_languages)
  find_package(CUDAToolkit REQUIRED)
  foreach(variant default ptds)
    set(exe test_gpu_semantics_${variant})
    add_executable(${exe} ${CMT_ROOT}/tests/test_gpu_semantics.cu ${CMT_ROOT}/tests/gpu_audit.cpp)
    # cuda_driver: the VMM allocation (cuMemCreate + cuMemMap) of the generic-copy checks. cmt_copying_config:
    # only for the audit Snapshot's headers (gpu_audit.cpp).
    target_link_libraries(${exe} PRIVATE CUDA::cudart CUDA::cuda_driver cmt_copying_config)
    if(variant STREQUAL "ptds")
      target_compile_definitions(${exe} PRIVATE HOOK_TEST_PTDS)
      target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    endif()
    add_test(NAME gpu_semantics_${variant}_stock COMMAND ${exe})
    add_test(NAME gpu_semantics_${variant}_hooked
      COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:cmt_hook>
      STAGING_PIPELINE_WORKERS=2 STAGING_THREADS_COPY_NUM_BUFFERS=3 STAGING_BUFFER_SIZE=64
      CMT_TEST_STAGED=1 $<TARGET_FILE:${exe}>)
    set_tests_properties(gpu_semantics_${variant}_stock gpu_semantics_${variant}_hooked
      PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu RUN_SERIAL TRUE)
  endforeach()
  # The generic-copy size gate needs a measured table of THIS machine (tables are per machine): give one with
  # -DCMT_GPU_SEMANTICS_TUNE_TABLE=<tune.json>, and a size that table stages in neither direction (its Driver rows;
  # 4 KiB is below the smallest staged row of every measured RTX 3090 table).
  set(CMT_GPU_SEMANTICS_TUNE_TABLE "" CACHE FILEPATH "tune.json of this machine for gpu_semantics_default_table")
  set(CMT_GPU_SEMANTICS_UNSTAGED_BYTES 4096 CACHE STRING "a copy size that table stages in neither direction")
  if(CMT_GPU_SEMANTICS_TUNE_TABLE)
    add_test(NAME gpu_semantics_default_table
      COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:cmt_hook> CMT_CONFIG=${CMT_GPU_SEMANTICS_TUNE_TABLE}
      CMT_TEST_UNSTAGED_BYTES=${CMT_GPU_SEMANTICS_UNSTAGED_BYTES} $<TARGET_FILE:test_gpu_semantics_default>)
    set_tests_properties(gpu_semantics_default_table PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu RUN_SERIAL TRUE)
  endif()
endif()

# Progressive pinning lifetime checks use the real immutable eligibility probe.
# Without the verified loaded kernel bundle, the executable reports an explicit skip.
add_executable(test_reg_cache_safety ${CMT_ROOT}/tests/test_reg_cache_safety.cpp)
target_compile_options(test_reg_cache_safety PRIVATE ${CMT_HOOK_FLAGS})
target_link_libraries(test_reg_cache_safety PRIVATE cmt_linux Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(test_reg_cache_safety fake_cuda cmt_hook fake_alloc)
set(MH_PINNING_ENV STAGING_THREADS_COPY_NUM_BUFFERS=2 STAGING_BUFFER_SIZE=64
  CMT_PINNING_MIN_KIB=1024 CMT_PINNING_STEP_MIB=1 CMT_PINNING_FIRST_MIB=1)
add_test(NAME regcache_reuse_munmap
  COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda ${MH_PINNING_ENV}
  ${MH_RUN} $<TARGET_FILE:test_reg_cache_safety> reuse munmap)
add_test(NAME regcache_stress
  COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>:$<TARGET_FILE:fake_alloc>"
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda ${MH_PINNING_ENV} CMT_PINNING_MAX_MIB=48
  ${MH_RUN} $<TARGET_FILE:test_reg_cache_safety> stress none)
# A sticky CUDA error while pinned copies wait for their DMA: the error is returned, the registrations are kept.
add_test(NAME regcache_sticky
  COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
  LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda ${MH_PINNING_ENV}
  ${MH_RUN} $<TARGET_FILE:test_reg_cache_safety> sticky none)
foreach(mode foreign hole)
  add_test(NAME regcache_typed_${mode}
    COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:${MH_PRELOAD}>
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda ${MH_PINNING_ENV}
    ${MH_RUN} $<TARGET_FILE:test_reg_cache_safety> ${mode} none)
endforeach()
set_tests_properties(regcache_reuse_munmap regcache_stress regcache_sticky regcache_typed_foreign
  regcache_typed_hole PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)

# These fixtures exercise registration directly, without a measured tuning profile.
# Auto requires a measured winning row; On still requires real kernel eligibility.
set_property(TEST e2e_progressive_pinning registration_owned_pinned registration_release_failure
  readonly_progressive readonly_default readonly_missing_api readonly_pinned_bypass readonly_release_reuse
  ring_stream_order_regfail regcache_reuse_munmap regcache_stress regcache_sticky regcache_typed_foreign
  regcache_typed_hole APPEND PROPERTY ENVIRONMENT "CMT_OVERRIDE=progressive_pinning=on")
# Progressive pinning needs the verified host-registration modules loaded (`cmt patch`); without them these 18 exit
# 77. `ctest -L kmod` runs exactly them on a patched host.
set_property(TEST e2e_progressive_pinning registration_owned_pinned registration_release_failure
  registration_limit_default registration_limit_env registration_limit_override registration_refusals
  readonly_progressive readonly_default readonly_missing_api readonly_pinned_bypass readonly_release_reuse
  ring_stream_order_regfail regcache_reuse_munmap regcache_stress regcache_sticky regcache_typed_foreign
  regcache_typed_hole APPEND PROPERTY LABELS kmod)

# A cross build with no emulator: every test above is built (that is the compile check) but cannot run here.
if(MH_CANNOT_RUN)
  get_property(mh_all_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${mh_all_tests} PROPERTIES DISABLED TRUE)
  message(STATUS "libcmt_hook: cross build without an emulator; ${CMT_ARCH} tests built and registered DISABLED")
endif()

# The default frontend checks exercise the complete preloaded library and fake driver.
get_property(_hook_tests DIRECTORY PROPERTY TESTS)
foreach(test IN LISTS _hook_tests)
  if(test MATCHES "^(e2e_|loader_|readonly_|registration_|regcache_|tune_|ring_|early_|gpu_semantics_)")
    set_property(TEST ${test} APPEND PROPERTY LABELS e2e)
  endif()
endforeach()

# Optional copy-distribution tracing: no CUDA hardware or supported patch table.
if(TARGET cmt AND NOT CMAKE_CROSSCOMPILING)
  add_library(fake_runtime_caller SHARED ${CMT_ROOT}/tests/fixtures/fake_libcuda.cpp)
  target_compile_definitions(fake_runtime_caller PRIVATE CMT_FAKE_IMPORT_CALLER=1)
  target_compile_options(fake_runtime_caller PRIVATE ${CMT_HOOK_FLAGS})
  target_link_libraries(fake_runtime_caller PRIVATE fake_cuda)
  set_target_properties(fake_runtime_caller PROPERTIES CXX_VISIBILITY_PRESET hidden)
  add_library(fake_runtime_alt SHARED ${CMT_ROOT}/tests/fixtures/fake_libcuda.cpp)
  target_compile_definitions(fake_runtime_alt PRIVATE CMT_FAKE_RUNTIME_ALT=1)
  target_compile_options(fake_runtime_alt PRIVATE ${CMT_HOOK_FLAGS})
  set_target_properties(fake_runtime_alt PROPERTIES CXX_VISIBILITY_PRESET hidden)
  add_dependencies(test_ring_lifecycle cmt_trace cmt fake_runtime_caller fake_runtime_alt)
  add_test(NAME trace_cli_fake_cuda COMMAND ${CMAKE_COMMAND} -E env
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fakecuda
    $<TARGET_FILE:test_ring_lifecycle> trace_launch $<TARGET_FILE:cmt> $<TARGET_FILE:cmt_trace> $<TARGET_FILE:fake_runtime_caller> $<TARGET_FILE:fake_runtime_alt>)
  set_tests_properties(trace_cli_fake_cuda PROPERTIES TIMEOUT 30 LABELS e2e)
endif()

# The report (evaluation/plot.py) on a fixture run: 77 (skipped, never passed) without python3 or matplotlib.
add_test(NAME plot COMMAND sh -c "command -v python3 >/dev/null || exit 77; exec python3 -B \"$0\""
  ${CMT_ROOT}/tests/test_plot.py)
set_tests_properties(plot PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
# sglang-rl-disk-update's untimed parts (evaluation/workloads/engines.py): trainer stand-in, head-swap verification,
# page-cache residency. Standard library only, no GPU; 77 (skipped, never passed) without python3.
add_test(NAME rl_disk_update COMMAND sh -c "command -v python3 >/dev/null || exit 77; exec python3 -B \"$0\""
  ${CMT_ROOT}/tests/test_rl_disk_update.py)
set_tests_properties(rl_disk_update PROPERTIES SKIP_RETURN_CODE 77)
# kmod.sh's journal, lock, rollback and catalog logic (kernel_patches/nvidia-open): no root, GPU or network;
# 77 (skipped, never passed) without python3.
add_test(NAME kmod_logic COMMAND sh -c "command -v python3 >/dev/null || exit 77; exec python3 -I -B \"$0\""
  ${CMT_ROOT}/tests/test_kmod.py)
set_tests_properties(kmod_logic PROPERTIES SKIP_RETURN_CODE 77)
# tools/libcuda_check.py's redirect guard (HTTPS only; the GitHub token never leaves its host). No network.
add_test(NAME libcuda_check_redirects COMMAND sh -c "command -v python3 >/dev/null || exit 77; exec python3 -I -B \"$0\""
  ${CMT_ROOT}/tests/test_libcuda_check.py)
set_tests_properties(libcuda_check_redirects PROPERTIES SKIP_RETURN_CODE 77)
# The runner's unit tests, with the cargo, rustc and crates the evaluation build already required.
if(CMT_BUILD_EVALUATION)
  add_test(NAME evaluation_units COMMAND ${CMAKE_COMMAND} -E env RUSTC=${CMT_RUSTC}
    ${CMT_CARGO} test --locked --offline --manifest-path ${CMT_ROOT}/evaluation/Cargo.toml
    --target-dir ${CMT_EVALUATION_TARGET_DIR})
  set_tests_properties(evaluation_units PROPERTIES TIMEOUT 1200)
  add_test(NAME evaluation_bundle COMMAND ${CMAKE_COMMAND} -DBUILD=${PROJECT_BINARY_DIR}
    -DPREFIX=${CMAKE_CURRENT_BINARY_DIR}/evaluation_bundle -P ${CMT_ROOT}/tests/test_evaluation_bundle.cmake)
endif()
