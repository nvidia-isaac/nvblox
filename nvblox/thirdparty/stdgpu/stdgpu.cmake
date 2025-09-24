if(USE_SYSTEM_STDGPU)
  find_package(stdgpu REQUIRED)
  add_library(nvblox_stdgpu ALIAS stdgpu::stdgpu)
else()
  include(FetchContent)

  # Patches to stdgpu
  set(apply_patch
      git
      apply
      # Patch that overrides the estimated number of hash collissions by stdgpu
      # and sets the worst-case number. This is necessary to ensure stability in
      # case of an extraordinary amount of collisions.
      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdgpu/stdgpu_handle_collisions.patch
      # Patch that fixes a cmake error in Findthrust in later versions of cmake.
      # This error has been fixed in more recent version of stdgpu
      # (https://github.com/stotko/stdgpu/pull/408)
      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdgpu/stdgpu_thrust_version_regex.patch
      # Patch that exposes the "occupied" array. We need this when copying the
      # hash.
      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdgpu/stdgpu_expose_occupied.patch
      # Patch that prepends stdgpu namespace to conflicting cuda functions.
      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdgpu/stdgpu_fix_cuda12_6.patch
      # Patch that fixes deprecated calls in CUDA13
      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdgpu/stdgpu_fix_cuda13_0.patch)

  FetchContent_Declare(
    ext_stdgpu
    SYSTEM
    PREFIX
    stdgpu
    GIT_REPOSITORY https://github.com/stotko/stdgpu.git
    GIT_TAG 71a5aef26626eda47d15e5f577ca3b1538ff996a
    PATCH_COMMAND ${apply_patch}
    UPDATE_COMMAND "")

  # stdgpu build options
  set(STDGPU_BUILD_SHARED_LIBS OFF)
  set(STDGPU_BUILD_EXAMPLES OFF)
  set(STDGPU_BUILD_TESTS OFF)
  set(STDGPU_ENABLE_CONTRACT_CHECKS OFF)
  set(STDGPU_BUILD_BENCHMARKS OFF)

  # Download the files
  FetchContent_MakeAvailable(ext_stdgpu)

  # Apply nvblox compile options to exported targets
  set_nvblox_compiler_options_nowarnings(stdgpu)
  add_library(nvblox_stdgpu INTERFACE)
  target_link_libraries(nvblox_stdgpu INTERFACE stdgpu)
  target_include_directories(
    nvblox_stdgpu INTERFACE $<BUILD_INTERFACE:${ext_stdgpu_SOURCE_DIR}/src>
                            $<INSTALL_INTERFACE:include/stdgpu>)
endif()
