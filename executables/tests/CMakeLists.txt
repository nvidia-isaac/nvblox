include(GoogleTest)

# Copy test data into the build directory
file(INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/data"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

set(TEST_OPTIONS DISCOVERY_TIMEOUT 30)

function(add_fuser_test test_name file_extension)
  add_nvblox_executable(
    ${test_name}
    SOURCE_FILES
    ${test_name}.${file_extension}
    LINK_LIBRARIES_PUBLIC
    nvblox_fuser
    nvblox_test_utils
    LINK_LIBRARIES_PRIVATE
    nvblox_gpu_hash
    nvblox_stdgpu)

  gtest_discover_tests(
    ${test_name} ${TEST_OPTIONS}
    PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0"
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_SOURCE_DIR}
      DISCOVERY_MODE
      PRE_TEST)

  target_compile_options(${test_name} PRIVATE -Wno-maybe-uninitialized)
endfunction()

function(add_fuser_cpp_test test_name)
  add_fuser_test(${test_name} cpp)
endfunction()

function(add_fuser_cuda_test test_name)
  add_fuser_test(${test_name} cu)
endfunction()

# Tests that require the fuser goes here
add_fuser_cpp_test(test_3dmatch)
add_fuser_cpp_test(test_replica)
add_fuser_cpp_test(test_lidarply)

# We exclude certain tests from running wiht the gcc-sanitizer
if(NOT USE_SANITIZER)

  # These tests are too slow to run under sanitizer and are disabled to keep CI
  # times sane.
  add_fuser_cpp_test(test_fuser)
  add_fuser_cuda_test(test_lidar_motion_compensation)

  # Add test that prevents us from accidentally introducing work on the default
  # cuda stream. The test comes with a helper executable for for determining ID
  # of the default cuda stream.
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    add_nvblox_executable(
      run_memcpy_on_default_cuda_stream SOURCE_FILES
      run_memcpy_on_default_cuda_stream.cpp LINK_LIBRARIES_PUBLIC nvblox_lib)
    add_test(test_default_stream_utilization bash
             ${CMAKE_CURRENT_SOURCE_DIR}/check_default_stream_utilization.sh)
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
endif()
