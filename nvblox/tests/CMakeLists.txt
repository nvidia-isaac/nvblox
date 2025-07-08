# Copy test data into the testing space (build/tests)
include(GoogleTest)

file(INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/data"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Copy some (python) visualization tools
file(CREATE_LINK
     "${CMAKE_CURRENT_SOURCE_DIR}/visualization/visualize_csv_depth_image.py"
     "${CMAKE_CURRENT_BINARY_DIR}/visualize_csv_depth_image.py" SYMBOLIC)
file(CREATE_LINK
     "${CMAKE_CURRENT_SOURCE_DIR}/../visualization/visualize_mesh.py"
     "${CMAKE_CURRENT_BINARY_DIR}/visualize_mesh.py" SYMBOLIC)

# A library containing functions used by many tests.
add_nvblox_static_library(
  nvblox_test_utils
  SOURCE_FILES
  lib/cuda/blox_utils.cu
  lib/cuda/index_layer_specialization.cu
  lib/cuda/boolean_layer_specialization.cu
  lib/cuda/float_layer_specialization.cu
  lib/cuda/initialization_test_layer_specialization.cu
  lib/cuda/gpu_image_routines.cu
  lib/cuda/gpu_indexing.cu
  lib/cuda/gpu_layer_utils.cu
  lib/cuda/increment_kernel.cu
  lib/cuda/interpolation_2d_gpu.cu
  lib/cuda/layer.cu
  lib/cuda/mesh_utils.cu
  lib/cuda/projective_tsdf_integrator_cuda_components.cu
  lib/cuda/test_utils_cuda.cu
  lib/cuda/weighting_utils.cu
  lib/cuda/test_layer.cu
  lib/projective_tsdf_integrator_cpu.cpp
  lib/tsdf_zero_crossings_extractor_cpu.cpp
  lib/utils.cpp
  lib/esdf_integrator_cpu.cpp
  lib/integrator_utils.cpp
  lib/ransac_plane_fitter_utils.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  nvblox_datasets
  gtest
  LINK_LIBRARIES_PRIVATE
  nvblox_gpu_hash
  nvblox_stdgpu
  INCLUDE_DIRECTORIES_PUBLIC
  include)

set(TEST_OPTIONS DISCOVERY_TIMEOUT 30)

function(add_nvblox_test test_name is_cuda_test)
  # Switch between .cpp and .cu tests.
  set(test_file_extension "cpp")
  if(${is_cuda_test})
    set(test_file_extension "cu")
  endif()
  # Build the test
  add_nvblox_executable(
    ${test_name}
    SOURCE_FILES
    ${test_name}.${test_file_extension}
    LINK_LIBRARIES_PUBLIC
    nvblox_test_utils
    LINK_LIBRARIES_PRIVATE
    nvblox_gpu_hash
    nvblox_stdgpu)

  # Sanitizer ASAN option to avoid crash during cuda init
  gtest_discover_tests(
    ${test_name} ${TEST_OPTIONS} PROPERTIES ENVIRONMENT
                                            "ASAN_OPTIONS=protect_shadow_gap=0"
					    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # gcc13 discovers warnings in third-party libs which we suppress here.
  target_compile_options(${test_name} PRIVATE -Wno-maybe-uninitialized)
endfunction()

function(add_nvblox_cpp_test test_name)
  add_nvblox_test(${test_name} FALSE)
endfunction()

function(add_nvblox_cuda_test test_name)
  add_nvblox_test(${test_name} TRUE)
endfunction()

add_nvblox_cpp_test(test_3d_interpolation)
add_nvblox_cpp_test(test_3dmatch)
add_nvblox_cpp_test(test_blox)
add_nvblox_cpp_test(test_shape_clearer)
add_nvblox_cpp_test(test_bounding_spheres)
add_nvblox_cpp_test(test_mask_preprocessor)
add_nvblox_cpp_test(test_mask_from_detections)
add_nvblox_cpp_test(test_cake)
add_nvblox_cpp_test(test_camera)
add_nvblox_cpp_test(test_color_image)
add_nvblox_cpp_test(test_color_integrator)
add_nvblox_cpp_test(test_feature_integrator)
add_nvblox_cpp_test(test_cuda_stream)
add_nvblox_cpp_test(test_mono_image)
add_nvblox_cpp_test(test_depth_image)
add_nvblox_cpp_test(test_dynamics)
add_nvblox_cpp_test(test_for_memory_leaks)
add_nvblox_cpp_test(test_freespace_integrator)
add_nvblox_cpp_test(test_frustum)
add_nvblox_cpp_test(test_gpu_layer_view)
add_nvblox_cpp_test(test_image_io)
add_nvblox_cpp_test(test_image_masker)
add_nvblox_cpp_test(test_image_projector)
add_nvblox_cpp_test(test_indexing)
add_nvblox_cpp_test(test_interpolation_2d)
add_nvblox_cpp_test(test_layer)
add_nvblox_cpp_test(test_lidar)
add_nvblox_cpp_test(test_lidar_integration)
add_nvblox_cpp_test(test_mapper)
add_nvblox_cpp_test(test_mesh_appearance)
add_nvblox_cpp_test(test_mesh)
add_nvblox_cpp_test(test_mesh_serializer)
add_nvblox_cpp_test(test_nvtx_ranges)
add_nvblox_cpp_test(test_occupancy_decay)
add_nvblox_cpp_test(test_occupancy_integrator)
add_nvblox_cpp_test(test_pointcloud)
add_nvblox_cpp_test(test_ray_caster)
add_nvblox_cpp_test(test_scene)
add_nvblox_cpp_test(test_serialization)
add_nvblox_cpp_test(test_sphere_tracing)
add_nvblox_cpp_test(test_time)
add_nvblox_cpp_test(test_timing)
add_nvblox_cpp_test(test_traits)
add_nvblox_cpp_test(test_tsdf_decay)
add_nvblox_cpp_test(test_tsdf_error)
add_nvblox_cpp_test(test_tsdf_integrator)
add_nvblox_cpp_test(test_tsdf_integrator_cuda_components)
add_nvblox_cpp_test(test_unified_3d_grid)
add_nvblox_cpp_test(test_unified_ptr)
add_nvblox_cpp_test(test_unified_vector)
add_nvblox_cpp_test(test_weighting_function)
add_nvblox_cpp_test(test_workspace_bounds)
add_nvblox_cpp_test(test_rates)
add_nvblox_cpp_test(test_layer_streamer)
add_nvblox_cpp_test(test_npp_image_operations)
add_nvblox_cpp_test(test_depth_image_preprocessing)
add_nvblox_cpp_test(test_layer_serializer_gpu)
add_nvblox_cpp_test(test_image_cache)
add_nvblox_cpp_test(test_params)
add_nvblox_cpp_test(test_block_memory_pool)
add_nvblox_cpp_test(test_delays)
add_nvblox_cpp_test(test_image_view)
add_nvblox_cpp_test(test_bitmask)
add_nvblox_cpp_test(test_nvblox_h)
add_nvblox_cpp_test(test_ransac_plane_fitter_cpu)
add_nvblox_cpp_test(test_ransac_plane_fitter)
add_nvblox_cpp_test(test_layer_cake_streamer)
add_nvblox_cpp_test(test_replica)
add_nvblox_cpp_test(test_checks)
add_nvblox_cuda_test(regression_test_query_after_clear)
add_nvblox_cuda_test(test_layer_to_3d_grid)
add_nvblox_cuda_test(test_gpu_hash_interface)
add_nvblox_cuda_test(test_masked_image_view)
add_nvblox_cuda_test(test_esdf_integrator_slicing)
add_nvblox_cuda_test(test_plane)
add_nvblox_cuda_test(test_device_function_utils)
add_nvblox_cuda_test(test_zero_crossings_extractor)
add_nvblox_cuda_test(test_error_check)
add_nvblox_cuda_test(test_array)

# The error check tests some scenarios that we're normally not going to
# encounter. Relevant warnings are disabled
target_compile_options(
  test_error_check PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550,187>")

if(NOT USE_SANITIZER)
  # TODO(dtingdahl) Fix tests that fails under sanitizer
  add_nvblox_cpp_test(test_esdf_integrator)

  # Since CUDA 12.6 these tests fail with an exception "invalid device ordinal".
  # The exception comes from cub and is thust not an asan error. It is possible
  # that CUDA 12.6 introduced further incompatibilies with asan. TODO(dtingdahl)
  # investigate
  add_nvblox_cpp_test(test_fuser)
  add_nvblox_cpp_test(test_mapper_block_allocation)
  add_nvblox_cpp_test(test_multi_mapper)

  # Add test that prevents us from accidentally introducing work on the default
  # cuda stream. The test comes with a helper executable for for determining ID
  # of the default cuda stream.
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    add_nvblox_executable(
      run_memcpy_on_default_cuda_stream SOURCE_FILES
      run_memcpy_on_default_cuda_stream.cpp LINK_LIBRARIES_PUBLIC nvblox_lib)
    add_test(test_default_stream_utilization
             ${CMAKE_CURRENT_SOURCE_DIR}/check_default_stream_utilization.sh)
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
endif(NOT USE_SANITIZER)

# Benchmarking executable
add_nvblox_executable(
  nvblox_benchmark SOURCE_FILES benchmark.cpp LINK_LIBRARIES_PUBLIC
  nvblox_test_utils benchmark::benchmark)
