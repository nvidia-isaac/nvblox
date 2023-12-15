# Copy test data into the testing space (build/tests)
file(INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Copy some (python) visualization tools
file(CREATE_LINK
    "${CMAKE_CURRENT_SOURCE_DIR}/visualization/visualize_csv_depth_image.py"
    "${CMAKE_CURRENT_BINARY_DIR}/visualize_csv_depth_image.py"
    SYMBOLIC
)
file(CREATE_LINK
    "${CMAKE_CURRENT_SOURCE_DIR}/../visualization/visualize_mesh.py"
    "${CMAKE_CURRENT_BINARY_DIR}/visualize_mesh.py"
    SYMBOLIC
)

# A library containing functions used by many tests.
add_library(nvblox_test_utils SHARED
    lib/cuda/blox_utils.cu
    lib/cuda/blox.cu
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
    lib/projective_tsdf_integrator_cpu.cpp
    lib/utils.cpp
    lib/esdf_integrator_cpu.cpp
    lib/integrator_utils.cpp
)
target_include_directories(nvblox_test_utils PUBLIC
    include
    ${GTEST_INCLUDE_DIRS}
  )
target_compile_options(nvblox_test_utils PRIVATE ${TEST_COMPILE_OPTIONS})
target_link_libraries(nvblox_test_utils
    nvblox_lib nvblox_datasets ${GTEST_LIBRARIES} ${CUDA_LIBRARIES} pthread ${TEST_LINK_OPTIONS}
)
set_target_properties(nvblox_test_utils PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

set(TEST_OPTIONS DISCOVERY_TIMEOUT 30)

function (add_nvblox_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} nvblox_test_utils)
    # Sanitizer ASAN option to avoid crash during cuda init
    gtest_discover_tests(${test_name} ${TEST_OPTIONS} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    target_compile_options(${test_name} PRIVATE ${TEST_COMPILE_OPTIONS})
    target_link_options(${test_name} PRIVATE ${TEST_LINK_OPTIONS})
endfunction()

add_nvblox_test(test_3d_interpolation)
add_nvblox_test(test_3dmatch)
add_nvblox_test(test_blox)
add_nvblox_test(test_bounding_spheres)
add_nvblox_test(test_connected_components)
add_nvblox_test(test_cake)
add_nvblox_test(test_camera)
add_nvblox_test(test_color_image)
add_nvblox_test(test_color_integrator)
add_nvblox_test(test_cuda_stream)
add_nvblox_test(test_depth_image)
add_nvblox_test(test_dynamics)
add_nvblox_test(test_for_memory_leaks)
add_nvblox_test(test_freespace_integrator)
add_nvblox_test(test_frustum)
add_nvblox_test(test_fuser)
add_nvblox_test(test_gpu_layer_view)
add_nvblox_test(test_image_io)
add_nvblox_test(test_image_masker)
add_nvblox_test(test_image_projector)
add_nvblox_test(test_indexing)
add_nvblox_test(test_layer)
add_nvblox_test(test_lidar)
add_nvblox_test(test_lidar_integration)
add_nvblox_test(test_mapper)
add_nvblox_test(test_mesh_coloring)
add_nvblox_test(test_mesh)
add_nvblox_test(test_mesh_serializer)
add_nvblox_test(test_multi_mapper)
add_nvblox_test(test_nvtx_ranges)
add_nvblox_test(test_occupancy_decay)
add_nvblox_test(test_occupancy_integrator)
add_nvblox_test(test_pointcloud)
add_nvblox_test(test_ray_caster)
add_nvblox_test(test_scene)
add_nvblox_test(test_serialization)
add_nvblox_test(test_sphere_tracing)
add_nvblox_test(test_time)
add_nvblox_test(test_traits)
add_nvblox_test(test_tsdf_decay)
add_nvblox_test(test_tsdf_integrator)
add_nvblox_test(test_tsdf_integrator_cuda_components)
add_nvblox_test(test_unified_ptr)
add_nvblox_test(test_unified_vector)
add_nvblox_test(test_weighting_function)
add_nvblox_test(test_rates)
add_nvblox_test(test_mesh_streamer)
add_nvblox_test(test_npp_image_operations)
add_nvblox_test(test_depth_image_preprocessing)

# TODO(dtingdahl) Fix tests that fails under sanitizer
if(NOT USE_SANITIZER)
  add_nvblox_test(test_esdf_integrator)

  # Add test that prevents us from accidentally introducing work on the default cuda stream.
  # The test comes with a helper executable for for determining ID of the default cuda stream.
  if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    add_executable(run_memcpy_on_default_cuda_stream
      run_memcpy_on_default_cuda_stream.cpp
    )
    target_link_libraries(run_memcpy_on_default_cuda_stream ${CUDA_LIBRARIES} nvblox_lib)
    add_test(test_default_stream_utilization ${CMAKE_CURRENT_SOURCE_DIR}/check_default_stream_utilization.sh
    )
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
endif(NOT USE_SANITIZER)
