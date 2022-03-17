find_package(GTest REQUIRED)
enable_testing()

# Copy test data into the testing space (build/tests)
file(INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# A library containing functions used by many tests.
add_library(nvblox_test_utils SHARED
    lib/utils.cpp
    lib/projective_tsdf_integrator_cpu.cpp
    lib/cuda/increment_kernel.cu
    lib/cuda/gpu_image_routines.cu
    lib/cuda/interpolation_2d_gpu.cu
    lib/cuda/projective_tsdf_integrator_cuda_components.cu
    lib/cuda/test_utils_cuda.cu
    lib/cuda/blox.cu
    lib/cuda/blox_utils.cu
    lib/cuda/gpu_layer_utils.cu
    lib/cuda/gpu_indexing.cu
)
target_include_directories(nvblox_test_utils PUBLIC
  include
  ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(nvblox_test_utils 
    nvblox_lib ${GTEST_LIBRARIES} ${CUDA_LIBRARIES} pthread
)
set_target_properties(nvblox_test_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(TEST_OPTIONS DISCOVERY_TIMEOUT 30)

# The tests
add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera nvblox_test_utils)
gtest_discover_tests(test_camera ${TEST_OPTIONS})

add_executable(test_indexing test_indexing.cpp)
target_link_libraries(test_indexing nvblox_test_utils)
gtest_discover_tests(test_indexing ${TEST_OPTIONS})

add_executable(test_layer test_layer.cpp)
target_link_libraries(test_layer nvblox_test_utils)
gtest_discover_tests(test_layer ${TEST_OPTIONS})

add_executable(test_3d_interpolation test_3d_interpolation.cpp)
target_link_libraries(test_3d_interpolation nvblox_test_utils)
gtest_discover_tests(test_3d_interpolation ${TEST_OPTIONS})

add_executable(test_tsdf_integrator test_tsdf_integrator.cpp)
target_link_libraries(test_tsdf_integrator nvblox_test_utils)
gtest_discover_tests(test_tsdf_integrator ${TEST_OPTIONS})

add_executable(test_3dmatch test_3dmatch.cpp)
target_link_libraries(test_3dmatch nvblox_test_utils nvblox_lib)
gtest_discover_tests(test_3dmatch ${TEST_OPTIONS})
set_target_properties(test_3dmatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_unified_ptr test_unified_ptr.cpp)
target_link_libraries(test_unified_ptr nvblox_test_utils)
gtest_discover_tests(test_unified_ptr ${TEST_OPTIONS})

add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh nvblox_test_utils)
gtest_discover_tests(test_mesh ${TEST_OPTIONS})

add_executable(test_scene test_scene.cpp)
target_link_libraries(test_scene nvblox_test_utils)
gtest_discover_tests(test_scene ${TEST_OPTIONS})

add_executable(test_depth_image test_depth_image.cpp)
target_link_libraries(test_depth_image nvblox_test_utils)
gtest_discover_tests(test_depth_image ${TEST_OPTIONS})

add_executable(test_tsdf_integrator_cuda_components test_tsdf_integrator_cuda_components.cpp)
target_link_libraries(test_tsdf_integrator_cuda_components nvblox_test_utils)
gtest_discover_tests(test_tsdf_integrator_cuda_components ${TEST_OPTIONS})

add_executable(test_unified_vector test_unified_vector.cpp)
target_link_libraries(test_unified_vector nvblox_test_utils)
gtest_discover_tests(test_unified_vector ${TEST_OPTIONS})

add_executable(test_esdf_integrator test_esdf_integrator.cpp)
target_link_libraries(test_esdf_integrator nvblox_test_utils)
gtest_discover_tests(test_esdf_integrator ${TEST_OPTIONS})

add_executable(test_color_image test_color_image.cpp)
target_link_libraries(test_color_image nvblox_test_utils)
gtest_discover_tests(test_color_image ${TEST_OPTIONS})

add_executable(test_color_integrator test_color_integrator.cpp)
target_link_libraries(test_color_integrator nvblox_test_utils)
gtest_discover_tests(test_color_integrator ${TEST_OPTIONS})

add_executable(test_mesh_coloring test_mesh_coloring.cpp)
target_link_libraries(test_mesh_coloring nvblox_test_utils)
gtest_discover_tests(test_mesh_coloring ${TEST_OPTIONS})

add_executable(test_for_memory_leaks test_for_memory_leaks.cpp)
target_link_libraries(test_for_memory_leaks nvblox_test_utils)
gtest_discover_tests(test_for_memory_leaks ${TEST_OPTIONS})

add_executable(test_frustum test_frustum.cpp)
target_link_libraries(test_frustum nvblox_test_utils)
gtest_discover_tests(test_frustum ${TEST_OPTIONS})

add_executable(test_gpu_layer_view test_gpu_layer_view.cpp)
target_link_libraries(test_gpu_layer_view nvblox_test_utils)
gtest_discover_tests(test_gpu_layer_view ${TEST_OPTIONS})

add_executable(test_sphere_tracing test_sphere_tracing.cpp)
target_link_libraries(test_sphere_tracing nvblox_test_utils)
gtest_discover_tests(test_sphere_tracing ${TEST_OPTIONS})

add_executable(test_cake test_cake.cpp)
target_link_libraries(test_cake nvblox_test_utils)
gtest_discover_tests(test_cake ${TEST_OPTIONS})

add_executable(test_traits test_traits.cpp)
target_link_libraries(test_traits nvblox_test_utils)
gtest_discover_tests(test_traits ${TEST_OPTIONS})
