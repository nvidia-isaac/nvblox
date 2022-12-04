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
    lib/cuda/mesh_utils.cu
    lib/cuda/projective_tsdf_integrator_cuda_components.cu
    lib/cuda/test_utils_cuda.cu
    lib/projective_tsdf_integrator_cpu.cpp
    lib/utils.cpp
)
target_include_directories(nvblox_test_utils PUBLIC
  include
  ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(nvblox_test_utils 
    nvblox_lib nvblox_datasets ${GTEST_LIBRARIES} ${CUDA_LIBRARIES} pthread
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
target_link_libraries(test_3dmatch nvblox_test_utils nvblox_lib nvblox_datasets_3dmatch)
gtest_discover_tests(test_3dmatch ${TEST_OPTIONS})
set_target_properties(test_3dmatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_unified_ptr test_unified_ptr.cpp)
target_link_libraries(test_unified_ptr nvblox_test_utils)
gtest_discover_tests(test_unified_ptr ${TEST_OPTIONS})

add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh nvblox_test_utils nvblox_datasets_3dmatch)
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
target_link_libraries(test_color_image nvblox_test_utils nvblox_datasets_3dmatch)
gtest_discover_tests(test_color_image ${TEST_OPTIONS})

add_executable(test_color_integrator test_color_integrator.cpp)
target_link_libraries(test_color_integrator nvblox_test_utils)
gtest_discover_tests(test_color_integrator ${TEST_OPTIONS})

add_executable(test_mesh_coloring test_mesh_coloring.cpp)
target_link_libraries(test_mesh_coloring nvblox_test_utils nvblox_datasets_3dmatch)
gtest_discover_tests(test_mesh_coloring ${TEST_OPTIONS})

add_executable(test_for_memory_leaks test_for_memory_leaks.cpp)
target_link_libraries(test_for_memory_leaks nvblox_test_utils nvblox_datasets_3dmatch)
gtest_discover_tests(test_for_memory_leaks ${TEST_OPTIONS})

add_executable(test_frustum test_frustum.cpp)
target_link_libraries(test_frustum nvblox_test_utils nvblox_datasets_3dmatch)
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

add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization nvblox_test_utils)
gtest_discover_tests(test_serialization ${TEST_OPTIONS})

add_executable(test_lidar test_lidar.cpp)
target_link_libraries(test_lidar nvblox_test_utils)
gtest_discover_tests(test_lidar ${TEST_OPTIONS})

add_executable(test_lidar_integration test_lidar_integration.cpp)
target_link_libraries(test_lidar_integration nvblox_test_utils)
gtest_discover_tests(test_lidar_integration ${TEST_OPTIONS})

### NOTE(gogojjh): fuser is already replaced by the fuser_rgbd, no need to test_fuser
# add_executable(test_fuser test_fuser.cpp)
# target_link_libraries(test_fuser nvblox_test_utils)
# gtest_discover_tests(test_fuser ${TEST_OPTIONS})

add_executable(test_fuser_rgbd test_fuser_rgbd.cpp)
target_link_libraries(test_fuser_rgbd nvblox_test_utils nvblox_datasets_3dmatch)
gtest_discover_tests(test_fuser_rgbd ${TEST_OPTIONS})

add_executable(test_bounding_spheres test_bounding_spheres.cpp)
target_link_libraries(test_bounding_spheres nvblox_test_utils)
gtest_discover_tests(test_bounding_spheres ${TEST_OPTIONS})

add_executable(test_mapper test_mapper.cpp)
target_link_libraries(test_mapper nvblox_test_utils)
gtest_discover_tests(test_mapper ${TEST_OPTIONS})

### NOTE(gogojjh): need to rewrite the test_oslidar to adjust the new oslidar model
# add_executable(test_oslidar test_oslidar.cpp)
# target_link_libraries(test_oslidar nvblox_test_utils)
# gtest_discover_tests(test_oslidar ${TEST_OPTIONS})

#### camera_pinhole
add_executable(test_camera_pinhole test_camera_pinhole.cpp)
target_link_libraries(test_camera_pinhole nvblox_test_utils)
gtest_discover_tests(test_camera_pinhole ${TEST_OPTIONS})
