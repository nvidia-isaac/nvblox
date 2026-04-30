# nvblox_renderer tests
#
# Uses add_nvblox_executable to get consistent compiler options.

include(GoogleTest)

# Options shared by all renderer tests. ASan's shadow memory conflicts with CUDA
# runtime initialization; protect_shadow_gap=0 matches the workaround in
# nvblox/tests/CMakeLists.txt.
set(COMMON_TEST_OPTIONS DISCOVERY_TIMEOUT 30 PROPERTIES ENVIRONMENT
                        "ASAN_OPTIONS=protect_shadow_gap=0")

# Vulkan-dependent tests get an additional label so CI stages without graphics
# drivers can exclude them via: ctest -LE vulkan_capable
set(VULKAN_TEST_OPTIONS
    DISCOVERY_TIMEOUT
    30
    PROPERTIES
    ENVIRONMENT
    "ASAN_OPTIONS=protect_shadow_gap=0"
    LABELS
    "vulkan_capable")

# ==============================================================================
# VkFrameSync test - links to core module
# ==============================================================================
add_nvblox_executable(
  test_vk_frame_sync
  SOURCE_FILES
  test_vk_frame_sync.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_vk_frame_sync ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# VkContext test - Tests VkContext lifecycle, accessors, and frame management
# ==============================================================================
add_nvblox_executable(
  test_vk_context
  SOURCE_FILES
  test_vk_context.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_vk_context ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# VkUtils test - Tests vk_utils and error_check utilities
# ==============================================================================
add_nvblox_executable(
  test_vk_utils
  SOURCE_FILES
  test_vk_utils.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_vk_utils ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# SharedBuffer test - CUDA, links to core module
# ==============================================================================
add_nvblox_executable(
  test_shared_buffer
  SOURCE_FILES
  test_shared_buffer.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_shared_buffer ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# SharedTexture test - CUDA, links to core module
# ==============================================================================
add_nvblox_executable(
  test_shared_texture
  SOURCE_FILES
  test_shared_texture.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_shared_texture ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# CudaVulkanSemaphore test - CUDA, links to core module
# ==============================================================================
add_nvblox_executable(
  test_cuda_vulkan_semaphore
  SOURCE_FILES
  test_cuda_vulkan_semaphore.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_cuda_vulkan_semaphore ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# ViewCamera test - CPU only, links to utils module
# ==============================================================================
add_nvblox_executable(
  test_view_camera
  SOURCE_FILES
  test_view_camera.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_utils
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_view_camera ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# ShaderUtils test - CPU only, links to utils and core modules
# ==============================================================================
add_nvblox_executable(
  test_shader_utils
  SOURCE_FILES
  test_shader_utils.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_utils
  nvblox_renderer_core
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_shader_utils ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# RendererConstants test - CPU only, tests utility functions
# ==============================================================================
add_nvblox_executable(
  test_buffer_utils
  SOURCE_FILES
  test_buffer_utils.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_utils
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_buffer_utils ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# PushConstants test - Tests push constant struct sizes and layouts
# ==============================================================================
add_nvblox_executable(
  test_push_constants
  SOURCE_FILES
  test_push_constants.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_utils
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_push_constants ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# PipelineBuilder test - CPU only, links to visualizers module
# ==============================================================================
add_nvblox_executable(
  test_pipeline_builder
  SOURCE_FILES
  test_pipeline_builder.cpp
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_visualizers
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_pipeline_builder ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Kernel tests - CUDA only, link to kernels module
# ==============================================================================
add_nvblox_executable(
  test_depth_to_pointcloud
  SOURCE_FILES
  test_depth_to_pointcloud.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_kernels
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_depth_to_pointcloud ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# PointCloudVisualizer test - CUDA, links to visualizers module
# ==============================================================================
add_nvblox_executable(
  test_point_cloud_visualizer
  SOURCE_FILES
  test_point_cloud_visualizer.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_visualizers
  nvblox_renderer_targets
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_point_cloud_visualizer ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# ImageVisualizer test - CUDA, links to visualizers module
# ==============================================================================
add_nvblox_executable(
  test_image_visualizer
  SOURCE_FILES
  test_image_visualizer.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_visualizers
  nvblox_renderer_targets
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_image_visualizer ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# MeshToVertex kernel test - CUDA, links to kernels module
# ==============================================================================
add_nvblox_executable(
  test_mesh_to_vertex
  SOURCE_FILES
  test_mesh_to_vertex.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_kernels
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_mesh_to_vertex ${COMMON_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Visualizer comprehensive tests - CUDA, links to visualizers + targets
# ==============================================================================
add_nvblox_executable(
  test_visualizers
  SOURCE_FILES
  test_visualizers.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer_visualizers
  nvblox_renderer_targets
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_visualizers ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Headless Renderer Integration test - CUDA, links to main renderer
# ==============================================================================
add_nvblox_executable(
  test_headless_renderer
  SOURCE_FILES
  test_headless_renderer.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_headless_renderer ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# End-to-End Pipeline tests - CUDA, links to main renderer
# ==============================================================================
add_nvblox_executable(
  test_e2e_pipeline
  SOURCE_FILES
  test_e2e_pipeline.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer
  nvblox_test_utils
  gtest)
gtest_discover_tests(test_e2e_pipeline ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Window Renderer tests - CUDA, requires display
# ==============================================================================
add_nvblox_executable(
  test_window_renderer
  SOURCE_FILES
  test_window_renderer.cu
  LINK_LIBRARIES_PRIVATE
  nvblox_renderer
  nvblox_test_utils
  gtest
  glfw)
gtest_discover_tests(test_window_renderer ${VULKAN_TEST_OPTIONS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
