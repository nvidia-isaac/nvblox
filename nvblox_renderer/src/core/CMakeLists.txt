# nvblox_renderer_core - Vulkan context, CUDA interop (no display dependencies)

add_nvblox_shared_library(
  nvblox_renderer_core
  SOURCE_FILES
  vk_utils.cpp
  vk_frame_sync.cpp
  vk_context.cpp
  shared_buffer.cu
  shared_texture.cu
  cuda_vulkan_semaphore.cu
  framebuffer_readback.cu
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  Vulkan::Vulkan
  nvblox_eigen
  CUDA::cudart
  glog::glog
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})

# Enable Vulkan validation layers in debug builds
target_compile_definitions(
  nvblox_renderer_core
  PRIVATE $<$<CONFIG:Debug>:ENABLE_VALIDATION_LAYERS>
          $<$<CONFIG:RelWithDebInfo>:ENABLE_VALIDATION_LAYERS>)
