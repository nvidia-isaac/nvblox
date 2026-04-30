# nvblox_renderer_targets - Render targets (headless and window output)

# TODO (Tao Jin): we can add CMake flag to choose the target type (headless or
# window)
add_nvblox_shared_library(
  nvblox_renderer_targets
  SOURCE_FILES
  vk_render_target_base.cpp
  vk_headless_target.cpp
  vk_window_target.cpp
  vk_window.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_renderer_core
  LINK_LIBRARIES_PRIVATE
  glfw
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})
