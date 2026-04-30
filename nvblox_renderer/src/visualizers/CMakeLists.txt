# nvblox_renderer_visualizers - Rendering pipelines

add_nvblox_shared_library(
  nvblox_renderer_visualizers
  SOURCE_FILES
  base_visualizer.cpp
  pipeline_builder.cpp
  point_cloud_visualizer.cpp
  image_visualizer.cpp
  mesh_visualizer.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_renderer_core
  nvblox_renderer_utils
  nvblox_renderer_kernels
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})

# Visualizers load shaders at runtime, ensure they're compiled before build
# completes
add_dependencies(nvblox_renderer_visualizers shaders)
