# nvblox_renderer_kernels - CUDA data transformation kernels

add_nvblox_shared_library(
  nvblox_renderer_kernels
  SOURCE_FILES
  depth_to_pointcloud.cu
  mesh_to_vertex.cu
  LINK_LIBRARIES_PUBLIC
  nvblox_renderer_core
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})
