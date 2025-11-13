# Datasets library
add_nvblox_shared_library(
  nvblox_datasets
  SOURCE_FILES
  src/datasets/3dmatch.cpp
  src/datasets/image_loader.cpp
  src/datasets/replica.cpp
  src/datasets/redwood.cpp
  src/datasets/cusfm_data.cpp
  src/fuser.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# 3Dmatch executable
add_nvblox_executable(
  fuse_3dmatch
  SOURCE_FILES
  src/fuse_3dmatch.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_datasets
  nvblox_stdgpu
  nvblox_gpu_hash)

# Replica executable
add_nvblox_executable(
  fuse_replica
  SOURCE_FILES
  src/fuse_replica.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  nvblox_datasets
  nvblox_stdgpu
  nvblox_gpu_hash)

# Redwood executable
add_nvblox_executable(
  fuse_redwood
  SOURCE_FILES
  src/fuse_redwood.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  nvblox_datasets
  nvblox_stdgpu
  nvblox_gpu_hash)

# CUSFM executable
add_nvblox_executable(
  fuse_cusfm
  SOURCE_FILES
  src/fuse_cusfm.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  nvblox_datasets
  nvblox_stdgpu
  nvblox_gpu_hash)
