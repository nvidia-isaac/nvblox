# Datasets library
add_nvblox_shared_library(
  nvblox_datasets
  SOURCE_FILES
  src/3dmatch.cpp
  src/image_loader.cpp
  src/replica.cpp
  src/redwood.cpp
  src/cusfm_data.cpp
  src/lidarply_loader.cpp
  src/lidarply_writer.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)
