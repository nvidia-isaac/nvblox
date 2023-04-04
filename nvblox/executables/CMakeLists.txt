include_directories(include)

# Datasets library
add_library(nvblox_datasets SHARED
	src/datasets/image_loader.cpp
)
target_include_directories(nvblox_datasets PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

target_link_libraries(nvblox_datasets nvblox_lib)
set_target_properties(nvblox_datasets PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

####################################
#### NOTE(gogojjh): compile libraries for datasets
add_library(nvblox_datasets_3dmatch SHARED
	src/datasets/3dmatch.cpp
	src/fuser_rgbd.cpp
)
target_link_libraries(nvblox_datasets_3dmatch nvblox_lib nvblox_datasets)

add_library(nvblox_datasets_replica SHARED
	src/datasets/replica.cpp
	src/fuser_rgbd.cpp
)
target_link_libraries(nvblox_datasets_replica nvblox_lib nvblox_datasets)

add_library(nvblox_datasets_fusionportable SHARED
	src/datasets/fusionportable.cpp
	src/fuser_lidar.cpp
)
target_link_libraries(nvblox_datasets_fusionportable nvblox_lib nvblox_datasets)

add_library(nvblox_datasets_kitti SHARED
	src/datasets/kitti.cpp
	src/fuser_lidar.cpp
)
target_link_libraries(nvblox_datasets_kitti nvblox_lib nvblox_datasets)

####################################
# 3Dmatch executable
add_executable(fuse_3dmatch	src/fuse_3dmatch.cpp)
target_link_libraries(fuse_3dmatch
	nvblox_lib nvblox_datasets nvblox_datasets_3dmatch
)
set_target_properties(fuse_3dmatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Replica executable
add_executable(fuse_replica	src/fuse_replica.cpp)
target_link_libraries(fuse_replica
	nvblox_lib nvblox_datasets nvblox_datasets_replica
)
set_target_properties(fuse_replica PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# FusionPortable executable
add_executable(fuse_fusionportable src/fuse_fusionportable.cpp)
target_link_libraries(fuse_fusionportable
	nvblox_lib nvblox_datasets nvblox_datasets_fusionportable
)
set_target_properties(fuse_fusionportable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# KITTI executable
add_executable(fuse_kitti	src/fuse_kitti.cpp)
target_link_libraries(fuse_kitti
	nvblox_lib nvblox_datasets nvblox_datasets_kitti
)
set_target_properties(fuse_kitti PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
