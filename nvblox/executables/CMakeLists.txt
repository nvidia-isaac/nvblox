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
# 3Dmatch executable
add_executable(fuse_3dmatch
	src/fuse_3dmatch.cpp
	src/datasets/3dmatch.cpp
	src/fuser_rgbd.cpp
)
target_link_libraries(fuse_3dmatch
	nvblox_lib nvblox_datasets
)
set_target_properties(fuse_3dmatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Replica executable
add_executable(fuse_replica
	src/fuse_replica.cpp
	src/datasets/replica.cpp
	src/fuser_rgbd.cpp
)
target_link_libraries(fuse_replica
	nvblox_lib nvblox_datasets
)
set_target_properties(fuse_replica PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# FusionPortable executable
add_executable(fuse_fusionportable
    src/fuse_fusionportable.cpp
    src/fuser_lidar.cpp
    src/datasets/fusionportable.cpp
)
target_link_libraries(fuse_fusionportable
    nvblox_lib nvblox_datasets
)
set_target_properties(fuse_fusionportable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# KITTI executable
# add_executable(fuse_kitti
#     src/fuse_kitti.cpp
#     src/fuser_lidar.cpp
#     src/datasets/fusionportable.cpp
# )
# target_link_libraries(fuse_kitti
#     nvblox_lib nvblox_datasets
# )
# set_target_properties(fuse_kitti PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
