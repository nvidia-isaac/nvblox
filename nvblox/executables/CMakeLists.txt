# Datasets library
add_library(nvblox_datasets SHARED
    src/datasets/3dmatch.cpp
    src/datasets/image_loader.cpp
    src/datasets/replica.cpp
    src/datasets/redwood.cpp
    src/fuser.cpp
)
target_include_directories(nvblox_datasets PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(nvblox_datasets nvblox_lib)
set_target_properties(nvblox_datasets PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


# 3Dmatch executable
add_executable(fuse_3dmatch
    src/fuse_3dmatch.cpp
)
target_link_libraries(fuse_3dmatch
    nvblox_lib nvblox_datasets
)
set_target_properties(fuse_3dmatch PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Replica executable
add_executable(fuse_replica
    src/fuse_replica.cpp
)
target_link_libraries(fuse_replica
    nvblox_lib nvblox_datasets
)
set_target_properties(fuse_replica PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Redwood executable
add_executable(fuse_redwood
    src/fuse_redwood.cpp
)
target_link_libraries(fuse_redwood
    nvblox_lib nvblox_datasets
)
set_target_properties(fuse_redwood PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Benchmarking executable
add_executable(benchmark
    src/benchmark.cpp
)
target_link_libraries(benchmark
    nvblox_lib nvblox_datasets nvblox_test_utils benchmark::benchmark
)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
