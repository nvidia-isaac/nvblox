# ESDF query executable
add_executable(esdf_query
    src/esdf_query.cu
)
target_link_libraries(esdf_query
    nvblox_lib
)
set_target_properties(esdf_query PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# An example for map loading
add_executable(load_map_and_mesh
    src/load_map_and_mesh.cpp
)
target_link_libraries(load_map_and_mesh
    nvblox_lib
)

set_target_properties(load_map_and_mesh PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
