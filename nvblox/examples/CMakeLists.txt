# ESDF query executable
add_executable(esdf_query
    src/esdf_query.cu
)
target_link_libraries(esdf_query
    nvblox_lib
)
set_target_properties(esdf_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
