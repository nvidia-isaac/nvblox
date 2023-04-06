add_executable(test_bench
    src/test_bench.cpp
)
target_include_directories(test_bench PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(test_bench
    nvblox_lib nvblox_datasets
)
