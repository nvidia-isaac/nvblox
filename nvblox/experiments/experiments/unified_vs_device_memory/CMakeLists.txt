add_executable(unified_vs_device_memory main.cpp)
target_link_libraries(unified_vs_device_memory nvblox_experiments_common)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/run.py" "${CMAKE_CURRENT_BINARY_DIR}/" unified_vs_device_memory)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/analyze_results.py" "${CMAKE_CURRENT_BINARY_DIR}/" unified_vs_device_memory)
