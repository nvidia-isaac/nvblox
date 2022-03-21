add_executable(threaded_image_loading main.cpp)
target_link_libraries(threaded_image_loading nvblox_experiments_common)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/run.py" "${CMAKE_CURRENT_BINARY_DIR}/" threaded_image_loading)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/plot_comparison.py" "${CMAKE_CURRENT_BINARY_DIR}/" threaded_image_loading)
