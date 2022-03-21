add_library(vector_copies_cuda vector_copies.cu)
target_include_directories(vector_copies_cuda PUBLIC "../../../tests/include")
target_link_libraries(vector_copies_cuda nvblox_experiments_common nvblox_test_utils)

add_executable(vector_copies main.cpp)
target_link_libraries(vector_copies vector_copies_cuda nvblox_experiments_common)

makeLink("${CMAKE_CURRENT_SOURCE_DIR}/plot_timings.py" "${CMAKE_CURRENT_BINARY_DIR}/" vector_copies)
