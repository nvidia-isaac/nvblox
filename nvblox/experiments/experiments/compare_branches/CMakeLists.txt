add_custom_target( compare_branches )
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/compare_branches.py" "${CMAKE_CURRENT_BINARY_DIR}/" compare_branches)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/plot_comparison.py" "${CMAKE_CURRENT_BINARY_DIR}/" compare_branches)
