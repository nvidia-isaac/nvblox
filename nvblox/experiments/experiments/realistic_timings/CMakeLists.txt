add_custom_target( realistic_timings )
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/gather_timings.py" "${CMAKE_CURRENT_BINARY_DIR}/" realistic_timings)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/plot_timings.py" "${CMAKE_CURRENT_BINARY_DIR}/" realistic_timings)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/compare_platforms.py" "${CMAKE_CURRENT_BINARY_DIR}/" realistic_timings)
