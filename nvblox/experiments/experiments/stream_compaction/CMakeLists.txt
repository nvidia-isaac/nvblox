add_executable(stream_compaction
main.cpp
lib/stream_compaction.cu
lib/cached_allocator.cu
)
target_include_directories(stream_compaction PUBLIC
include
)
target_link_libraries(stream_compaction nvblox_experiments_common)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/run.py" "${CMAKE_CURRENT_BINARY_DIR}/" stream_compaction)
makeLink("${CMAKE_CURRENT_SOURCE_DIR}/plot.py" "${CMAKE_CURRENT_BINARY_DIR}/" stream_compaction)
