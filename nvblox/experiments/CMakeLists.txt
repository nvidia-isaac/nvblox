# Common experiments library
add_library(nvblox_experiments_common SHARED
    src/integrators/cuda/depth_frame_texture.cu
    src/integrators/cuda/experimental_integrator_input_frames.cu
    src/integrators/cuda/experimental_projective_tsdf_integrators.cu
)
target_include_directories(nvblox_experiments_common PUBLIC
    include
)
add_dependencies(nvblox_experiments_common nvblox_datasets)
target_link_libraries(nvblox_experiments_common nvblox_lib nvblox_datasets)
set_target_properties(nvblox_experiments_common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Macro for installing python script SYMBOLICALLY to the buildspace
macro(makeLink src dest target)
  add_custom_command(TARGET ${target} PRE_BUILD
    COMMAND ln -sf ${src} ${dest} DEPENDS ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

# Python module for experiments
set(PYTHON_EXPERIMENTS_MODULE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
set(PYTHON_EXPERIMENTS_MODULE_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
file(MAKE_DIRECTORY ${PYTHON_EXPERIMENTS_MODULE_DST_DIR})
file(COPY ${PYTHON_EXPERIMENTS_MODULE_SRC_DIR}/
     DESTINATION ${PYTHON_EXPERIMENTS_MODULE_DST_DIR})

# Script for comparing branches
add_subdirectory(experiments/compare_branches)

# Experiments
# NOTE(alexmillane): Experiments disabled with "EXCLUDE_FROM_ALL" are no longer
# maintained/building but are kept around for posterity
add_subdirectory(experiments/texture_vs_global_memory_interpolation EXCLUDE_FROM_ALL)
add_subdirectory(experiments/unified_vs_device_memory)
add_subdirectory(experiments/threaded_image_loading)
add_subdirectory(experiments/realistic_timings)
add_subdirectory(experiments/vector_copies)
add_subdirectory(experiments/isolate_tsdf_block_update)
add_subdirectory(experiments/stream_compaction)
add_subdirectory(experiments/layer_cake_interface)
add_subdirectory(experiments/ratio_of_freespace)
