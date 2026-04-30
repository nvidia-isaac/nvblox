# ##############################################################################
# DEPENDENCIES #
# ##############################################################################

# Include package deps
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(CUDAToolkit REQUIRED)

# Download thirdparty deps
message(STATUS "Downloading 3rdparty dependencies")
message(STATUS "Downloading GFLAGS")
include(thirdparty/gflags/gflags.cmake)
message(STATUS "Downloading SQLITE")
include(thirdparty/sqlite/sqlite.cmake)
message(STATUS "Downloading Eigen")
include(thirdparty/eigen/eigen.cmake)
message(STATUS "Downloading STDGPU")
include(thirdparty/stdgpu/stdgpu.cmake)
message(STATUS "Downloading GLOG")
include(thirdparty/glog/glog.cmake)
message(STATUS "Downloading GTEST")
include(thirdparty/gtest/gtest.cmake)
message(STATUS "Downloading benchmark")
include(thirdparty/benchmark/benchmark.cmake)

# Enable compiler warnings. We do this after inclusion of third party libraries
# to avoid warnings we have no control over TODO(dtingdahl) Fix pytorch warnings
# and re-enable these flags add_compile_options(-Wall -Wextra -Wshadow)

# ##############################################################################
# INCLUDES #
# ##############################################################################

# Internal includes
include_directories(include)

# External includes
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ##############################################################################
# LIBRARIES #
# ##############################################################################
# NOTE(alexmillane): nvblox (unfortunately) is split into two libraries,
# "nvblox_gpu_hash" which wraps our interactions with stdgpu and "nvblox_lib"
# which only interacts with our gpu hash wrapper. Ideally, I would have liked to
# have just a single object, however this caused run-time errors. We compile the
# first *without* the separable code flag, and the second with. This is the only
# way I've managed to get things working so far.
add_nvblox_static_library(
  nvblox_gpu_hash
  SOURCE_FILES
  src/core/error_check.cu
  src/utils/timing.cpp
  src/utils/nvtx_ranges.cpp
  src/gpu_hash/tsdf_layer_specialization.cu
  src/gpu_hash/feature_layer_specialization.cu
  src/gpu_hash/freespace_layer_specialization.cu
  src/gpu_hash/esdf_layer_specialization.cu
  src/gpu_hash/color_layer_specialization.cu
  src/gpu_hash/occupancy_layer_specialization.cu
  src/gpu_hash/mesh_layer_specialization.cu
  LINK_LIBRARIES_PUBLIC
  nvblox_eigen
  LINK_LIBRARIES_PRIVATE
  nvblox_stdgpu
  glog::glog
  gflags)

add_nvblox_shared_library(
  nvblox_lib
  SOURCE_FILES
  src/core/cuda_stream.cpp
  src/core/warmup.cu
  src/core/error_check.cu
  src/core/parameter_tree.cpp
  src/dynamics/dynamics_detection.cu
  src/experimental/ground_plane/tsdf_zero_crossings_extractor.cu
  src/experimental/ground_plane/ransac_plane_fitter_cpu.cpp
  src/experimental/ground_plane/ransac_plane_fitter.cu
  src/experimental/ground_plane/ground_plane_estimator.cpp
  src/map/blocks_to_update_tracker.cpp
  src/map/blox.cu
  src/map/layer.cu
  src/sensors/mask_preprocessor.cpp
  src/sensors/camera.cpp
  src/sensors/color.cpp
  src/sensors/pointcloud.cu
  src/sensors/pointcloud_to_depth_conversion.cu
  src/sensors/image.cu
  src/sensors/npp_image_operations.cpp
  src/sensors/depth_preprocessing.cpp
  src/geometry/bounding_boxes.cpp
  src/geometry/bounding_shape.cpp
  src/geometry/bounding_spheres.cpp
  src/geometry/workspace_bounds.cpp
  src/geometry/transforms.cpp
  src/mapper/mapper.cpp
  src/mapper/multi_mapper.cpp
  src/integrators/shape_clearer.cu
  src/integrators/view_calculator.cu
  src/integrators/occupancy_decay_integrator.cu
  src/integrators/tsdf_decay_integrator.cu
  src/integrators/projective_occupancy_integrator.cu
  src/integrators/projective_tsdf_integrator.cu
  src/integrators/projective_appearance_integrator.cu
  src/integrators/freespace_integrator.cu
  src/integrators/esdf_integrator.cu
  src/integrators/esdf_slicer.cu
  src/integrators/occupancy_conversions.cpp
  src/integrators/instantiations/camera_inst.cu
  src/integrators/instantiations/lidar_inst.cu
  src/rays/sphere_tracer.cu
  src/interpolation/interpolation_3d.cpp
  src/io/mesh_io.cpp
  src/io/ply_writer.cpp
  src/io/ply_parser.cpp
  src/io/layer_cake_io.cpp
  src/io/pointcloud_io.cpp
  src/io/image_io.cpp
  src/map_saving/serializer.cpp
  src/map_saving/sqlite_database.cpp
  src/map_saving/layer_type_register.cpp
  src/mesh/mesh_block.cu
  src/mesh/mesh_integrator_appearance.cu
  src/mesh/mesh_integrator.cu
  src/mesh/mesh_transform.cu
  src/mesh/flat_mesh_integrator.cu
  src/primitives/primitives.cpp
  src/primitives/scene.cpp
  src/utils/nvtx_ranges.cpp
  src/utils/timing.cpp
  src/utils/rates.cpp
  src/utils/nvblox_art.cpp
  src/utils/delays.cpp
  src/serialization/mesh_serializer_gpu.cu
  src/serialization/serialization_gpu.cu
  src/serialization/mesh_serializer_gpu.cu
  src/semantics/image_masker.cu
  src/semantics/image_projector.cu
  src/semantics/mask_from_detections.cu
  src/projective_texturing/texture_atlas.cu
  src/projective_texturing/projective_texture_mapper.cu
  LINK_LIBRARIES_PUBLIC
  glog::glog
  gflags
  nvblox_eigen
  CUDA::cudart
  LINK_LIBRARIES_PRIVATE
  nvblox_stdgpu
  nvblox_gpu_hash
  sqlite3
  CUDA::nppc
  CUDA::nppial
  CUDA::nppitc
  CUDA::nppim
  CUDA::nppidei
  INCLUDE_DIRECTORIES_PUBLIC
  ${THRUST_INCLUDE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# ##############################################################################
# BINARIES #
# ##############################################################################
# NOTE(alexmillane, 2025-04-24): JP5 is requiring linking executables against
# nvblox_stdgpu and nvblox_gpu_hash to get some transitive dependencies. This
# should not be required, as isn't required on JP6/7. Remove this overlinking
# once JP5 is no longer supported.
if(BUILD_BENCHMARKS)
  add_nvblox_executable(
    sphere_benchmark
    SOURCE_FILES
    src/benchmarks/sphere_benchmark.cpp
    LINK_LIBRARIES_PUBLIC
    nvblox_lib
    nvblox_stdgpu
    nvblox_gpu_hash)
endif()

# Dataset loaders
add_subdirectory(datasets)

# Tiny example binaries.
add_subdirectory(examples)

#
# TESTS #
#
include(CTest)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# ##############################################################################
# EXPERIMENTS #
# ##############################################################################
add_subdirectory(experiments)

# ##############################################################################
# EXPORT #
# ##############################################################################
include(GNUInstallDirs)

install(
  TARGETS nvblox_lib nvblox_datasets nvblox_eigen
  EXPORT nvbloxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/datasets/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(NOT USE_SYSTEM_EIGEN)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*/unsupported**" EXCLUDE)
endif()

if(NOT USE_SYSTEM_STDGPU)
  install(DIRECTORY ${ext_stdgpu_SOURCE_DIR}/src/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
