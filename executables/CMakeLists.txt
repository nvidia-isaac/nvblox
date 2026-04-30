# Select the fuser visualizer implementation based on BUILD_RENDERER parameter.
if(BUILD_RENDERER)
  set(FUSER_VISUALIZER_SRC src/fuser/fuser_visualizer.cpp)
else()
  set(FUSER_VISUALIZER_SRC src/fuser/fuser_visualizer_stub.cpp) # no visualizer
endif()

# Fuser library (fuser utilities shared across dataset executables)
add_nvblox_shared_library(
  nvblox_fuser
  SOURCE_FILES
  src/fuser/fuser.cpp
  ${FUSER_VISUALIZER_SRC}
  LINK_LIBRARIES_PUBLIC
  nvblox_datasets
  nvblox_lib
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(BUILD_RENDERER)
  target_link_libraries(nvblox_fuser PUBLIC nvblox_renderer)
endif()

# 3Dmatch executable
add_nvblox_executable(
  fuse_3dmatch
  SOURCE_FILES
  src/fuser/fuse_3dmatch.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_fuser
  nvblox_stdgpu
  nvblox_gpu_hash)

# Replica executable
add_nvblox_executable(
  fuse_replica
  SOURCE_FILES
  src/fuser/fuse_replica.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_fuser
  nvblox_stdgpu
  nvblox_gpu_hash)

# Redwood executable
add_nvblox_executable(
  fuse_redwood
  SOURCE_FILES
  src/fuser/fuse_redwood.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_fuser
  nvblox_stdgpu
  nvblox_gpu_hash)

# CUSFM executable
add_nvblox_executable(
  fuse_cusfm
  SOURCE_FILES
  src/fuser/fuse_cusfm.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_fuser
  nvblox_stdgpu
  nvblox_gpu_hash)

# Lidar dataset fuser executable
add_nvblox_executable(
  fuse_lidarply
  SOURCE_FILES
  src/fuser/fuse_lidarply.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_fuser
  nvblox_stdgpu
  nvblox_gpu_hash)

if(BUILD_BENCHMARKS)
  add_nvblox_executable(
    nvblox_benchmark
    SOURCE_FILES
    benchmark/benchmark.cpp
    LINK_LIBRARIES_PUBLIC
    nvblox_fuser
    nvblox_test_utils
    benchmark::benchmark)

  add_nvblox_executable(
    nvblox_benchmark_camera
    SOURCE_FILES
    benchmark/benchmark_camera.cpp
    LINK_LIBRARIES_PUBLIC
    nvblox_fuser
    nvblox_test_utils
    benchmark::benchmark)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
