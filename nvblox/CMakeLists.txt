cmake_minimum_required(VERSION 3.16)

# set the project name and version
project(nvblox VERSION 0.0.1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Build options
option(BUILD_EXPERIMENTS "Build performance experimentation binaries" ON)
option(BUILD_TESTS "Build the C++ tests of the nvblox library" ON)

# Suppress spammy Eigen CUDA warnings.
# "expt-relaxed-constexpr" allows sharing constexpr between host and device
# code.
# "display_error_number" shows a warning number with all warnings, and the
# rest is just suppressing specific warnings from Eigen. Note that the numbers
# keep changing with every CUDA release so this list is a bit arbitrary.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr  --generate-line-info -lineinfo -Xcudafe --display_error_number -Xcudafe --diag_suppress=2977  -Xcudafe --diag_suppress=3057  -Xcudafe --diag_suppress=3059 ")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")

# New warning numbers above CUDA 11.2.
if (CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 2)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012 -Xcudafe --diag_suppress=20011 -Xcudafe --diag_suppress=20014")
endif()

# Download thirdparty deps
message(STATUS "Downloading 3rdparty dependencies")
message(STATUS "Downloading Eigen")
include(thirdparty/eigen/eigen.cmake)

# Build stdgpu as part of this
message(STATUS "Downloading STDGPU")
include(thirdparty/stdgpu/stdgpu.cmake)

# Include dem deps
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(CUDA REQUIRED)
# In the case of ROS builds, glog will likely be found at a higher level.
# We want to link against that version in that case.
if(NOT Glog_FOUND)
    find_package(Glog REQUIRED)
endif()
# TODO(alexmillane): check the ROS builds here.
find_package(gflags REQUIRED)

# Include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${STDGPU_INCLUDE_DIRS})
include_directories(include)

#############
# LIBRARIES #
#############

add_library(nvblox_cuda_check SHARED
    src/core/cuda/error_check.cu
)

add_library(nvblox_gpu_hash SHARED
    src/gpu_hash/cuda/gpu_layer_view.cu
    src/gpu_hash/cuda/gpu_set.cu
    src/utils/timing.cpp
    src/utils/nvtx_ranges.cpp
)
add_dependencies(nvblox_gpu_hash ext_eigen stdgpu)
target_link_libraries(nvblox_gpu_hash PUBLIC
    stdgpu
    nvblox_cuda_check
    ${CUDA_nvToolsExt_LIBRARY}
)

add_library(nvblox_lib SHARED
    src/core/bounding_boxes.cpp
    src/core/camera.cpp
    src/core/color.cpp
    src/core/cuda/blox.cu
    src/core/cuda/image_cuda.cu
    src/core/cuda/warmup.cu
    src/core/image.cpp
    src/core/interpolation_3d.cpp
    src/core/mapper.cpp
    src/datasets/image_loader.cpp
    src/datasets/parse_3dmatch.cpp
    src/integrators/cuda/frustum.cu
    src/integrators/cuda/projective_tsdf_integrator.cu
    src/integrators/cuda/projective_color_integrator.cu
    src/integrators/cuda/esdf_integrator.cu
    src/integrators/esdf_integrator.cpp
    src/integrators/frustum.cpp
    src/integrators/projective_integrator_base.cpp
    src/ray_tracing/cuda/sphere_tracer.cu
    src/io/csv.cpp
    src/io/mesh_io.cpp
    src/io/ply_writer.cpp
    src/mesh/marching_cubes.cu
    src/mesh/mesh_block.cu
    src/mesh/mesh_integrator_color.cu
    src/mesh/mesh_integrator.cpp
    src/mesh/mesh_integrator.cu
    src/mesh/mesh.cpp
    src/primitives/primitives.cpp
    src/primitives/scene.cpp
    src/utils/nvtx_ranges.cpp
    src/utils/timing.cpp
)
add_dependencies(nvblox_lib ext_eigen)
target_link_libraries(nvblox_lib PUBLIC
    ${GLOG_LIBRARIES}
    gflags
    ${CUDA_LIBRARIES}
    ${CUDA_nvToolsExt_LIBRARY}
    nvblox_gpu_hash
    nvblox_cuda_check
)
target_link_libraries(nvblox_lib INTERFACE
    stdgpu
)
target_include_directories(nvblox_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include/eigen3>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)    
set_target_properties(nvblox_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

############
# BINARIES #
############
add_executable(sphere_benchmark src/benchmarks/sphere_benchmark.cpp)
target_link_libraries(sphere_benchmark
    nvblox_lib
)
set_target_properties(sphere_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#########
# TESTS #
#########

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

###############
# EXPERIMENTS #
###############

if (BUILD_EXPERIMENTS)
    add_subdirectory(experiments)
endif()

#############################
# INTERFACE LIBRARY FOR ROS #
#############################
# TODO: delete
add_library(nvblox_interface INTERFACE)
target_link_libraries(nvblox_interface INTERFACE 
    nvblox_lib 
    nvblox_gpu_hash 
    nvblox_cuda_check 
    ${GLOG_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    stdgpu 
    Eigen3::Eigen
)
target_include_directories(nvblox_interface INTERFACE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_dependencies(nvblox_interface ext_eigen ext_stdgpu)

##########
# EXPORT #
##########
include(GNUInstallDirs)

install( 
  TARGETS nvblox_lib nvblox_gpu_hash nvblox_cuda_check stdgpu
  EXPORT nvblox
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h*"
)


include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake"
  INSTALL_DESTINATION "lib/cmake"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfigVersion.cmake"
  VERSION "${nvblox_VERSION_MAJOR}.${nvblox_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nvbloxTargets.cmake
    DESTINATION share/nvblox/cmake)

# generate the export targets for the build tree
# needs to be after the install(TARGETS ) command
export(EXPORT nvblox
    NAMESPACE nvblox::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/nvbloxTargets.cmake"
)

install(
    EXPORT nvblox 
    FILE nvbloxTargets.cmake
    NAMESPACE nvblox::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
