cmake_minimum_required(VERSION 3.16)

# set the project name and version
project(nvblox VERSION 0.0.2 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# This link flag replaces "runpath" with "rpath" in executables and shared objects.
# This is important because it means the search paths are passed down the shared object tree.
# https://stackoverflow.com/questions/58997230/cmake-project-fails-to-find-shared-library
SET(nvblox_link_options "-Wl,--disable-new-dtags")

# Build options
option(BUILD_EXPERIMENTS "Build performance experimentation binaries" ON)
option(BUILD_TESTS "Build the C++ tests of the nvblox library" ON)

# GPU architectures
# By default this flag is NOT set. Cmake then detects the architecture of the build computer
# and compiles for that architecture only. This can be an issue if you're building on one
# machine, and running on machines with a different GPU achitecture. In this case, set the flag.
# The penalty for doing this is increased build times.
option(BUILD_FOR_ALL_ARCHS "Build for all GPU architectures" OFF)

# Only used if the flag above is true.
set(CUDA_ARCHITECTURE_FLAGS " -gencode=arch=compute_86,code=sm_86 \
                              -gencode=arch=compute_80,code=sm_80 \
                              -gencode=arch=compute_75,code=sm_75 \
                              -gencode=arch=compute_72,code=sm_72 \
                              -gencode=arch=compute_70,code=sm_70 \
                              -gencode=arch=compute_61,code=sm_61 \
                              -gencode=arch=compute_60,code=sm_60")

# Suppress spammy Eigen CUDA warnings.
# "expt-relaxed-constexpr" allows sharing constexpr between host and device code.
# "display_error_number" shows a warning number with all warnings, and the
# rest is just suppressing specific warnings from Eigen. Note that the numbers
# keep changing with every CUDA release so this list is a bit arbitrary.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --disable-warnings --generate-line-info -lineinfo -Xcudafe --display_error_number")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")

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

find_package(gflags REQUIRED)

find_package(SQLite3 REQUIRED)

# Include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
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
add_dependencies(nvblox_gpu_hash nvblox_eigen stdgpu)
target_link_libraries(nvblox_gpu_hash PUBLIC
    stdgpu
    nvblox_cuda_check
    nvblox_eigen
    ${CUDA_nvToolsExt_LIBRARY}
    gflags
    ${GLOG_LIBRARIES}
)
target_link_options(nvblox_gpu_hash PUBLIC ${nvblox_link_options})

add_library(nvblox_lib SHARED
    src/core/bounding_boxes.cpp
    src/core/bounding_spheres.cpp
    src/core/camera.cpp
    src/core/color.cpp
    src/core/cuda/blox.cu
    src/core/cuda/image_cuda.cu
    src/core/cuda/warmup.cu
    src/core/image.cpp
    src/core/interpolation_3d.cpp
    src/core/mapper.cpp
    src/integrators/cuda/view_calculator.cu
    src/integrators/cuda/projective_tsdf_integrator.cu
    src/integrators/cuda/projective_color_integrator.cu
    src/integrators/cuda/esdf_integrator.cu
    src/integrators/esdf_integrator.cpp
    src/integrators/view_calculator.cpp
    src/integrators/projective_integrator_base.cpp
    src/rays/cuda/sphere_tracer.cu
    src/io/csv.cpp
    src/io/mesh_io.cpp
    src/io/ply_writer.cpp
    src/io/layer_cake_io.cpp
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
    src/serialization/serializer.cpp
    src/serialization/sqlite_database.cpp
    src/serialization/layer_type_register.cpp
)
target_link_libraries(nvblox_lib PUBLIC
    ${GLOG_LIBRARIES}
    gflags
    ${CUDA_LIBRARIES}
    ${CUDA_nvToolsExt_LIBRARY}
    nvblox_gpu_hash
    nvblox_cuda_check
    nvblox_eigen
    ${SQLite3_LIBRARIES}
)
target_link_options(nvblox_lib PUBLIC ${nvblox_link_options})
target_include_directories(nvblox_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SQLite3_INCLUDE_DIRS}
)

set_target_properties(nvblox_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_target_properties(nvblox_lib PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_target_properties(nvblox_lib PROPERTIES EXPORT_PROPERTIES CUDA_ARCHITECTURES)
else()
    set_target_properties(nvblox_lib PROPERTIES CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set_target_properties(nvblox_lib PROPERTIES EXPORT_PROPERTIES CMAKE_CUDA_FLAGS)
endif()

############
# BINARIES #
############
add_executable(sphere_benchmark src/benchmarks/sphere_benchmark.cpp)
target_link_libraries(sphere_benchmark
    nvblox_lib
)
set_target_properties(sphere_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Binaries for specific datasets
add_subdirectory(executables)

# Tiny example binaries.
add_subdirectory(examples)

#########
# TESTS #
#########
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

###############
# EXPERIMENTS #
###############
if(BUILD_EXPERIMENTS)
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
    Eigen3::Eigen
    gflags
    SQLite::SQLite3
)
target_include_directories(nvblox_interface INTERFACE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

##########
# EXPORT #
##########
include(GNUInstallDirs)

set_target_properties(stdgpu PROPERTIES INTERFACE_LINK_LIBRARIES "")

install(
    TARGETS nvblox_lib nvblox_gpu_hash nvblox_cuda_check nvblox_datasets stdgpu nvblox_eigen fuse_3dmatch fuse_replica
    EXPORT nvbloxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*/unsupported**" EXCLUDE
)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
    "nvbloxConfigVersion.cmake"
    VERSION "${nvblox_VERSION_MAJOR}.${nvblox_VERSION_MINOR}"
    COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfigVersion.cmake
    DESTINATION share/nvblox/cmake)

install(
    EXPORT nvbloxTargets
    NAMESPACE nvblox::
    DESTINATION share/nvblox/cmake
)
