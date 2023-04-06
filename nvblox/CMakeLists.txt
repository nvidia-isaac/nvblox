cmake_minimum_required(VERSION 3.16)

# set the project name and version
project(nvblox VERSION 0.0.4 LANGUAGES CXX CUDA)

########################
# OPTIONS AND SETTINGS #
########################
# Build options
option(BUILD_EXPERIMENTS "Build performance experimentation binaries" OFF)
option(BUILD_TESTING "Build tests" ON)

# Whether to build redistributable artifacts using static libraries. In this
# case, the pre-build static libraries for stdgpu, glog, and gflags MUST be
# specified, and built with -fPIC. (See README instructions).
option(BUILD_REDISTRIBUTABLE "Whether to build redistributable artifacts, 
  linking in static versions of the dependent libraries" OFF)

# These are only used if BUILD_REDISTRIBUTABLE is ON:
# Set these from the command line by setting -DSQLITE3_BASE_PATH="YOUR/PATH/HERE" when calling cmake.
set(SQLITE3_BASE_PATH "" CACHE STRING "Base path to static output for Sqlite3, build with -fPIC.")
set(GLOG_BASE_PATH "" CACHE STRING "Base path to static output for glog, build with -fPIC.")
set(GFLAGS_BASE_PATH "" CACHE STRING "Base path to static output for gflags, build with -fPIC.")

# GPU architectures
# By default this flag is NOT set. Cmake then detects the architecture of the build computer
# and compiles for that architecture only. This can be an issue if you're building on one
# machine, and running on machines with a different GPU achitecture. In this case, set the flag.
# The penalty for doing this is increased build times.
option(BUILD_FOR_ALL_ARCHS "Build for all GPU architectures" OFF)

# This option avoids any implementations using std::string in their signature in header files
# Useful for Nvblox PyTorch wrapper, which requires the old Pre-CXX11 ABI
option(PRE_CXX11_ABI_LINKABLE "Better support pre-C++11 ABI library users" OFF)
IF(PRE_CXX11_ABI_LINKABLE)
    ADD_DEFINITIONS(-DPRE_CXX11_ABI_LINKABLE)
ENDIF(PRE_CXX11_ABI_LINKABLE)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# This link flag replaces "runpath" with "rpath" in executables and shared objects.
# This is important because it means the search paths are passed down the shared object tree.
# https://stackoverflow.com/questions/58997230/cmake-project-fails-to-find-shared-library
set(nvblox_link_options "-Wl,--disable-new-dtags")

# Cmake -fPIC flag.
# NOTE(alexmillane): I needed to add this when I changed to linking against a static version
# of stdgpu. Without is we get an error from thrust/cub linking.
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Only used if the BUILD_FOR_ALL_ARCHS flag above is true.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(CUDA_ARCHITECTURE_FLAGS "86;80;75;72;70;61;60")
else()
    set(CUDA_ARCHITECTURE_FLAGS " -gencode=arch=compute_86,code=sm_86 \
                                  -gencode=arch=compute_80,code=sm_80 \
                                  -gencode=arch=compute_75,code=sm_75 \
                                  -gencode=arch=compute_72,code=sm_72 \
                                  -gencode=arch=compute_70,code=sm_70 \
                                  -gencode=arch=compute_61,code=sm_61 \
                                  -gencode=arch=compute_60,code=sm_60")
endif()

# Suppress spammy Eigen CUDA warnings.
# "expt-relaxed-constexpr" allows sharing constexpr between host and device code.
# "display_error_number" shows a warning number with all warnings, and the
# rest is just suppressing specific warnings from Eigen. Note that the numbers
# keep changing with every CUDA release so this list is a bit arbitrary.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr --disable-warnings --generate-line-info -lineinfo -Xcudafe --display_error_number")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")

################
# DEPENDENCIES #
################
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

# Treat redistributable and non-redistributable builds differently for
# the other dependencies.
if(BUILD_REDISTRIBUTABLE)
    if(SQLITE3_BASE_PATH STREQUAL "")
        message(FATAL_ERROR "No SQLite3 static libraries specified! Set the SQLITE3_BASE_PATH variable!")
    endif()

    if(GLOG_BASE_PATH STREQUAL "")
        message(FATAL_ERROR "No Glog static libraries specified! Set the GLOG_BASE_PATH variable!")
    endif()

    if(GFLAGS_BASE_PATH STREQUAL "")
        message(FATAL_ERROR "No gflags static libraries specified! Set the GFLAGS_BASE_PATH variable!")
    endif()

    # SQLite static lib:
    set(SQLite3_LIBRARIES "${SQLITE3_BASE_PATH}/lib/libsqlite3.a")
    set(SQLite3_INCLUDE_DIRS "${SQLITE3_BASE_PATH}/include/")

    # GLOG STATIC LIB
    set(GLOG_LIBRARIES "${GLOG_BASE_PATH}/lib/libglog.a")
    set(GLOG_INCLUDE_DIRS "${GLOG_BASE_PATH}/include")

    # GFLAGS STATIC LIB
    set(gflags_LIBRARIES "${GFLAGS_BASE_PATH}/lib/libgflags.a")
    set(gflags_INCLUDE_DIRS "${GFLAGS_BASE_PATH}/include")

    # Set dependent settings, like turning testing off.
    set(BUILD_TESTING OFF)
else()
    # By default, use system dependencies for these.
    # In the case of ROS builds, glog will likely be found at a higher level.
    # We want to link against that version in that case.
    set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION FALSE)

    if(NOT Glog_FOUND)
        find_package(Glog REQUIRED)
    endif()

    find_package(gflags REQUIRED)
    find_package(SQLite3 REQUIRED)

    set(gflags_INCLUDE_DIRS ${gflag_INCLUDE_DIR})
    set(gflags_LIBRARIES "gflags")
endif()

############
# INCLUDES #
############
# Include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(${GLOG_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})

#############
# LIBRARIES #
#############
# NOTE(alexmillane): nvblox (unfortunately) is split into two libraries, "nvblox_gpu_hash" which
# wraps our interactions with stdgpu and "nvblox_lib" which only interacts with our gpu hash wrapper.
# Ideally, I would have liked to have just a single object, however this caused run-time errors.
# We compile the first *without* the separable code flag, and the second with. This is the only way
# I've managed to get things working so far.
add_library(nvblox_gpu_hash STATIC
    src/core/error_check.cu
    src/utils/timing.cpp
    src/utils/nvtx_ranges.cpp
    src/gpu_hash/gpu_layer_view.cu
    src/gpu_hash/gpu_set.cu
)
add_dependencies(nvblox_gpu_hash nvblox_eigen stdgpu)
target_link_libraries(nvblox_gpu_hash
    PUBLIC
    stdgpu
    nvblox_eigen
    ${CUDA_nvToolsExt_LIBRARY}
    PRIVATE
    ${GLOG_LIBRARIES}
    ${gflags_LIBRARIES}
)
target_link_options(nvblox_gpu_hash PUBLIC ${nvblox_link_options})

add_library(nvblox_lib SHARED
    src/core/warmup.cu
    src/core/error_check.cu
    src/map/blox.cu
    src/map/layer.cu
    src/sensors/camera.cpp
    src/sensors/color.cpp
    src/sensors/pointcloud.cu
    src/sensors/image.cu
    src/geometry/bounding_boxes.cpp
    src/geometry/bounding_spheres.cpp
    src/mapper/mapper.cpp
    src/mapper/multi_mapper.cpp
    src/integrators/view_calculator.cu
    src/integrators/occupancy_decay_integrator.cu
    src/integrators/projective_occupancy_integrator.cu
    src/integrators/projective_tsdf_integrator.cu
    src/integrators/projective_color_integrator.cu
    src/integrators/esdf_integrator.cu
    src/rays/sphere_tracer.cu
    src/interpolation/interpolation_3d.cpp
    src/io/mesh_io.cpp
    src/io/ply_writer.cpp
    src/io/layer_cake_io.cpp
    src/io/pointcloud_io.cpp
    src/io/image_io.cpp
    src/mesh/marching_cubes.cu
    src/mesh/mesh_block.cu
    src/mesh/mesh_integrator_color.cu
    src/mesh/mesh_integrator.cu
    src/mesh/mesh.cpp
    src/primitives/primitives.cpp
    src/primitives/scene.cpp
    src/utils/nvtx_ranges.cpp
    src/utils/timing.cpp
    src/serialization/serializer.cpp
    src/serialization/sqlite_database.cpp
    src/serialization/layer_type_register.cpp
    src/semantics/image_masker.cu
    src/semantics/image_projector.cu
)
target_link_libraries(nvblox_lib
    PUBLIC
    ${GLOG_LIBRARIES}
    ${gflags_LIBRARIES}
    nvblox_eigen
    ${CUDA_LIBRARIES}
    nvblox_gpu_hash
    PRIVATE
    ${CUDA_nvToolsExt_LIBRARY}
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
    set_property(TARGET nvblox_lib APPEND PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_property(TARGET nvblox_lib APPEND PROPERTY EXPORT_PROPERTIES CUDA_ARCHITECTURES)
endif()

set_property(TARGET nvblox_lib PROPERTY CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
set_property(TARGET nvblox_lib APPEND PROPERTY EXPORT_PROPERTIES CMAKE_CUDA_FLAGS)

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
include(CTest)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()

###############
# EXPERIMENTS #
###############
add_subdirectory(experiments)

##########
# EXPORT #
##########
include(GNUInstallDirs)

set_target_properties(stdgpu PROPERTIES INTERFACE_LINK_LIBRARIES "")

install(
    TARGETS nvblox_lib nvblox_gpu_hash nvblox_datasets stdgpu nvblox_eigen fuse_3dmatch fuse_replica
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
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/executables/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*/unsupported**" EXCLUDE
)

if(BUILD_REDISTRIBUTABLE)
    # Only re-distribute these headers if we're building the redistributable artifacts.
    install(
        DIRECTORY ${gflags_INCLUDE_DIRS}/gflags
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        DIRECTORY ${GLOG_INCLUDE_DIRS}/glog
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

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
