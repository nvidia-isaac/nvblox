cmake_minimum_required(VERSION 3.22.1)

# Set default build type if not provided by user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo"
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# By default we build for the native cuda architecture. Customize by passing
# '-DCMAKE_CUDA_ARCHITECTURES=89;75;72' to cmake.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY TRUE)
else()
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY FALSE)
endif()

# Set the project name and version. Note that this will also set
# CMAKE_CUDA_ARCHITECTURES to a default (potentially non-native) value
project(
  nvblox
  VERSION 0.0.10
  LANGUAGES CXX C CUDA)

# ##############################################################################
# OPTIONS AND SETTINGS #
# ##############################################################################
# Build options
option(BUILD_EXPERIMENTS "Build performance experimentation binaries" OFF)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmark binaries" ON)
option(BUILD_PYTORCH_WRAPPER "Wheter to build the pytorch wrapper" ON)

# Renderer option
option(BUILD_RENDERER "Build the nvblox renderer" ON)

if(BUILD_TESTING)
  include(CTest)
  enable_testing()
endif()

# Include file that defines functions for adding nvblox binary targets
include(cmake/cuda/setup_compute_capability.cmake)
include(cmake/nvblox_targets.cmake)
include(cmake/setup_pytorch_cpp11_abi.cmake)

if(BUILD_PYTORCH_WRAPPER)
  add_subdirectory(nvblox_torch/cpp)
endif()

add_subdirectory(nvblox)

add_subdirectory(executables)

# Optional renderer
if(BUILD_RENDERER)
  add_subdirectory(nvblox_renderer)
endif()

include(CMakePackageConfigHelpers)

# Ggenerate the config file that is includes the exports
configure_package_config_file(
  cmake/Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
  NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# generate the version file for the config file
write_basic_package_version_file(
  "nvbloxConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion)

# install the configuration file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/nvbloxConfigVersion.cmake
        DESTINATION share/nvblox/cmake)

install(
  EXPORT nvbloxTargets
  NAMESPACE nvblox::
  DESTINATION share/nvblox/cmake)
