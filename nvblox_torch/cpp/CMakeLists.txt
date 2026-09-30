#[[
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
]]

# Torch messes up the architecture flags, so have to restore them
set(CMAKE_CUDA_ARCHITECTURES_OLD ${CMAKE_CUDA_ARCHITECTURES})
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_OLD})

# Torch exports its required dialect as a target property, not an interface
# feature. Newer releases require C++20; older/Jetson releases still use C++17.
get_target_property(NVBLOX_TORCH_CXX_STANDARD torch CXX_STANDARD)
if(NOT NVBLOX_TORCH_CXX_STANDARD)
  set(NVBLOX_TORCH_CXX_STANDARD 17)
endif()

# On Jetson, we need to link explicitly against cudnn
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND TORCH_LIBRARIES cudnn)
endif()

# Nvblox-torch python wrapper
add_nvblox_shared_library(
  py_nvblox
  SOURCE_FILES
  src/py_nvblox.cu
  src/py_mapper.cu
  src/py_scene.cu
  src/py_sensor.cu
  src/convert_tensors.cpp
  src/sdf_query.cu
  src/py_layer.cpp
  src/py_mesh.cpp
  src/py_mapper_params.cpp
  src/cuda_stream.cpp
  src/py_rendering.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  ${TORCH_LIBRARIES}
  LINK_LIBRARIES_PRIVATE
  nvblox_stdgpu
  nvblox_gpu_hash
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Nvblox-torch python wrapper
add_nvblox_shared_library(
  py_nvblox_test
  SOURCE_FILES
  src/convert_tensors.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  ${TORCH_LIBRARIES}
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Propagate the header requirement to C++ consumers and use the same dialect for
# for the CUDA translation units. The project's language minimum also applies.
foreach(target_name py_nvblox py_nvblox_test)
  target_compile_features(
    ${target_name}
    PUBLIC cxx_std_${NVBLOX_TORCH_CXX_STANDARD}
    PRIVATE cuda_std_${NVBLOX_TORCH_CXX_STANDARD})
endforeach()

# Record the public Torch package version for wheel metadata. Preserve JetPack
# prerelease versions when Python and the selected Torch CMake package agree;
# LibTorch-only builds can use the version exported by its CMake configuration.
set(NVBLOX_TORCH_PACKAGE_VERSION "${Torch_VERSION}")
execute_process(
  COMMAND
    python3 -c
    "from importlib.metadata import version; print(version('torch').split('+')[0])"
  OUTPUT_VARIABLE NVBLOX_PYTHON_TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  RESULT_VARIABLE NVBLOX_TORCH_VERSION_RESULT)
if(NVBLOX_TORCH_VERSION_RESULT EQUAL 0 AND NVBLOX_PYTHON_TORCH_VERSION
                                           VERSION_EQUAL Torch_VERSION)
  set(NVBLOX_TORCH_PACKAGE_VERSION "${NVBLOX_PYTHON_TORCH_VERSION}")
endif()
file(
  GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/torch_version.txt.in"
  CONTENT "${NVBLOX_TORCH_PACKAGE_VERSION}\n")
set_property(
  TARGET py_nvblox
  APPEND
  PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/torch_version.txt.in")
# Only publish the record after a successful link, not during configuration.
add_custom_command(
  TARGET py_nvblox
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_BINARY_DIR}/torch_version.txt.in"
    "${CMAKE_CURRENT_BINARY_DIR}/torch_version.txt"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/torch_version.txt"
  VERBATIM)

# Tests
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
