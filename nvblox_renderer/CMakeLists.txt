# nvblox_renderer - Modular Vulkan renderer with CUDA interop for nvblox Must be
# built as part of nvblox with -DBUILD_RENDERER=ON

message(STATUS "Building nvblox_renderer as part of nvblox")

# Include GNUInstallDirs early for install path variables used in generator
# expressions
include(GNUInstallDirs)

# ##############################################################################
# DEPENDENCIES
# ##############################################################################

# CUDA
find_package(CUDAToolkit REQUIRED)

# nvcc 11.x cannot compile C++17 deduction guides (CTAD) in standard library
# headers (e.g. <optional>, <functional>) that are transitively included by
# Vulkan interop code. Require CUDA 12+.
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
  message(
    FATAL_ERROR
      "[nvblox_renderer] Fatal: requires CUDA >= 12.0 "
      "(found ${CUDAToolkit_VERSION}). "
      "nvcc 11.x cannot compile C++17 headers used by Vulkan interop.")
endif()

# Vulkan (require 1.2+ for timeline semaphores)
find_package(Vulkan 1.2 REQUIRED)
message(STATUS "[nvblox_renderer] Found Vulkan: ${Vulkan_VERSION}")

# glog - should already be available from nvblox.
if(NOT TARGET glog::glog)
  find_package(glog REQUIRED)
endif()

# GLFW (3.3+ required for Vulkan support) Note: GLFW builds fine on headless
# systems - window creation just fails at runtime. Users should use headless
# mode (config.headless = true) on systems without a display.
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
  message(
    STATUS
      "[nvblox_renderer] System GLFW3 not found, fetching v3.4 from source...")
  include(FetchContent)
  set(GLFW_BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4)
  FetchContent_MakeAvailable(glfw)
else()
  message(STATUS "[nvblox_renderer] Found system GLFW3: ${glfw3_VERSION}")
  # System GLFW3 creates glfw3::glfw3 target, but FetchContent creates 'glfw'.
  # Create an alias for consistent target naming across both cases.
  if(NOT TARGET glfw)
    add_library(glfw ALIAS glfw3::glfw3)
  endif()
endif()

# Common include directories for all renderer modules
set(NVBLOX_RENDERER_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# ##############################################################################
# SHADERS - Compiled before modules that depend on them
# ##############################################################################

find_program(GLSLC glslangValidator HINTS "$ENV{VULKAN_SDK}/bin" /usr/bin)
if(NOT GLSLC)
  message(
    FATAL_ERROR
      "[nvblox_renderer] glslangValidator not found.\n"
      "  Install glslang-tools: apt-get install glslang-tools\n"
      "  Searched: $ENV{VULKAN_SDK}/bin, /usr/bin")
endif()

message(STATUS "[nvblox_renderer] Found glslangValidator: ${GLSLC}")

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

# Enable shader optimization for release builds
if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
  set(GLSLC_OPT_FLAG "-Os")
  message(STATUS "[nvblox_renderer] Shader optimization enabled (-Os)")
else()
  set(GLSLC_OPT_FLAG "")
endif()

set(SHADERS image_quad.vert image_quad.frag point_cloud.vert point_cloud.frag
            mesh.vert mesh.frag)

foreach(SHADER ${SHADERS})
  set(SHADER_SRC ${SHADER_SOURCE_DIR}/${SHADER})
  set(SHADER_SPV ${SHADER_BINARY_DIR}/${SHADER}.spv)

  add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC} -V ${GLSLC_OPT_FLAG} -I${SHADER_SOURCE_DIR} -o
            ${SHADER_SPV} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC} ${SHADER_SOURCE_DIR}/color_utils.glsl
    COMMENT "Compiling ${SHADER}")
  list(APPEND SHADER_SPIRV_FILES ${SHADER_SPV})
endforeach()

add_custom_target(shaders DEPENDS ${SHADER_SPIRV_FILES})

# ##############################################################################
# MODULES
# ##############################################################################

add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(src/render_targets)
add_subdirectory(src/kernels)
add_subdirectory(src/visualizers)

# ##############################################################################
# MAIN LIBRARY - Combines all modules, user-facing API
# ##############################################################################

add_nvblox_shared_library(
  nvblox_renderer
  SOURCE_FILES
  src/renderer.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_renderer_core
  nvblox_renderer_utils
  nvblox_renderer_targets
  nvblox_renderer_kernels
  nvblox_renderer_visualizers
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})

# ##############################################################################
# TESTS
# ##############################################################################

if(BUILD_TESTING)
  if(USE_SANITIZER)
    message(
      STATUS
        "[nvblox_renderer] Skipping tests because USE_SANITIZER is enabled "
        "(Vulkan/CUDA interop tests tend to timeout with gcc address sanitizer)"
    )
  else()
    add_subdirectory(tests)
  endif()
endif()

# ##############################################################################
# INSTALL RULES
# ##############################################################################

# Install libraries
install(
  TARGETS nvblox_renderer nvblox_renderer_core nvblox_renderer_utils
          nvblox_renderer_targets nvblox_renderer_kernels
          nvblox_renderer_visualizers
  EXPORT nvbloxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install renderer headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install compiled shaders
install(
  DIRECTORY ${SHADER_BINARY_DIR}/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/nvblox_renderer/shaders
  FILES_MATCHING
  PATTERN "*.spv")
