# nvblox_renderer_utils - Utility functions

add_nvblox_shared_library(
  nvblox_renderer_utils
  SOURCE_FILES
  shader_utils.cpp
  view_camera.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_renderer_core
  INCLUDE_DIRECTORIES_PUBLIC
  ${NVBLOX_RENDERER_INCLUDE_DIRS})

# Shader directories needed by shader_utils.cpp The shader loader tries
# directories in this order: 1. NVBLOX_SHADER_DIR environment variable (if set)
# 2. BUILD_SHADER_DIR - build tree location (for development) 3.
# INSTALL_SHADER_DIR - installed location
#
# IMPORTANT: INSTALL_SHADER_DIR is evaluated at CMake configure time, not
# install time. If you install to a different prefix than CMAKE_INSTALL_PREFIX:
# cmake --install . --prefix /custom/path You MUST set the NVBLOX_SHADER_DIR
# environment variable at runtime: export
# NVBLOX_SHADER_DIR=/custom/path/share/nvblox_renderer/shaders
target_compile_definitions(
  nvblox_renderer_utils
  PRIVATE
    BUILD_SHADER_DIR="${SHADER_BINARY_DIR}"
    INSTALL_SHADER_DIR="${CMAKE_INSTALL_FULL_DATADIR}/nvblox_renderer/shaders")
add_dependencies(nvblox_renderer_utils shaders)
