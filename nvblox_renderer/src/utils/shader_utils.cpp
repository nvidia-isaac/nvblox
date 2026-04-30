/*
Copyright 2026 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "nvblox/renderer/utils/shader_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <glog/logging.h>

namespace nvblox {
namespace renderer {

namespace {

bool isReadableDirectory(const std::string& path) {
  std::error_code ec;
  auto status = std::filesystem::status(path, ec);
  if (ec) {
    return false;
  }
  return std::filesystem::is_directory(status) &&
         (status.permissions() & std::filesystem::perms::owner_read) !=
             std::filesystem::perms::none;
}

}  // namespace

std::string getShaderDir() {
  // Environment variable override (highest priority)
  // Useful when installed to non-standard prefix or for testing
  const char* env_shader_dir = std::getenv("NVBLOX_SHADER_DIR");
  if (env_shader_dir && isReadableDirectory(env_shader_dir)) {
    return env_shader_dir;
  }

  // Try build directory (for development)
#ifdef BUILD_SHADER_DIR
  if (isReadableDirectory(BUILD_SHADER_DIR)) {
    return BUILD_SHADER_DIR;
  }
#endif

  // Fall back to install directory
#ifdef INSTALL_SHADER_DIR
  if (isReadableDirectory(INSTALL_SHADER_DIR)) {
    return INSTALL_SHADER_DIR;
  }
#endif

  // Neither directory found
  LOG(ERROR) << "Shader directory not found. Checked:"
             << "\n  Environment (NVBLOX_SHADER_DIR): "
             << (env_shader_dir ? env_shader_dir : "(not set)")
#ifdef BUILD_SHADER_DIR
             << "\n  Build: " << BUILD_SHADER_DIR
#endif
#ifdef INSTALL_SHADER_DIR
             << "\n  Install: " << INSTALL_SHADER_DIR
#endif
      ;
  return "";
}

std::vector<uint32_t> readShaderFile(const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open shader file: " << path
               << " (SHADER_DIR=" << getShaderDir() << ")";
    return {};
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  if (file_size == 0) {
    LOG(ERROR) << "Shader file is empty: " << path;
    return {};
  }
  if (file_size % sizeof(uint32_t) != 0) {
    LOG(ERROR) << "Shader file size is not 4-byte aligned: " << path;
    return {};
  }

  std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), file_size);

  if (!file) {
    LOG(ERROR) << "Failed to read shader file: " << path;
    return {};
  }

  return buffer;
}

VkShaderModule createShaderModule(VkDevice device,
                                  const std::vector<uint32_t>& code) {
  if (device == VK_NULL_HANDLE) {
    LOG(ERROR) << "Cannot create shader module with null device";
    return VK_NULL_HANDLE;
  }
  if (code.empty()) {
    LOG(ERROR) << "Cannot create shader module from empty bytecode";
    return VK_NULL_HANDLE;
  }

  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size() * sizeof(uint32_t);
  create_info.pCode = code.data();

  VkShaderModule shader_module;
  if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) !=
      VK_SUCCESS) {
    LOG(ERROR) << "Failed to create shader module";
    return VK_NULL_HANDLE;
  }

  return shader_module;
}

ShaderPair loadShaderPair(VkDevice device, const std::string& name) {
  ShaderPair pair;
  std::string shader_dir = getShaderDir();

  if (shader_dir.empty()) {
    LOG(ERROR) << "Shader directory not configured";
    return pair;
  }

  // Load vertex shader
  std::string vert_path = shader_dir + "/" + name + ".vert.spv";
  auto vert_code = readShaderFile(vert_path);
  if (vert_code.empty()) {
    LOG(ERROR) << "Failed to load vertex shader: " << vert_path
               << "\n  Hint: Run 'make shaders' or rebuild the project if you "
                  "modified shader source files.";
    return pair;
  }
  pair.vert = createShaderModule(device, vert_code);
  if (pair.vert == VK_NULL_HANDLE) {
    return pair;
  }

  // Load fragment shader
  std::string frag_path = shader_dir + "/" + name + ".frag.spv";
  auto frag_code = readShaderFile(frag_path);
  if (frag_code.empty()) {
    LOG(ERROR) << "Failed to load fragment shader: " << frag_path
               << "\n  Hint: Run 'make shaders' or rebuild the project if you "
                  "modified shader source files.";
    pair.destroy(device);
    return pair;
  }
  pair.frag = createShaderModule(device, frag_code);
  if (pair.frag == VK_NULL_HANDLE) {
    pair.destroy(device);
    return pair;
  }

  return pair;
}

}  // namespace renderer
}  // namespace nvblox
