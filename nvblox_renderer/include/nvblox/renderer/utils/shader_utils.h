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
#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace nvblox {
namespace renderer {

/// Get the shader directory path.
/// @return Path to compiled shaders (SHADER_DIR), or empty string if not
/// defined.
std::string getShaderDir();

/// Read a SPIR-V shader file from disk.
/// @param path Full path to the .spv file.
/// @return SPIR-V bytecode as uint32_t (4-byte aligned), or empty on failure.
std::vector<uint32_t> readShaderFile(const std::string& path);

/// Create a Vulkan shader module from SPIR-V bytecode.
/// @param device Vulkan logical device.
/// @param code SPIR-V bytecode as uint32_t (must not be empty).
/// @return Shader module, or VK_NULL_HANDLE on failure.
VkShaderModule createShaderModule(VkDevice device,
                                  const std::vector<uint32_t>& code);

/// A pair of vertex and fragment shader modules.
struct ShaderPair {
  VkShaderModule vert = VK_NULL_HANDLE;
  VkShaderModule frag = VK_NULL_HANDLE;

  /// Check if both shaders are valid.
  bool isValid() const {
    return vert != VK_NULL_HANDLE && frag != VK_NULL_HANDLE;
  }

  /// Destroy both shader modules.
  void destroy(VkDevice device) {
    if (vert != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, vert, nullptr);
      vert = VK_NULL_HANDLE;
    }
    if (frag != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, frag, nullptr);
      frag = VK_NULL_HANDLE;
    }
  }
};

/// Load a shader pair (vertex + fragment) by name.
/// Looks for {name}.vert.spv and {name}.frag.spv in the shader directory.
/// @param device Vulkan logical device.
/// @param name Base name of the shader (e.g., "mesh", "point_cloud").
/// @return ShaderPair with both modules, or invalid pair on failure.
ShaderPair loadShaderPair(VkDevice device, const std::string& name);

}  // namespace renderer
}  // namespace nvblox
