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

#include <vulkan/vulkan.h>

#include "nvblox/renderer/visualizers/visualizer_interface.h"

namespace nvblox {
namespace renderer {

class VkContext;
struct ShaderPair;

/// Base class for visualizers providing common functionality.
/// Derived classes should call BaseVisualizer::init() first in their init(),
/// and BaseVisualizer::destroyBase() last in their destroy().
///
/// Provides:
/// - Context storage (ctx_)
/// - Pipeline layout management (pipeline_layout_)
/// - Common helper methods for pipeline creation
class BaseVisualizer : public VisualizerInterface {
 public:
  ~BaseVisualizer() override = default;

  // Non-copyable
  BaseVisualizer(const BaseVisualizer&) = delete;
  BaseVisualizer& operator=(const BaseVisualizer&) = delete;

  /// Get the current state of the visualizer.
  /// @return Current lifecycle state.
  VisualizerState state() const { return state_; }

  /// Check if the visualizer is ready to render.
  /// @return True if state is kReady.
  bool isReady() const { return state_ == VisualizerState::kReady; }

 protected:
  BaseVisualizer() = default;

  /// Initialize the base visualizer (stores context).
  /// Derived classes should call this first in their init().
  /// @param ctx Vulkan context.
  /// @return True if initialization succeeded.
  bool initBase(VkContext* ctx);

  /// Destroy base resources (pipeline layout).
  /// Derived classes should call this in their destroy() after destroying
  /// their own pipelines.
  /// @note Derived classes MUST call ctx_->waitIdle() before destroying any
  ///       resources. This method does not call waitIdle() to avoid redundant
  ///       waits when derived classes have already done so.
  void destroyBase();

  /// Create a pipeline layout with push constants only (no descriptor sets).
  /// @param push_constant_size Size of push constants in bytes.
  /// @param push_constant_stages Shader stages that use push constants.
  /// @return True if creation succeeded.
  bool createPipelineLayout(uint32_t push_constant_size,
                            VkShaderStageFlags push_constant_stages);

  /// Create a pipeline layout with descriptor sets and push constants.
  /// @param descriptor_set_layouts Array of descriptor set layouts.
  /// @param layout_count Number of descriptor set layouts.
  /// @param push_constant_size Size of push constants in bytes (0 if none).
  /// @param push_constant_stages Shader stages that use push constants.
  /// @return True if creation succeeded.
  bool createPipelineLayoutWithDescriptors(
      const VkDescriptorSetLayout* descriptor_set_layouts,
      uint32_t layout_count, uint32_t push_constant_size,
      VkShaderStageFlags push_constant_stages);

  /// Load a shader pair by name using the shader utility.
  /// @param name Base name of the shader (e.g., "mesh", "point_cloud").
  /// @return ShaderPair with both modules, caller must destroy after use.
  ShaderPair loadShaders(const std::string& name);

  /// Destroy a pipeline if it exists and set it to VK_NULL_HANDLE.
  /// @param pipeline Pointer to pipeline handle to destroy.
  void destroyPipeline(VkPipeline* pipeline);

  VkContext* ctx_ = nullptr;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VisualizerState state_ = VisualizerState::kUninitialized;
};

}  // namespace renderer
}  // namespace nvblox
