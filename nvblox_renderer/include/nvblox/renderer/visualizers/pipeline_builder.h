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

/// Helper class for building Vulkan graphics pipelines.
/// Reduces boilerplate code across visualizers by providing sensible defaults
/// and a fluent interface for customization.
class PipelineBuilder {
 public:
  PipelineBuilder() { reset(); }

  /// Reset to default state for building a new pipeline.
  void reset();

  /// Set shader modules.
  /// @param vert_module Vertex shader module (required).
  /// @param frag_module Fragment shader module (required).
  PipelineBuilder& setShaders(VkShaderModule vert_module,
                              VkShaderModule frag_module);

  /// Add a vertex binding description.
  PipelineBuilder& addVertexBinding(
      uint32_t binding, uint32_t stride,
      VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX);

  /// Add a vertex attribute description.
  PipelineBuilder& addVertexAttribute(uint32_t location, uint32_t binding,
                                      VkFormat format, uint32_t offset);

  /// Set primitive topology (default: TRIANGLE_LIST).
  PipelineBuilder& setTopology(VkPrimitiveTopology topology);

  /// Set polygon mode (default: FILL).
  PipelineBuilder& setPolygonMode(VkPolygonMode mode);

  /// Set cull mode (default: BACK).
  PipelineBuilder& setCullMode(VkCullModeFlags cull_mode);

  /// Enable/disable depth testing (default: enabled).
  PipelineBuilder& setDepthTest(bool enable, bool write_enable = true);

  /// Set depth compare operation (default: LESS).
  PipelineBuilder& setDepthCompareOp(VkCompareOp op);

  /// Enable/disable blending (default: disabled).
  PipelineBuilder& setBlending(bool enable);

  /// Set blending factors (for when blending is enabled).
  PipelineBuilder& setBlendFactors(VkBlendFactor src_color,
                                   VkBlendFactor dst_color,
                                   VkBlendFactor src_alpha,
                                   VkBlendFactor dst_alpha);

  /// Add a dynamic state (default: VIEWPORT and SCISSOR).
  PipelineBuilder& addDynamicState(VkDynamicState state);

  /// Clear all dynamic states.
  PipelineBuilder& clearDynamicStates();

  /// Build the pipeline.
  /// @param device Vulkan device.
  /// @param layout Pipeline layout.
  /// @param render_pass Render pass.
  /// @param subpass Subpass index (default: 0).
  /// @param cache Pipeline cache for faster creation (optional).
  /// @return Created pipeline, or VK_NULL_HANDLE on failure.
  VkPipeline build(VkDevice device, VkPipelineLayout layout,
                   VkRenderPass render_pass, uint32_t subpass = 0,
                   VkPipelineCache cache = VK_NULL_HANDLE);

 private:
  // Shader stages
  VkShaderModule vert_module_ = VK_NULL_HANDLE;
  VkShaderModule frag_module_ = VK_NULL_HANDLE;

  // Vertex input
  std::vector<VkVertexInputBindingDescription> vertex_bindings_;
  std::vector<VkVertexInputAttributeDescription> vertex_attributes_;

  // Input assembly
  VkPrimitiveTopology topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  // Rasterization
  VkPolygonMode polygon_mode_ = VK_POLYGON_MODE_FILL;
  VkCullModeFlags cull_mode_ = VK_CULL_MODE_BACK_BIT;

  // Depth stencil
  bool depth_test_enable_ = true;
  bool depth_write_enable_ = true;
  VkCompareOp depth_compare_op_ = VK_COMPARE_OP_LESS;

  // Color blending
  bool blend_enable_ = false;
  VkBlendFactor src_color_blend_ = VK_BLEND_FACTOR_SRC_ALPHA;
  VkBlendFactor dst_color_blend_ = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  VkBlendFactor src_alpha_blend_ = VK_BLEND_FACTOR_ONE;
  VkBlendFactor dst_alpha_blend_ = VK_BLEND_FACTOR_ZERO;

  // Dynamic state
  std::vector<VkDynamicState> dynamic_states_;
};

/// Set viewport and scissor for rendering.
/// @param cmd Command buffer.
/// @param width Viewport width.
/// @param height Viewport height.
void setViewportAndScissor(VkCommandBuffer cmd, uint32_t width,
                           uint32_t height);

/// Set viewport and scissor to a sub-region of the framebuffer.
/// @param cmd Command buffer.
/// @param x Viewport x offset in pixels.
/// @param y Viewport y offset in pixels.
/// @param width Viewport width in pixels.
/// @param height Viewport height in pixels.
void setViewportAndScissor(VkCommandBuffer cmd, uint32_t x, uint32_t y,
                           uint32_t width, uint32_t height);

}  // namespace renderer
}  // namespace nvblox
