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

/// @file pipeline_builder.cpp
/// @brief Implementation of PipelineBuilder for Vulkan graphics pipelines.
///
/// ## Error Handling
///
/// This file uses recoverable error handling (checkVkResult) for all Vulkan
/// operations. Pipeline creation failures are recoverable - callers can retry
/// with different parameters or fall back gracefully. Fatal errors
/// (checkVkErrors) are reserved for programming bugs that indicate
/// unrecoverable state.

#include "nvblox/renderer/visualizers/pipeline_builder.h"

#include <array>

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"

namespace nvblox {
namespace renderer {

void PipelineBuilder::reset() {
  vert_module_ = VK_NULL_HANDLE;
  frag_module_ = VK_NULL_HANDLE;
  vertex_bindings_.clear();
  vertex_attributes_.clear();
  topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  polygon_mode_ = VK_POLYGON_MODE_FILL;
  cull_mode_ = VK_CULL_MODE_BACK_BIT;
  depth_test_enable_ = true;
  depth_write_enable_ = true;
  depth_compare_op_ = VK_COMPARE_OP_LESS;
  blend_enable_ = false;
  src_color_blend_ = VK_BLEND_FACTOR_SRC_ALPHA;
  dst_color_blend_ = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  src_alpha_blend_ = VK_BLEND_FACTOR_ONE;
  dst_alpha_blend_ = VK_BLEND_FACTOR_ZERO;
  dynamic_states_ = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
}

PipelineBuilder& PipelineBuilder::setShaders(VkShaderModule vert_module,
                                             VkShaderModule frag_module) {
  vert_module_ = vert_module;
  frag_module_ = frag_module;
  return *this;
}

PipelineBuilder& PipelineBuilder::addVertexBinding(
    uint32_t binding, uint32_t stride, VkVertexInputRate input_rate) {
  VkVertexInputBindingDescription desc{};
  desc.binding = binding;
  desc.stride = stride;
  desc.inputRate = input_rate;
  vertex_bindings_.push_back(desc);
  return *this;
}

PipelineBuilder& PipelineBuilder::addVertexAttribute(uint32_t location,
                                                     uint32_t binding,
                                                     VkFormat format,
                                                     uint32_t offset) {
  VkVertexInputAttributeDescription desc{};
  desc.location = location;
  desc.binding = binding;
  desc.format = format;
  desc.offset = offset;
  vertex_attributes_.push_back(desc);
  return *this;
}

PipelineBuilder& PipelineBuilder::setTopology(VkPrimitiveTopology topology) {
  topology_ = topology;
  return *this;
}

PipelineBuilder& PipelineBuilder::setPolygonMode(VkPolygonMode mode) {
  polygon_mode_ = mode;
  return *this;
}

PipelineBuilder& PipelineBuilder::setCullMode(VkCullModeFlags cull_mode) {
  cull_mode_ = cull_mode;
  return *this;
}

PipelineBuilder& PipelineBuilder::setDepthTest(bool enable, bool write_enable) {
  depth_test_enable_ = enable;
  depth_write_enable_ = write_enable;
  return *this;
}

PipelineBuilder& PipelineBuilder::setDepthCompareOp(VkCompareOp op) {
  depth_compare_op_ = op;
  return *this;
}

PipelineBuilder& PipelineBuilder::setBlending(bool enable) {
  blend_enable_ = enable;
  return *this;
}

PipelineBuilder& PipelineBuilder::setBlendFactors(VkBlendFactor src_color,
                                                  VkBlendFactor dst_color,
                                                  VkBlendFactor src_alpha,
                                                  VkBlendFactor dst_alpha) {
  src_color_blend_ = src_color;
  dst_color_blend_ = dst_color;
  src_alpha_blend_ = src_alpha;
  dst_alpha_blend_ = dst_alpha;
  return *this;
}

PipelineBuilder& PipelineBuilder::addDynamicState(VkDynamicState state) {
  dynamic_states_.push_back(state);
  return *this;
}

PipelineBuilder& PipelineBuilder::clearDynamicStates() {
  dynamic_states_.clear();
  return *this;
}

VkPipeline PipelineBuilder::build(VkDevice device, VkPipelineLayout layout,
                                  VkRenderPass render_pass, uint32_t subpass,
                                  VkPipelineCache cache) {
  // Validate required parameters
  if (device == VK_NULL_HANDLE) {
    LOG(ERROR) << "PipelineBuilder::build: null device";
    return VK_NULL_HANDLE;
  }
  if (layout == VK_NULL_HANDLE) {
    LOG(ERROR) << "PipelineBuilder::build: null pipeline layout";
    return VK_NULL_HANDLE;
  }
  if (render_pass == VK_NULL_HANDLE) {
    LOG(ERROR) << "PipelineBuilder::build: null render pass";
    return VK_NULL_HANDLE;
  }
  if (vert_module_ == VK_NULL_HANDLE || frag_module_ == VK_NULL_HANDLE) {
    LOG(ERROR) << "PipelineBuilder: shader modules not set";
    return VK_NULL_HANDLE;
  }

  // Shader stages
  std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};
  shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  shader_stages[0].module = vert_module_;
  shader_stages[0].pName = "main";

  shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shader_stages[1].module = frag_module_;
  shader_stages[1].pName = "main";

  // Vertex input
  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount =
      static_cast<uint32_t>(vertex_bindings_.size());
  vertex_input.pVertexBindingDescriptions = vertex_bindings_.data();
  vertex_input.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(vertex_attributes_.size());
  vertex_input.pVertexAttributeDescriptions = vertex_attributes_.data();

  // Input assembly
  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = topology_;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  // Viewport state (dynamic, so we just set counts)
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  // Rasterization
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = polygon_mode_;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = cull_mode_;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  // Multisampling
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Depth stencil
  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = depth_test_enable_ ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = depth_write_enable_ ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = depth_compare_op_;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  // Color blending
  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = blend_enable_ ? VK_TRUE : VK_FALSE;
  if (blend_enable_) {
    color_blend_attachment.srcColorBlendFactor = src_color_blend_;
    color_blend_attachment.dstColorBlendFactor = dst_color_blend_;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = src_alpha_blend_;
    color_blend_attachment.dstAlphaBlendFactor = dst_alpha_blend_;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
  }

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  // Dynamic state
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount =
      static_cast<uint32_t>(dynamic_states_.size());
  dynamic_state.pDynamicStates = dynamic_states_.data();

  // Pipeline create info
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
  pipeline_info.pStages = shader_stages.data();
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.layout = layout;
  pipeline_info.renderPass = render_pass;
  pipeline_info.subpass = subpass;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  VkPipeline pipeline;
  VkResult result = vkCreateGraphicsPipelines(device, cache, 1, &pipeline_info,
                                              nullptr, &pipeline);
  if (!checkVkResult(result, "vkCreateGraphicsPipelines")) {
    return VK_NULL_HANDLE;
  }
  return pipeline;
}

void setViewportAndScissor(VkCommandBuffer cmd, uint32_t width,
                           uint32_t height) {
  setViewportAndScissor(cmd, 0, 0, width, height);
}

void setViewportAndScissor(VkCommandBuffer cmd, uint32_t x, uint32_t y,
                           uint32_t width, uint32_t height) {
  // Validate dimensions - Vulkan requires viewport dimensions > 0
  if (width == 0 || height == 0) {
    LOG(WARNING) << "setViewportAndScissor: invalid dimensions " << width << "x"
                 << height << ", using 1x1 fallback";
    width = 1;
    height = 1;
  }

  VkViewport viewport{};
  viewport.x = static_cast<float>(x);
  viewport.y = static_cast<float>(y);
  viewport.width = static_cast<float>(width);
  viewport.height = static_cast<float>(height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
  scissor.extent = {width, height};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

}  // namespace renderer
}  // namespace nvblox
