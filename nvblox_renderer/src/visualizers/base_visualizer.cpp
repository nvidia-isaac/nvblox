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

/// @file base_visualizer.cpp
/// @brief Implementation of BaseVisualizer common functionality.
///
/// ## Error Handling
///
/// This file uses recoverable error handling (checkVkResult) for all Vulkan
/// operations. Initialization failures are recoverable - callers can check
/// the state() and retry or fall back. On failure, state_ is set to kError.
/// Fatal errors (checkVkErrors) are reserved for programming bugs.

#include "nvblox/renderer/visualizers/base_visualizer.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/shader_utils.h"

namespace nvblox {
namespace renderer {

bool BaseVisualizer::initBase(VkContext* ctx) {
  if (!ctx) {
    LOG(ERROR) << "BaseVisualizer::initBase: null context";
    state_ = VisualizerState::kError;
    return false;
  }
  ctx_ = ctx;
  state_ = VisualizerState::kReady;
  return true;
}

void BaseVisualizer::destroyBase() {
  // Note: Derived classes are responsible for calling ctx_->waitIdle() before
  // destroying their resources. We don't call it here to avoid redundant waits.
  if (ctx_) {
    VkDevice device = ctx_->device();

    if (pipeline_layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
      pipeline_layout_ = VK_NULL_HANDLE;
    }
  }
  ctx_ = nullptr;
  state_ = VisualizerState::kUninitialized;
}

bool BaseVisualizer::createPipelineLayout(
    uint32_t push_constant_size, VkShaderStageFlags push_constant_stages) {
  return createPipelineLayoutWithDescriptors(nullptr, 0, push_constant_size,
                                             push_constant_stages);
}

bool BaseVisualizer::createPipelineLayoutWithDescriptors(
    const VkDescriptorSetLayout* descriptor_set_layouts, uint32_t layout_count,
    uint32_t push_constant_size, VkShaderStageFlags push_constant_stages) {
  if (!ctx_) {
    LOG(ERROR) << "BaseVisualizer: context not set";
    return false;
  }

  VkDevice device = ctx_->device();

  VkPushConstantRange push_constant{};
  push_constant.stageFlags = push_constant_stages;
  push_constant.offset = 0;
  push_constant.size = push_constant_size;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = layout_count;
  layout_info.pSetLayouts = descriptor_set_layouts;

  if (push_constant_size > 0) {
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;
  } else {
    layout_info.pushConstantRangeCount = 0;
    layout_info.pPushConstantRanges = nullptr;
  }

  VkResult result =
      vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_);
  if (!checkVkResult(result, "vkCreatePipelineLayout")) {
    state_ = VisualizerState::kError;
    return false;
  }
  return true;
}

ShaderPair BaseVisualizer::loadShaders(const std::string& name) {
  if (!ctx_) {
    LOG(ERROR) << "BaseVisualizer::loadShaders: context not set";
    return ShaderPair{};
  }
  return loadShaderPair(ctx_->device(), name);
}

void BaseVisualizer::destroyPipeline(VkPipeline* pipeline) {
  if (pipeline && *pipeline != VK_NULL_HANDLE && ctx_) {
    vkDestroyPipeline(ctx_->device(), *pipeline, nullptr);
    *pipeline = VK_NULL_HANDLE;
  }
}

}  // namespace renderer
}  // namespace nvblox
