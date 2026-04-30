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

#include "nvblox/renderer/visualizers/image_visualizer.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/push_constants.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/pipeline_builder.h"

namespace nvblox {
namespace renderer {

// Define constexpr in cpp for ODR-use
constexpr uint32_t ImageVisualizer::pushConstantSize() {
  return sizeof(ImagePushConstants);
}

ImageVisualizer::~ImageVisualizer() { destroy(); }

void ImageVisualizer::destroy() {
  // Ensure GPU is idle before destroying resources
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Destroy pipeline before base (which destroys layout)
  destroyPipeline(&pipeline_);

  // Destroy textures, descriptor sets, and base resources
  destroyTextured();
}

bool ImageVisualizer::init(VkContext* ctx) {
  // Initialize with placeholder 1x1 textures.
  // Actual textures will be created when updateDepthImage/updateColorImage is
  // called.
  return init(ctx, ImageVisualizerConfig{});
}

bool ImageVisualizer::init(VkContext* ctx,
                           const ImageVisualizerConfig& config) {
  // Validate texture dimensions before initializing
  auto validateDimension = [](uint32_t dim, const char* name) -> bool {
    if (dim < kMinTextureDimension) {
      LOG(ERROR) << "ImageVisualizer: " << name << " dimension " << dim
                 << " is below minimum " << kMinTextureDimension;
      return false;
    }
    if (dim > kMaxTextureDimension) {
      LOG(ERROR) << "ImageVisualizer: " << name << " dimension " << dim
                 << " exceeds maximum " << kMaxTextureDimension;
      return false;
    }
    return true;
  };

  if (!validateDimension(config.depth_width, "depth width") ||
      !validateDimension(config.depth_height, "depth height") ||
      !validateDimension(config.color_width, "color width") ||
      !validateDimension(config.color_height, "color height")) {
    return false;
  }

  // Use TexturedVisualizer base class for texture and descriptor set creation
  // Textures are created with default 1x1 dimensions, then resized below
  if (!initTextured(ctx)) {
    return false;
  }

  // Resize textures to actual dimensions if different from default 1x1
  if (config.depth_width != 1 || config.depth_height != 1) {
    if (!resizeTexture(kDepthTexture, config.depth_width,
                       config.depth_height)) {
      return false;
    }
  }
  if (config.color_width != 1 || config.color_height != 1) {
    if (!resizeTexture(kColorTexture, config.color_width,
                       config.color_height)) {
      return false;
    }
  }

  if (!createPipeline()) {
    return false;
  }

  LOG(INFO) << "ImageVisualizer initialized";
  return true;
}

void ImageVisualizer::configurePipeline(PipelineBuilder& builder) const {
  // No vertex bindings - vertices generated in shader
  builder.setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      .setCullMode(VK_CULL_MODE_NONE)
      .setDepthTest(false, false);  // 2D image rendering, no depth
}

bool ImageVisualizer::createPipeline() {
  return createPipelineTextured(&pipeline_);
}

void ImageVisualizer::render(VkCommandBuffer cmd, const ViewCamera* /*camera*/,
                             uint32_t viewport_width,
                             uint32_t viewport_height) {
  render(cmd, 0, 0, viewport_width, viewport_height);
}

void ImageVisualizer::render(VkCommandBuffer cmd, uint32_t x, uint32_t y,
                             uint32_t width, uint32_t height) {
  if (!hasData() || !pipeline_) {
    return;
  }

  // Bind pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

  // Use descriptor set from base class
  VkDescriptorSet desc_set = descriptorSet();
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline_layout_, 0, 1, &desc_set, 0, nullptr);

  // Set viewport and scissor to the requested sub-region
  setViewportAndScissor(cmd, x, y, width, height);

  // Push constants
  ImagePushConstants push_constants;
  push_constants.minDepth = min_depth_;
  push_constants.maxDepth = max_depth_;
  push_constants.colormap = static_cast<float>(colormap_);
  push_constants.displayLayout = static_cast<float>(layout_);
  vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(push_constants), &push_constants);

  // Draw fullscreen quad (6 vertices, 2 triangles)
  vkCmdDraw(cmd, 6, 1, 0, 0);
}

}  // namespace renderer
}  // namespace nvblox
