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

#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"

#include <cstring>

#include <glog/logging.h>

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/push_constants.h"
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/pipeline_builder.h"

namespace nvblox {
namespace renderer {

PointCloudVisualizer::~PointCloudVisualizer() { destroy(); }

bool PointCloudVisualizer::init(VkContext* ctx) {
  // Use template base class for buffer initialization
  if (!initBuffered(ctx)) {
    return false;
  }

  if (!createPipeline()) {
    LOG(ERROR) << "Failed to create point cloud pipeline";
    return false;
  }

  LOG(INFO) << "PointCloudVisualizer initialized";
  return true;
}

void PointCloudVisualizer::destroy() {
  // Ensure GPU is idle before destroying resources
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Destroy pipeline before base (which destroys layout)
  destroyPipeline(&pipeline_);

  num_points_ = 0;

  // Destroy buffer and base resources
  destroyBuffered();
}

bool PointCloudVisualizer::createPipeline() {
  return createPipelineBuffered(&pipeline_);
}

void PointCloudVisualizer::configurePipeline(PipelineBuilder& builder) const {
  builder.addVertexBinding(0, sizeof(Point))
      .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Point, x))
      .addVertexAttribute(1, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(Point, r))
      .setTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
      .setCullMode(VK_CULL_MODE_NONE)
      .setDepthTest(true, true);
}

void PointCloudVisualizer::updatePoints(const Point* points_ptr,
                                        size_t num_points,
                                        const CudaStream& stream) {
  if (num_points == 0) {
    num_points_ = 0;
    return;
  }

  if (!points_ptr) {
    LOG(WARNING) << "updatePoints: null pointer provided";
    return;
  }

  // Use template helpers for buffer management
  if (!ensureBufferCapacity(num_points)) {
    return;
  }

  if (!copyToVertexBuffer(points_ptr, num_points, stream)) {
    return;
  }

  num_points_ = num_points;
}

void PointCloudVisualizer::render(VkCommandBuffer cmd, const ViewCamera* camera,
                                  uint32_t viewport_width,
                                  uint32_t viewport_height) {
  if (!hasData() || !camera) return;
  render(cmd, camera->viewProjMatrixPtr(), point_size_, 0, 0, viewport_width,
         viewport_height);
}

void PointCloudVisualizer::render(VkCommandBuffer cmd, const float* view_proj,
                                  float point_size, uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height) {
  if (!hasData() || !view_proj || !pipeline_) {
    return;
  }

  // Bind pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

  // Set viewport and scissor to the requested sub-region
  setViewportAndScissor(cmd, x, y, width, height);

  // Push constants (view-projection matrix + point size)
  PointCloudPushConstants push_constants;
  std::memcpy(push_constants.viewProj, view_proj,
              sizeof(push_constants.viewProj));
  push_constants.pointSize = point_size;

  vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(push_constants), &push_constants);

  // Bind vertex buffer and draw
  VkBuffer buffers[] = {vertexBuffer()->buffer()};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

  vkCmdDraw(cmd, static_cast<uint32_t>(num_points_), 1, 0, 0);
}

}  // namespace renderer
}  // namespace nvblox
