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

#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/utils/push_constants.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/visualizers/buffered_visualizer.h"

namespace nvblox {
namespace renderer {

class VkContext;
class ViewCamera;
class PipelineBuilder;

/// Point data structure (matches nvblox Pointcloud format).
struct PointCloudPoint {
  float x, y, z;       // Position
  uint8_t r, g, b, a;  // Color (RGBA)
};

/// Visualizer for colored point clouds.
/// Renders points with per-vertex position and color.
/// @note NOT thread-safe. See VisualizerInterface for thread safety
/// requirements.
class PointCloudVisualizer
    : public BufferedVisualizer<PointCloudVisualizer, PointCloudPoint> {
 public:
  /// Point type alias for backward compatibility.
  using Point = PointCloudPoint;

  PointCloudVisualizer() = default;
  ~PointCloudVisualizer() override;

  // Non-copyable, non-movable (prevent accidental moves of Vulkan resources)
  PointCloudVisualizer(const PointCloudVisualizer&) = delete;
  PointCloudVisualizer& operator=(const PointCloudVisualizer&) = delete;
  PointCloudVisualizer(PointCloudVisualizer&&) = delete;
  PointCloudVisualizer& operator=(PointCloudVisualizer&&) = delete;

  // VisualizerInterface methods
  bool init(VkContext* ctx) override;
  void destroy() override;
  void render(VkCommandBuffer cmd, const ViewCamera* camera,
              uint32_t viewport_width, uint32_t viewport_height) override;

  /// Render with an explicit view-projection matrix and viewport rect.
  void render(VkCommandBuffer cmd, const float* view_proj, float point_size,
              uint32_t x, uint32_t y, uint32_t width, uint32_t height);

  bool hasData() const override { return num_points_ > 0 && vertexBuffer(); }

  /// Update point cloud from CUDA memory.
  /// @param points_ptr CUDA device pointer to Point array.
  /// @param num_points Number of points.
  /// @param stream CUDA stream.
  void updatePoints(const Point* points_ptr, size_t num_points,
                    const CudaStream& stream);

  /// Set point size.
  void setPointSize(float size) { point_size_ = size; }

  /// Get point size.
  float pointSize() const { return point_size_; }

  /// Get number of points.
  size_t numPoints() const { return num_points_; }

  /// @cond INTERNAL
  // =========================================================================
  // BufferedVisualizer template requirements
  // =========================================================================

  /// Shader base name.
  static constexpr const char* shaderName() { return "point_cloud"; }

  /// Configure pipeline vertex attributes and topology.
  void configurePipeline(PipelineBuilder& builder) const;

  /// Push constant size in bytes.
  static constexpr uint32_t pushConstantSize() {
    return sizeof(PointCloudPushConstants);
  }

  /// Shader stages using push constants.
  static constexpr VkShaderStageFlags pushConstantStages() {
    return VK_SHADER_STAGE_VERTEX_BIT;
  }

  /// Default buffer size in elements.
  static constexpr size_t defaultBufferElementCount() {
    return kDefaultPointBufferSize;
  }

  /// Maximum element count.
  static constexpr size_t maxElementCount() { return kMaxPointCount; }

  /// Visualizer name for logging.
  static constexpr const char* visualizerName() {
    return "PointCloudVisualizer";
  }
  /// @endcond

 private:
  bool createPipeline();

  // Pipeline
  VkPipeline pipeline_ = VK_NULL_HANDLE;

  // Settings
  float point_size_ = 2.0f;
  size_t num_points_ = 0;
};

}  // namespace renderer
}  // namespace nvblox
