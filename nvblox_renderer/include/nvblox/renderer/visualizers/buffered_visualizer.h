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

#include <memory>
#include <string>
#include <type_traits>

#include <glog/logging.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_buffer.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/utils/shader_utils.h"
#include "nvblox/renderer/visualizers/base_visualizer.h"
#include "nvblox/renderer/visualizers/pipeline_builder.h"

namespace nvblox {
namespace renderer {

/// Template base class for visualizers that use vertex buffers.
///
/// This template provides common functionality for buffer-based visualizers:
/// - Primary vertex buffer management with automatic resizing
/// - Pipeline creation helpers
/// - Common destroy pattern
///
/// Derived classes must implement:
/// - `shaderName()` - Returns the shader base name (e.g., "point_cloud",
/// "mesh")
/// - `configurePipeline(PipelineBuilder&)` - Configure vertex attributes,
/// topology, etc.
/// - `pushConstantSize()` - Size of push constants in bytes
/// - `pushConstantStages()` - Shader stages that use push constants
/// - `defaultBufferElementCount()` - Initial buffer size in elements
/// - `maxElementCount()` - Maximum allowed element count
/// - `visualizerName()` - Name for logging (e.g., "PointCloudVisualizer")
///
/// @tparam Derived The derived class type.
/// @tparam VertexType The vertex structure type.
template <typename Derived, typename VertexType>
class BufferedVisualizer : public BaseVisualizer {
 public:
  using Vertex = VertexType;

  ~BufferedVisualizer() override = default;

  // Non-copyable, non-movable
  BufferedVisualizer(const BufferedVisualizer&) = delete;
  BufferedVisualizer& operator=(const BufferedVisualizer&) = delete;
  BufferedVisualizer(BufferedVisualizer&&) = delete;
  BufferedVisualizer& operator=(BufferedVisualizer&&) = delete;

 protected:
  BufferedVisualizer() = default;

  /// Initialize the vertex buffer and pipeline.
  /// Derived classes should call this in their init() implementation,
  /// then initialize any additional resources.
  /// @param ctx Vulkan context.
  /// @return True if initialization succeeded.
  bool initBuffered(VkContext* ctx);

  /// Create a graphics pipeline using the derived class configuration.
  /// @param pipeline Output pipeline handle.
  /// @return True if creation succeeded.
  bool createPipelineBuffered(VkPipeline* pipeline);

  /// Destroy the vertex buffer and base resources.
  /// Derived classes should call ctx_->waitIdle() and destroy their specific
  /// resources before calling this.
  void destroyBuffered();

  /// Ensure the vertex buffer is large enough to hold the given number of
  /// elements. Resizes if needed using the growth factor.
  /// @param num_elements Required number of elements.
  /// @return True if buffer is ready, false on validation or resize failure.
  bool ensureBufferCapacity(size_t num_elements);

  /// Copy data to the vertex buffer from CUDA memory.
  /// @param src Source pointer on device.
  /// @param num_elements Number of elements to copy.
  /// @param stream CUDA stream.
  /// @return True if copy succeeded.
  bool copyToVertexBuffer(const Vertex* src, size_t num_elements,
                          const CudaStream& stream);

  /// Get the vertex buffer.
  SharedBuffer* vertexBuffer() { return vertex_buffer_.get(); }
  const SharedBuffer* vertexBuffer() const { return vertex_buffer_.get(); }

 private:
  /// Get derived class reference.
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  std::unique_ptr<SharedBuffer> vertex_buffer_;
};

}  // namespace renderer
}  // namespace nvblox

#include "nvblox/renderer/visualizers/impl/buffered_visualizer_impl.h"
