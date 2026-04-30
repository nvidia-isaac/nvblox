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

#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_buffer.h"
#include "nvblox/renderer/core/shared_texture.h"
#include "nvblox/renderer/kernels/mesh_to_vertex.h"
#include "nvblox/renderer/utils/push_constants.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/visualizers/buffered_visualizer.h"

namespace nvblox {
namespace renderer {

class VkContext;
class ViewCamera;
class PipelineBuilder;
struct ShaderPair;

/// Visualizer for colored triangle meshes.
/// Renders indexed triangles with per-vertex position and color.
/// @note NOT thread-safe. See VisualizerInterface for thread safety
/// requirements.
class MeshVisualizer : public BufferedVisualizer<MeshVisualizer, MeshVertex> {
 public:
  /// Vertex type alias (uses MeshVertex from mesh_to_vertex.h).
  using Vertex = MeshVertex;

  MeshVisualizer() = default;
  ~MeshVisualizer() override;

  // Non-copyable, non-movable (prevent accidental moves of Vulkan resources)
  MeshVisualizer(const MeshVisualizer&) = delete;
  MeshVisualizer& operator=(const MeshVisualizer&) = delete;
  MeshVisualizer(MeshVisualizer&&) = delete;
  MeshVisualizer& operator=(MeshVisualizer&&) = delete;

  // VisualizerInterface methods
  bool init(VkContext* ctx) override;
  void destroy() override;
  void render(VkCommandBuffer cmd, const ViewCamera* camera,
              uint32_t viewport_width, uint32_t viewport_height) override;

  /// Render with an explicit view-projection matrix and viewport rect.
  /// Used by renderViews() for XR multi-view rendering.
  void render(VkCommandBuffer cmd, const float* view_proj, uint32_t x,
              uint32_t y, uint32_t width, uint32_t height);

  bool hasData() const override {
    return num_indices_ > 0 && vertexBuffer() && index_buffer_;
  }

  /// Update mesh from separate vertex position, color, UV, and triangle arrays.
  /// @param positions CUDA device pointer to float3 positions (x, y, z per
  /// vertex).
  /// @param colors CUDA device pointer to uint8 colors (r, g, b per vertex - 3
  /// bytes).
  /// @param uvs CUDA device pointer to float2 UVs (u, v per vertex), or
  ///            nullptr if no UVs (all vertices use vertex color).
  /// @param triangles CUDA device pointer to int triangle indices (3 ints per
  /// triangle).
  /// @param num_vertices Number of vertices.
  /// @param num_triangles Number of triangles.
  /// @param stream CUDA stream.
  ///
  /// @note Triangle indices are assumed to be valid (in range [0,
  /// num_vertices)).
  ///       Passing out-of-bounds indices results in undefined rendering
  ///       behavior. Index validation is not performed at runtime for
  ///       performance reasons (indices reside on GPU memory). The caller is
  ///       responsible for ensuring index validity.
  void updateMesh(const float* positions, const uint8_t* colors,
                  const int* triangles, size_t num_vertices,
                  size_t num_triangles, const CudaStream& stream,
                  const float* uvs = nullptr);

  /// Update the texture atlas for projective texture mapping.
  /// @param src CUDA device pointer to texture data. Accepts both RGB8 (3 bytes
  ///        per pixel) and RGBA8 (4 bytes per pixel) — internally stored as
  ///        RGBA via SharedTexture format conversion.
  /// @param width Texture width in pixels.
  /// @param height Texture height in pixels.
  /// @param stream CUDA stream.
  /// @return True if upload succeeded.
  bool updateTexture(const void* src, uint32_t width, uint32_t height,
                     const CudaStream& stream);

  /// Check if a texture atlas is currently bound.
  bool hasTexture() const { return has_texture_; }

  /// Get number of vertices.
  size_t numVertices() const { return num_vertices_; }

  /// Get number of triangles.
  size_t numTriangles() const { return num_indices_ / 3; }

  /// Set wireframe mode.
  void setWireframe(bool enabled) { wireframe_ = enabled; }

  /// Get wireframe mode.
  bool wireframe() const { return wireframe_; }

  /// Toggle wireframe mode.
  void toggleWireframe() { wireframe_ = !wireframe_; }

  /// @cond INTERNAL
  // =========================================================================
  // BufferedVisualizer template requirements
  // =========================================================================

  /// Shader base name.
  static constexpr const char* shaderName() { return "mesh"; }

  /// Configure pipeline vertex attributes and topology.
  void configurePipeline(PipelineBuilder& builder) const;

  /// Push constant size in bytes.
  static constexpr uint32_t pushConstantSize() {
    return sizeof(MeshPushConstants);
  }

  /// Shader stages using push constants.
  static constexpr VkShaderStageFlags pushConstantStages() {
    return VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  /// Default buffer size in elements.
  static constexpr size_t defaultBufferElementCount() {
    return kDefaultVertexBufferSize;
  }

  /// Maximum element count.
  static constexpr size_t maxElementCount() { return kMaxVertexCount; }

  /// Visualizer name for logging.
  static constexpr const char* visualizerName() { return "MeshVisualizer"; }
  /// @endcond

 private:
  /// Create a single pipeline with a specific polygon mode.
  bool createPipeline(VkPolygonMode polygon_mode, const ShaderPair& shaders,
                      VkPipeline* pipeline);
  bool createPipelines();

  /// Create descriptor set layout, pool, and set for texture binding.
  bool createTextureDescriptorSet();

  /// Update descriptor set to point to current texture.
  void updateTextureDescriptorSet();

  // Index buffer (in addition to vertex buffer from base)
  std::unique_ptr<SharedBuffer> index_buffer_;

  // Pipelines (fill and wireframe)
  VkPipeline pipeline_fill_ = VK_NULL_HANDLE;
  VkPipeline pipeline_wireframe_ = VK_NULL_HANDLE;

  // Texture atlas for projective texture mapping (composition pattern)
  std::unique_ptr<SharedTexture> mesh_texture_;
  VkDescriptorSetLayout texture_desc_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool texture_desc_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet texture_desc_set_ = VK_NULL_HANDLE;
  bool has_texture_ = false;

  // Settings
  bool wireframe_ = false;

  // Mesh stats
  size_t num_vertices_ = 0;
  size_t num_indices_ = 0;
};

}  // namespace renderer
}  // namespace nvblox
