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

#include "nvblox/renderer/visualizers/mesh_visualizer.h"

#include <cstring>

#include <glog/logging.h>

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/push_constants.h"
#include "nvblox/renderer/utils/shader_utils.h"
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/pipeline_builder.h"

namespace nvblox {
namespace renderer {

MeshVisualizer::~MeshVisualizer() { destroy(); }

bool MeshVisualizer::init(VkContext* ctx) {
  // Use template base class for vertex buffer initialization
  if (!initBuffered(ctx)) {
    return false;
  }

  // Create index buffer (mesh-specific)
  index_buffer_ = std::make_unique<SharedBuffer>();
  if (!index_buffer_->create(ctx, kDefaultIndexBufferSize * sizeof(uint32_t),
                             SharedBuffer::Usage::kIndex)) {
    LOG(ERROR) << "MeshVisualizer: Failed to create index buffer";
    return false;
  }

  // Create texture descriptor set (for optional texture atlas)
  if (!createTextureDescriptorSet()) {
    LOG(ERROR) << "MeshVisualizer: Failed to create texture descriptor set";
    return false;
  }

  if (!createPipelines()) {
    LOG(ERROR) << "MeshVisualizer: Failed to create pipelines";
    return false;
  }

  LOG(INFO) << "MeshVisualizer initialized";
  return true;
}

void MeshVisualizer::destroy() {
  // Ensure GPU is idle before destroying resources
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Destroy pipelines before base (which destroys layout)
  destroyPipeline(&pipeline_fill_);
  destroyPipeline(&pipeline_wireframe_);

  // Destroy texture resources
  if (mesh_texture_) {
    mesh_texture_->destroy();
    mesh_texture_.reset();
  }
  if (ctx_ && ctx_->device()) {
    VkDevice device = ctx_->device();
    if (texture_desc_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, texture_desc_pool_, nullptr);
      texture_desc_pool_ = VK_NULL_HANDLE;
    }
    texture_desc_set_ = VK_NULL_HANDLE;
    if (texture_desc_layout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, texture_desc_layout_, nullptr);
      texture_desc_layout_ = VK_NULL_HANDLE;
    }
  }
  has_texture_ = false;

  // Destroy mesh-specific index buffer
  if (index_buffer_) {
    index_buffer_->destroy();
    index_buffer_.reset();
  }

  num_vertices_ = 0;
  num_indices_ = 0;

  // Destroy vertex buffer and base resources
  destroyBuffered();
}

bool MeshVisualizer::createPipelines() {
  if (!ctx_) {
    LOG(ERROR) << "MeshVisualizer: context not set";
    return false;
  }

  // Create pipeline layout with descriptor set for texture binding
  if (!createPipelineLayoutWithDescriptors(
          &texture_desc_layout_, 1, pushConstantSize(), pushConstantStages())) {
    return false;
  }

  // Load shaders once and reuse for both pipelines
  ShaderPair shaders = loadShaders(shaderName());
  if (!shaders.isValid()) {
    LOG(ERROR) << "MeshVisualizer: Failed to load shaders";
    return false;
  }

  // Create both pipelines with the same shader modules
  bool success = true;
  if (!createPipeline(VK_POLYGON_MODE_FILL, shaders, &pipeline_fill_)) {
    LOG(ERROR) << "MeshVisualizer: Failed to create fill pipeline";
    success = false;
  }

  if (success &&
      !createPipeline(VK_POLYGON_MODE_LINE, shaders, &pipeline_wireframe_)) {
    LOG(ERROR) << "MeshVisualizer: Failed to create wireframe pipeline";
    success = false;
  }

  // Destroy shaders after creating both pipelines
  shaders.destroy(ctx_->device());

  return success;
}

void MeshVisualizer::configurePipeline(PipelineBuilder& builder) const {
  builder.addVertexBinding(0, sizeof(Vertex))
      .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x))
      .addVertexAttribute(1, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(Vertex, r))
      .addVertexAttribute(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, u))
      .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      .setCullMode(VK_CULL_MODE_NONE)  // Reconstructed meshes may have
                                       // inconsistent winding
      .setDepthTest(true, true);
}

bool MeshVisualizer::createPipeline(VkPolygonMode polygon_mode,
                                    const ShaderPair& shaders,
                                    VkPipeline* pipeline) {
  VkDevice device = ctx_->device();

  // Build pipeline using PipelineBuilder with base configuration + polygon mode
  PipelineBuilder builder;
  builder.setShaders(shaders.vert, shaders.frag);
  configurePipeline(builder);
  builder.setPolygonMode(polygon_mode);

  *pipeline = builder.build(device, pipeline_layout_, ctx_->renderPass(), 0,
                            ctx_->pipelineCache());

  if (*pipeline == VK_NULL_HANDLE) {
    LOG(ERROR) << "MeshVisualizer: Failed to create graphics pipeline";
    return false;
  }

  return true;
}

void MeshVisualizer::updateMesh(const float* positions, const uint8_t* colors,
                                const int* triangles, size_t num_vertices,
                                size_t num_triangles, const CudaStream& stream,
                                const float* uvs) {
  if (num_vertices == 0 || num_triangles == 0) {
    num_vertices_ = 0;
    num_indices_ = 0;
    return;
  }

  if (!positions || !colors || !triangles) {
    LOG(WARNING) << "updateMesh: null pointer provided";
    return;
  }

  if (!index_buffer_) {
    LOG(ERROR) << "updateMesh: index buffer not initialized";
    return;
  }

  // Use template helper for vertex buffer management
  if (!ensureBufferCapacity(num_vertices)) {
    return;
  }

  // Validate index buffer size (3 indices per triangle)
  size_t num_indices = num_triangles * 3;
  auto index_validation = validateBufferSize(
      num_indices, sizeof(uint32_t), kMaxTriangleCount * 3, "MeshIndices");
  if (!index_validation.valid) {
    return;
  }

  // Resize index buffer if needed
  if (index_validation.required_size > index_buffer_->size()) {
    size_t new_size = calculateResizeCapacity(index_validation.required_size);
    LOG(INFO) << "MeshVisualizer: Resizing index buffer from "
              << index_buffer_->size() << " to " << new_size << " bytes";
    if (!index_buffer_->resize(new_size)) {
      LOG(ERROR) << "MeshVisualizer: Failed to resize index buffer";
      return;
    }
  }

  // Interleave position, color, and UV data directly into shared buffer.
  // vertexBuffer()->cudaPtr() points to GPU memory shared with Vulkan,
  // so no additional copy is needed after the interleave kernel.
  // uvs may be nullptr (all vertices get UV = -1,-1 → vertex color fallback).
  Vertex* dst = static_cast<Vertex*>(vertexBuffer()->cudaPtr());
  interleaveMeshVertexData(positions, colors, dst, num_vertices, stream, uvs);

  // Copy indices to shared buffer (need to convert int to uint32_t)
  // nvblox uses int for triangle indices, Vulkan expects uint32_t.
  // Platform requirement: int must be 32 bits (true on all supported platforms:
  // Linux x86_64, aarch64). This assumption is verified at compile time.
  static_assert(
      sizeof(int) == sizeof(uint32_t),
      "Platform not supported: int must be 32 bits (same size as uint32_t). "
      "nvblox uses int for triangle indices which are cast to uint32_t for "
      "Vulkan.");
  if (!index_buffer_->copyFromCuda(triangles, index_validation.required_size,
                                   stream)) {
    LOG(WARNING) << "MeshVisualizer: Failed to copy index data to buffer";
    return;
  }

  num_vertices_ = num_vertices;
  num_indices_ = num_indices;
}

void MeshVisualizer::render(VkCommandBuffer cmd, const ViewCamera* camera,
                            uint32_t viewport_width, uint32_t viewport_height) {
  if (!hasData() || !camera) return;
  render(cmd, camera->viewProjMatrixPtr(), 0, 0, viewport_width,
         viewport_height);
}

void MeshVisualizer::render(VkCommandBuffer cmd, const float* view_proj,
                            uint32_t x, uint32_t y, uint32_t width,
                            uint32_t height) {
  // Check all required resources before rendering
  VkPipeline active_pipeline =
      wireframe_ ? pipeline_wireframe_ : pipeline_fill_;
  if (!hasData() || !view_proj || !active_pipeline) {
    return;
  }

  // Bind pipeline (select fill or wireframe)
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, active_pipeline);

  // Set viewport and scissor to the requested sub-region
  setViewportAndScissor(cmd, x, y, width, height);

  // Push constants (view-projection matrix + texture flag)
  MeshPushConstants push_constants{};
  std::memcpy(push_constants.viewProj, view_proj,
              sizeof(push_constants.viewProj));
  push_constants.hasTexture = has_texture_ ? 1u : 0u;
  vkCmdPushConstants(cmd, pipeline_layout_,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push_constants), &push_constants);

  // Bind texture descriptor set (always bound; shader checks hasTexture flag)
  if (texture_desc_set_ != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &texture_desc_set_, 0,
                            nullptr);
  }

  // Bind vertex buffer
  VkBuffer vertex_buffers[] = {vertexBuffer()->buffer()};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cmd, 0, 1, vertex_buffers, offsets);

  // Bind index buffer
  vkCmdBindIndexBuffer(cmd, index_buffer_->buffer(), 0, VK_INDEX_TYPE_UINT32);

  // Draw indexed triangles
  vkCmdDrawIndexed(cmd, static_cast<uint32_t>(num_indices_), 1, 0, 0, 0);
}

bool MeshVisualizer::createTextureDescriptorSet() {
  if (!ctx_) {
    return false;
  }

  VkDevice device = ctx_->device();

  // Create descriptor set layout with one combined image sampler at binding 0
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  binding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;

  VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                                &texture_desc_layout_);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << "MeshVisualizer: Failed to create descriptor set layout";
    return false;
  }

  // Create descriptor pool
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_size.descriptorCount = 1;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  pool_info.maxSets = 1;

  result =
      vkCreateDescriptorPool(device, &pool_info, nullptr, &texture_desc_pool_);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << "MeshVisualizer: Failed to create descriptor pool";
    return false;
  }

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = texture_desc_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &texture_desc_layout_;

  result = vkAllocateDescriptorSets(device, &alloc_info, &texture_desc_set_);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << "MeshVisualizer: Failed to allocate descriptor set";
    return false;
  }

  return true;
}

void MeshVisualizer::updateTextureDescriptorSet() {
  if (!ctx_ || texture_desc_set_ == VK_NULL_HANDLE || !mesh_texture_) {
    return;
  }

  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_info.imageView = mesh_texture_->imageView();
  image_info.sampler = mesh_texture_->sampler();

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = texture_desc_set_;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.descriptorCount = 1;
  write.pImageInfo = &image_info;

  vkUpdateDescriptorSets(ctx_->device(), 1, &write, 0, nullptr);
}

bool MeshVisualizer::updateTexture(const void* src, uint32_t width,
                                   uint32_t height, const CudaStream& stream) {
  if (!ctx_) {
    LOG(ERROR) << "MeshVisualizer: context not set";
    return false;
  }

  if (!src || width == 0 || height == 0) {
    LOG(WARNING) << "MeshVisualizer::updateTexture: invalid parameters";
    return false;
  }

  // Create or resize texture as needed
  if (!mesh_texture_) {
    mesh_texture_ = std::make_unique<SharedTexture>();
    // kRGB8 is stored internally as RGBA — SharedTexture handles
    // the RGB→RGBA conversion. nvblox Color type is RGBA.
    if (!mesh_texture_->create(ctx_, width, height,
                               SharedTexture::Format::kRGB8)) {
      LOG(ERROR) << "MeshVisualizer: Failed to create mesh texture";
      mesh_texture_.reset();
      return false;
    }
    updateTextureDescriptorSet();
  } else if (mesh_texture_->width() != width ||
             mesh_texture_->height() != height) {
    if (ctx_) {
      ctx_->waitIdle();
    }
    if (!mesh_texture_->resize(width, height)) {
      LOG(ERROR) << "MeshVisualizer: Failed to resize mesh texture";
      return false;
    }
    updateTextureDescriptorSet();
  }

  // Copy texture data from CUDA
  if (!mesh_texture_->copyFromCuda(src, stream)) {
    LOG(WARNING) << "MeshVisualizer: Failed to copy texture data";
    return false;
  }

  has_texture_ = true;
  return true;
}

}  // namespace renderer
}  // namespace nvblox
