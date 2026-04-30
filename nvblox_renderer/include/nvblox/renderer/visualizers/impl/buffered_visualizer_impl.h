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

#include "nvblox/renderer/visualizers/buffered_visualizer.h"

namespace nvblox {
namespace renderer {

template <typename Derived, typename VertexType>
bool BufferedVisualizer<Derived, VertexType>::initBuffered(VkContext* ctx) {
  static_assert(
      std::is_convertible_v<decltype(Derived::shaderName()), const char*>,
      "Derived class must implement: static constexpr const char* "
      "shaderName()");
  static_assert(
      std::is_convertible_v<decltype(Derived::pushConstantSize()), uint32_t>,
      "Derived class must implement: static constexpr uint32_t "
      "pushConstantSize()");
  static_assert(std::is_convertible_v<decltype(Derived::pushConstantStages()),
                                      VkShaderStageFlags>,
                "Derived class must implement: static constexpr "
                "VkShaderStageFlags pushConstantStages()");
  static_assert(
      std::is_convertible_v<decltype(Derived::defaultBufferElementCount()),
                            size_t>,
      "Derived class must implement: static constexpr size_t "
      "defaultBufferElementCount()");
  static_assert(
      std::is_convertible_v<decltype(Derived::maxElementCount()), size_t>,
      "Derived class must implement: static constexpr size_t "
      "maxElementCount()");
  static_assert(
      std::is_convertible_v<decltype(Derived::visualizerName()), const char*>,
      "Derived class must implement: static constexpr const char* "
      "visualizerName()");

  if (!initBase(ctx)) {
    return false;
  }

  vertex_buffer_ = std::make_unique<SharedBuffer>();
  size_t initial_size = derived().defaultBufferElementCount() * sizeof(Vertex);
  if (!vertex_buffer_->create(ctx, initial_size,
                              SharedBuffer::Usage::kVertex)) {
    LOG(ERROR) << derived().visualizerName()
               << ": Failed to create vertex buffer";
    return false;
  }

  return true;
}

template <typename Derived, typename VertexType>
bool BufferedVisualizer<Derived, VertexType>::createPipelineBuffered(
    VkPipeline* pipeline) {
  if (!ctx_) {
    LOG(ERROR) << derived().visualizerName() << ": context not set";
    return false;
  }

  VkDevice device = ctx_->device();

  ShaderPair shaders = loadShaders(derived().shaderName());
  if (!shaders.isValid()) {
    LOG(ERROR) << derived().visualizerName() << ": Failed to load shaders";
    return false;
  }

  if (pipeline_layout_ == VK_NULL_HANDLE) {
    if (!createPipelineLayout(derived().pushConstantSize(),
                              derived().pushConstantStages())) {
      shaders.destroy(device);
      return false;
    }
  }

  PipelineBuilder builder;
  builder.setShaders(shaders.vert, shaders.frag);
  derived().configurePipeline(builder);

  *pipeline = builder.build(device, pipeline_layout_, ctx_->renderPass(), 0,
                            ctx_->pipelineCache());

  shaders.destroy(device);

  if (*pipeline == VK_NULL_HANDLE) {
    LOG(ERROR) << derived().visualizerName()
               << ": Failed to create graphics pipeline";
    return false;
  }

  return true;
}

template <typename Derived, typename VertexType>
void BufferedVisualizer<Derived, VertexType>::destroyBuffered() {
  if (vertex_buffer_) {
    vertex_buffer_->destroy();
    vertex_buffer_.reset();
  }
  destroyBase();
}

template <typename Derived, typename VertexType>
bool BufferedVisualizer<Derived, VertexType>::ensureBufferCapacity(
    size_t num_elements) {
  if (!vertex_buffer_) {
    LOG(ERROR) << derived().visualizerName() << ": buffer not initialized";
    return false;
  }

  auto validation = validateBufferSize(num_elements, sizeof(Vertex),
                                       derived().maxElementCount(),
                                       derived().visualizerName());
  if (!validation.valid) {
    return false;
  }

  if (validation.required_size > vertex_buffer_->size()) {
    size_t new_size = calculateResizeCapacity(validation.required_size);
    LOG(INFO) << derived().visualizerName() << ": Resizing buffer from "
              << vertex_buffer_->size() << " to " << new_size << " bytes";
    if (!vertex_buffer_->resize(new_size)) {
      LOG(ERROR) << derived().visualizerName() << ": Failed to resize buffer";
      return false;
    }
  }

  return true;
}

template <typename Derived, typename VertexType>
bool BufferedVisualizer<Derived, VertexType>::copyToVertexBuffer(
    const Vertex* src, size_t num_elements, const CudaStream& stream) {
  size_t copy_size = num_elements * sizeof(Vertex);
  if (!vertex_buffer_->copyFromCuda(src, copy_size, stream)) {
    LOG(WARNING) << derived().visualizerName()
                 << ": Failed to copy data to buffer";
    return false;
  }
  return true;
}

}  // namespace renderer
}  // namespace nvblox
