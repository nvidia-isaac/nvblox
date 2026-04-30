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

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <glog/logging.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_texture.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/utils/shader_utils.h"
#include "nvblox/renderer/visualizers/base_visualizer.h"
#include "nvblox/renderer/visualizers/pipeline_builder.h"

namespace nvblox {
namespace renderer {

/// Descriptor binding configuration for a texture.
struct TextureBinding {
  uint32_t binding;          ///< Descriptor set binding index.
  VkShaderStageFlags stage;  ///< Shader stages that access the texture.
};

/// Template base class for visualizers that use textures with descriptor sets.
///
/// This template provides common functionality for texture-based visualizers:
/// - Texture management with automatic resizing
/// - Descriptor set creation and updates
/// - Pipeline creation helpers with descriptor sets
/// - Common destroy pattern
///
/// Derived classes must implement:
/// - `textureFormat(size_t index)` - Format for each texture
/// - `textureBinding(size_t index)` - Descriptor binding info for each texture
/// - `defaultTextureDimensions(size_t index)` - Initial texture dimensions
/// - `shaderName()` - Returns the shader base name (e.g., "image_quad")
/// - `configurePipeline(PipelineBuilder&)` - Configure pipeline settings
/// - `pushConstantSize()` - Size of push constants in bytes
/// - `pushConstantStages()` - Shader stages that use push constants
/// - `visualizerName()` - Name for logging (e.g., "ImageVisualizer")
///
/// @tparam Derived The derived class type.
/// @tparam TextureCount Number of textures managed by this visualizer.
template <typename Derived, size_t TextureCount = 1>
class TexturedVisualizer : public BaseVisualizer {
 public:
  static constexpr size_t kTextureCount = TextureCount;

  ~TexturedVisualizer() override = default;

  // Non-copyable, non-movable
  TexturedVisualizer(const TexturedVisualizer&) = delete;
  TexturedVisualizer& operator=(const TexturedVisualizer&) = delete;
  TexturedVisualizer(TexturedVisualizer&&) = delete;
  TexturedVisualizer& operator=(TexturedVisualizer&&) = delete;

 protected:
  TexturedVisualizer() = default;

  /// Initialize textures and descriptor sets.
  /// Derived classes should call this in their init() implementation,
  /// then initialize any additional resources.
  /// @param ctx Vulkan context.
  /// @return True if initialization succeeded.
  bool initTextured(VkContext* ctx) {
    // Compile-time validation of interface requirements
    static_assert(std::is_convertible_v<decltype(Derived::textureFormat(0)),
                                        SharedTexture::Format>,
                  "Derived class must implement: static SharedTexture::Format "
                  "textureFormat(size_t index)");
    static_assert(std::is_convertible_v<decltype(Derived::textureBinding(0)),
                                        TextureBinding>,
                  "Derived class must implement: static TextureBinding "
                  "textureBinding(size_t index)");
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
        std::is_convertible_v<decltype(Derived::visualizerName()), const char*>,
        "Derived class must implement: static constexpr const char* "
        "visualizerName()");

    if (!initBase(ctx)) {
      return false;
    }

    // Create textures with derived class-specified formats and dimensions
    for (size_t i = 0; i < kTextureCount; ++i) {
      textures_[i] = std::make_unique<SharedTexture>();
      auto dims = derived().defaultTextureDimensions(i);

      // Validate dimensions
      if (dims.first < kMinTextureDimension ||
          dims.second < kMinTextureDimension ||
          dims.first > kMaxTextureDimension ||
          dims.second > kMaxTextureDimension) {
        LOG(ERROR) << derived().visualizerName()
                   << ": Invalid default dimensions for texture " << i << ": "
                   << dims.first << "x" << dims.second;
        return false;
      }

      if (!textures_[i]->create(ctx, dims.first, dims.second,
                                derived().textureFormat(i))) {
        LOG(ERROR) << derived().visualizerName()
                   << ": Failed to create texture " << i;
        return false;
      }
    }

    if (!createDescriptorSets()) {
      LOG(ERROR) << derived().visualizerName()
                 << ": Failed to create descriptor sets";
      return false;
    }

    return true;
  }

  /// Create a graphics pipeline using the derived class configuration.
  /// @param pipeline Output pipeline handle.
  /// @return True if creation succeeded.
  bool createPipelineTextured(VkPipeline* pipeline) {
    if (!ctx_) {
      LOG(ERROR) << derived().visualizerName() << ": context not set";
      return false;
    }

    VkDevice device = ctx_->device();

    // Load shaders
    ShaderPair shaders = loadShaders(derived().shaderName());
    if (!shaders.isValid()) {
      LOG(ERROR) << derived().visualizerName() << ": Failed to load shaders";
      return false;
    }

    // Create pipeline layout with descriptor sets if not already created
    if (pipeline_layout_ == VK_NULL_HANDLE) {
      if (!createPipelineLayoutWithDescriptors(
              &descriptor_set_layout_, 1, derived().pushConstantSize(),
              derived().pushConstantStages())) {
        shaders.destroy(device);
        return false;
      }
    }

    // Build pipeline using PipelineBuilder with derived class configuration
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

  /// Destroy textures, descriptor sets, and base resources.
  /// Derived classes should call ctx_->waitIdle() and destroy their specific
  /// resources (like pipelines) before calling this.
  void destroyTextured() {
    if (ctx_ && ctx_->device()) {
      VkDevice device = ctx_->device();

      if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
      }
      descriptor_set_ = VK_NULL_HANDLE;  // Freed with pool

      if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
      }
    }

    for (auto& texture : textures_) {
      if (texture) {
        texture->destroy();
        texture.reset();
      }
    }

    destroyBase();
  }

  /// Get texture by index.
  /// @param index Texture index (must be < TextureCount).
  /// @return Pointer to texture, or nullptr if not initialized.
  SharedTexture* texture(size_t index) {
    return index < kTextureCount ? textures_[index].get() : nullptr;
  }
  const SharedTexture* texture(size_t index) const {
    return index < kTextureCount ? textures_[index].get() : nullptr;
  }

  /// Resize texture with automatic descriptor set update.
  /// @param index Texture index.
  /// @param width New width.
  /// @param height New height.
  /// @return True if resize succeeded or dimensions unchanged.
  bool resizeTexture(size_t index, uint32_t width, uint32_t height) {
    if (index >= kTextureCount) {
      LOG(ERROR) << derived().visualizerName() << ": Invalid texture index "
                 << index;
      return false;
    }

    SharedTexture* tex = textures_[index].get();
    if (!tex) {
      LOG(ERROR) << derived().visualizerName() << ": Texture " << index
                 << " not initialized";
      return false;
    }

    // Validate dimensions
    if (width < kMinTextureDimension || height < kMinTextureDimension ||
        width > kMaxTextureDimension || height > kMaxTextureDimension) {
      LOG(ERROR) << derived().visualizerName()
                 << ": Invalid texture dimensions: " << width << "x" << height
                 << " (must be " << kMinTextureDimension << "-"
                 << kMaxTextureDimension << ")";
      return false;
    }

    // Check if resize is needed
    if (tex->width() == width && tex->height() == height) {
      return true;
    }

    // Wait for GPU to finish using current texture
    if (ctx_) {
      ctx_->waitIdle();
    }

    // Resize texture (this destroys and recreates the image view)
    if (!tex->resize(width, height)) {
      LOG(ERROR) << derived().visualizerName() << ": Failed to resize texture "
                 << index;
      return false;
    }

    // Update descriptor sets to point to new image views
    updateDescriptorSets();

    LOG(INFO) << derived().visualizerName() << ": Resized texture " << index
              << " to " << width << "x" << height;
    return true;
  }

  /// Copy data from CUDA to texture.
  /// @param index Texture index.
  /// @param src Source CUDA device pointer (contiguous, no row padding).
  /// @param stream CUDA stream.
  /// @return True if copy succeeded.
  bool copyToTexture(size_t index, const void* src, const CudaStream& stream) {
    if (index >= kTextureCount) {
      LOG(WARNING) << derived().visualizerName() << ": Invalid texture index "
                   << index;
      return false;
    }

    if (!src) {
      LOG(WARNING) << derived().visualizerName()
                   << ": Null pointer provided for texture " << index;
      return false;
    }

    SharedTexture* tex = textures_[index].get();
    if (!tex) {
      LOG(WARNING) << derived().visualizerName() << ": Texture " << index
                   << " not initialized";
      return false;
    }

    if (!tex->copyFromCuda(src, stream)) {
      LOG(WARNING) << derived().visualizerName()
                   << ": Failed to copy data to texture " << index;
      return false;
    }

    return true;
  }

  /// Get the descriptor set for binding in render().
  VkDescriptorSet descriptorSet() const { return descriptor_set_; }

  /// Get the descriptor set layout.
  VkDescriptorSetLayout descriptorSetLayout() const {
    return descriptor_set_layout_;
  }

 private:
  /// Get derived class reference
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  /// Create descriptor set layout, pool, and set.
  bool createDescriptorSets() {
    if (!ctx_) {
      return false;
    }

    VkDevice device = ctx_->device();

    // Create descriptor set layout bindings from derived class configuration
    std::array<VkDescriptorSetLayoutBinding, kTextureCount> bindings{};
    for (size_t i = 0; i < kTextureCount; ++i) {
      TextureBinding tb = derived().textureBinding(i);
      bindings[i].binding = tb.binding;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = tb.stage;
      bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(kTextureCount);
    layout_info.pBindings = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                                  &descriptor_set_layout_);
    if (result != VK_SUCCESS) {
      LOG(ERROR) << derived().visualizerName()
                 << ": Failed to create descriptor set layout";
      return false;
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = static_cast<uint32_t>(kTextureCount);

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    result =
        vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
      LOG(ERROR) << derived().visualizerName()
                 << ": Failed to create descriptor pool";
      return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    result = vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_);
    if (result != VK_SUCCESS) {
      LOG(ERROR) << derived().visualizerName()
                 << ": Failed to allocate descriptor set";
      return false;
    }

    // Update descriptor set with current textures
    updateDescriptorSets();

    return true;
  }

  /// Update descriptor sets to point to current texture image views.
  void updateDescriptorSets() {
    if (!ctx_ || descriptor_set_ == VK_NULL_HANDLE) {
      return;
    }

    VkDevice device = ctx_->device();

    std::array<VkDescriptorImageInfo, kTextureCount> image_infos{};
    std::array<VkWriteDescriptorSet, kTextureCount> writes{};

    for (size_t i = 0; i < kTextureCount; ++i) {
      if (!textures_[i]) {
        continue;
      }

      image_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      image_infos[i].imageView = textures_[i]->imageView();
      image_infos[i].sampler = textures_[i]->sampler();

      TextureBinding tb = derived().textureBinding(i);

      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = tb.binding;
      writes[i].dstArrayElement = 0;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].descriptorCount = 1;
      writes[i].pImageInfo = &image_infos[i];
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(kTextureCount),
                           writes.data(), 0, nullptr);
  }

  std::array<std::unique_ptr<SharedTexture>, kTextureCount> textures_;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};

}  // namespace renderer
}  // namespace nvblox
