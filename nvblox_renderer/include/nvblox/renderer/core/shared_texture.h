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

#include <cstdint>

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_resource_base.h"

namespace nvblox {
namespace renderer {

class VkContext;

/// Shared texture between CUDA and Vulkan via external memory.
/// Allows zero-copy sharing of image data from nvblox (CUDA) to Vulkan.
///
/// Synchronization:
/// The caller must ensure CUDA writes are complete before Vulkan reads.
/// Currently uses cudaStreamSynchronize (CPU-blocking).
/// Future: CudaVulkanSemaphore will enable GPU-GPU sync without CPU stall.
///
/// Image Layout:
/// The texture is kept in VK_IMAGE_LAYOUT_GENERAL to allow CUDA writes.
/// This is compatible with shader reads in Vulkan.
class SharedTexture : public SharedResourceBase<SharedTexture> {
 public:
  /// Type aliases for type introspection (following nvblox pattern)
  using VkResourceType = VkImage;
  using CudaResourceType = cudaArray_t;

  /// Texture format for creation.
  enum class Format {
    kR32F,   // Single channel float (depth)
    kRGBA8,  // 4-channel 8-bit (color)
    kRGB8,   // 3-channel 8-bit (color) - stored as RGBA, converted on copy
    kR8,     // Single channel 8-bit (grayscale)
  };

  SharedTexture() = default;
  ~SharedTexture();

  // Non-copyable (inherited from base)
  SharedTexture(const SharedTexture&) = delete;
  SharedTexture& operator=(const SharedTexture&) = delete;

  // Movable
  SharedTexture(SharedTexture&& other) noexcept;
  SharedTexture& operator=(SharedTexture&& other) noexcept;

  /// Create a shared texture.
  /// @param ctx Vulkan context.
  /// @param width Texture width.
  /// @param height Texture height.
  /// @param format Texture format.
  /// @return True if creation succeeded.
  bool create(VkContext* ctx, uint32_t width, uint32_t height, Format format);

  /// Resize the texture.
  /// @param new_width New texture width.
  /// @param new_height New texture height.
  /// @return True if resize succeeded.
  bool resize(uint32_t new_width, uint32_t new_height);

  /// Destroy the texture and free resources.
  void destroy();

  /// Copy data from a CUDA device pointer to this texture.
  /// Source data must be contiguous (row-major, no padding).
  /// For RGB8 format, converts to RGBA internally.
  /// @param src Source CUDA device pointer.
  /// @param stream CUDA stream for the copy.
  /// @return True if copy succeeded, false on CUDA errors.
  bool copyFromCuda(const void* src, const CudaStream& stream);

  /// Get the CUDA array for direct access (e.g., for surface writes).
  /// @return CUDA array handle, or nullptr if texture is not valid.
  /// @note The array is valid for the lifetime of this SharedTexture object
  ///       and becomes invalid after destroy() is called or the object is
  ///       destroyed. Use isValid() to check if the array can be safely used.
  cudaArray_t cudaArray() const { return cuda_array_; }

  /// Get the Vulkan image.
  VkImage image() const { return image_; }

  /// Get the Vulkan image view.
  VkImageView imageView() const { return image_view_; }

  /// Get the Vulkan sampler.
  VkSampler sampler() const { return sampler_; }

  /// Get texture dimensions.
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  /// Check if texture is valid.
  bool isValid() const { return image_ != VK_NULL_HANDLE; }

 private:
  VkFormat toVkFormat(Format format) const;
  size_t bytesPerPixel(Format format) const;

  // Texture dimensions and format
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  Format format_ = Format::kRGBA8;

  // Texture-specific Vulkan resources
  VkImage image_ = VK_NULL_HANDLE;
  VkImageView image_view_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;

  // Texture-specific CUDA mapping
  cudaArray_t cuda_array_ = nullptr;
  cudaMipmappedArray_t cuda_mipmap_ = nullptr;

  // Staging buffer for RGB->RGBA conversion
  void* rgba_staging_buffer_ = nullptr;
  size_t rgba_staging_size_ = 0;
};

}  // namespace renderer
}  // namespace nvblox
