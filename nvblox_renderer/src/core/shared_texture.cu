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

#include "nvblox/renderer/core/shared_texture.h"

#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_utils.h"
#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {

struct RgbToRgba {
  __host__ __device__ uchar4 operator()(const uchar3& rgb) const {
    return make_uchar4(rgb.x, rgb.y, rgb.z, kOpaqueAlpha);
  }
};

SharedTexture::~SharedTexture() { destroy(); }

SharedTexture::SharedTexture(SharedTexture&& other) noexcept {
  *this = std::move(other);
}

SharedTexture& SharedTexture::operator=(SharedTexture&& other) noexcept {
  if (this != &other) {
    destroy();

    // Move base class members
    moveBaseFrom(other);

    // Move texture-specific members
    width_ = other.width_;
    height_ = other.height_;
    format_ = other.format_;
    image_ = other.image_;
    image_view_ = other.image_view_;
    sampler_ = other.sampler_;
    cuda_array_ = other.cuda_array_;
    cuda_mipmap_ = other.cuda_mipmap_;
    rgba_staging_buffer_ = other.rgba_staging_buffer_;
    rgba_staging_size_ = other.rgba_staging_size_;

    // Reset other's texture-specific members
    other.width_ = 0;
    other.height_ = 0;
    other.image_ = VK_NULL_HANDLE;
    other.image_view_ = VK_NULL_HANDLE;
    other.sampler_ = VK_NULL_HANDLE;
    other.cuda_array_ = nullptr;
    other.cuda_mipmap_ = nullptr;
    other.rgba_staging_buffer_ = nullptr;
    other.rgba_staging_size_ = 0;
  }
  return *this;
}

void SharedTexture::destroy() {
  // Wait for GPU to finish using the texture before destroying resources.
  // This prevents use-after-free if GPU operations are still in flight.
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Destroy texture-specific CUDA resources first with error checking
  // (warning-level in destructor path)
  if (rgba_staging_buffer_) {
    cudaError_t err = cudaFree(rgba_staging_buffer_);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to free RGBA staging buffer: "
                   << cudaGetErrorString(err);
    }
    rgba_staging_buffer_ = nullptr;
    rgba_staging_size_ = 0;
  }

  if (cuda_mipmap_) {
    cudaError_t err = cudaFreeMipmappedArray(cuda_mipmap_);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to free CUDA mipmapped array: "
                   << cudaGetErrorString(err);
    }
    cuda_mipmap_ = nullptr;
    cuda_array_ = nullptr;  // cuda_array_ is part of cuda_mipmap_
  }

  // Destroy texture-specific Vulkan resources before base cleanup frees memory
  if (ctx_ && ctx_->device()) {
    if (sampler_) {
      vkDestroySampler(ctx_->device(), sampler_, nullptr);
      sampler_ = VK_NULL_HANDLE;
    }
    if (image_view_) {
      vkDestroyImageView(ctx_->device(), image_view_, nullptr);
      image_view_ = VK_NULL_HANDLE;
    }
    if (image_) {
      vkDestroyImage(ctx_->device(), image_, nullptr);
      image_ = VK_NULL_HANDLE;
    }
  }

  // Destroy base class resources (sync, CUDA external memory, Vulkan memory)
  destroyBase();

  width_ = 0;
  height_ = 0;
}

VkFormat SharedTexture::toVkFormat(Format format) const {
  switch (format) {
    case Format::kR32F:
      return VK_FORMAT_R32_SFLOAT;
    case Format::kRGBA8:
    case Format::kRGB8:  // RGB stored as RGBA internally
      // Use SRGB format for color data - camera output is typically sRGB
      return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::kR8:
      return VK_FORMAT_R8_UNORM;
    default:
      return VK_FORMAT_R8G8B8A8_SRGB;
  }
}

size_t SharedTexture::bytesPerPixel(Format format) const {
  switch (format) {
    case Format::kR32F:
      return 4;
    case Format::kRGBA8:
      return 4;
    case Format::kRGB8:
      return 3;  // Source is 3 bytes, but stored as 4
    case Format::kR8:
      return 1;
    default:
      return 4;
  }
}

bool SharedTexture::create(VkContext* ctx, uint32_t width, uint32_t height,
                           Format format) {
  // Validate context using base class helper
  if (!validateContext(ctx)) {
    return false;
  }

  // Validate texture dimensions
  if (width < kMinTextureDimension || height < kMinTextureDimension) {
    LOG(ERROR) << "SharedTexture dimensions must be at least "
               << kMinTextureDimension << "x" << kMinTextureDimension
               << ", got " << width << "x" << height;
    return false;
  }
  if (width > kMaxTextureDimension || height > kMaxTextureDimension) {
    LOG(ERROR) << "SharedTexture dimensions must not exceed "
               << kMaxTextureDimension << "x" << kMaxTextureDimension
               << ", got " << width << "x" << height;
    return false;
  }

  ctx_ = ctx;
  width_ = width;
  height_ = height;
  format_ = format;

  VkDevice device = ctx_->device();
  VkFormat vk_format = toVkFormat(format);

  // Create image with external memory support
  VkExternalMemoryImageCreateInfo ext_mem_info{};
  ext_mem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  ext_mem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = &ext_mem_info;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = vk_format;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  checkVkErrors(vkCreateImage(device, &image_info, nullptr, &image_));

  // Get memory requirements
  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(device, image_, &mem_requirements);

  // Find suitable memory type using base class helper
  uint32_t memory_type_index = findMemoryType(
      mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memory_type_index == kMemoryTypeNotFound) {
    LOG(ERROR) << "Failed to find suitable memory type for texture";
    destroy();
    return false;
  }

  // Allocate exportable memory using base class helper
  if (!allocateExportableMemory(mem_requirements.size, memory_type_index)) {
    destroy();
    return false;
  }

  // Bind memory to image
  checkVkErrors(vkBindImageMemory(device, image_, memory_, 0));

  // Import to CUDA using base class helper
  if (!importToCuda(mem_requirements.size)) {
    destroy();
    return false;
  }

  // Map to CUDA mipmapped array (texture-specific)
  cudaExternalMemoryMipmappedArrayDesc mipmap_desc{};
  mipmap_desc.offset = 0;
  mipmap_desc.extent.width = width;
  mipmap_desc.extent.height = height;
  mipmap_desc.extent.depth = 0;
  mipmap_desc.numLevels = 1;

  switch (format) {
    case Format::kR32F:
      mipmap_desc.formatDesc = cudaCreateChannelDesc<float>();
      break;
    case Format::kRGBA8:
    case Format::kRGB8:  // RGB stored as RGBA
      mipmap_desc.formatDesc = cudaCreateChannelDesc<uchar4>();
      break;
    case Format::kR8:
      mipmap_desc.formatDesc = cudaCreateChannelDesc<unsigned char>();
      break;
  }

  cudaError_t cuda_err = cudaExternalMemoryGetMappedMipmappedArray(
      &cuda_mipmap_, cuda_external_memory_, &mipmap_desc);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "Failed to map external memory to CUDA array: "
               << cudaGetErrorString(cuda_err);
    destroy();
    return false;
  }

  // Get level 0 of the mipmap
  cuda_err = cudaGetMipmappedArrayLevel(&cuda_array_, cuda_mipmap_, 0);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "Failed to get mipmap level 0: "
               << cudaGetErrorString(cuda_err);
    destroy();
    return false;
  }

  // Allocate staging buffer for RGB->RGBA conversion if needed
  if (format == Format::kRGB8) {
    rgba_staging_size_ = width * height * 4;
    cuda_err = cudaMalloc(&rgba_staging_buffer_, rgba_staging_size_);
    if (cuda_err != cudaSuccess) {
      LOG(ERROR) << "Failed to allocate RGBA staging buffer: "
                 << cudaGetErrorString(cuda_err);
      destroy();
      return false;
    }
  }

  // Create image view using helper function
  checkVkErrors(createImageView2D(device, image_, vk_format,
                                  VK_IMAGE_ASPECT_COLOR_BIT, &image_view_));

  // Create sampler using helper function
  checkVkErrors(createDefaultSampler2D(device, &sampler_));

  // Transition image layout to GENERAL to allow CUDA writes.
  // VK_IMAGE_LAYOUT_GENERAL is compatible with both shader reads and
  // external writes from CUDA. Using SHADER_READ_ONLY_OPTIMAL would be
  // undefined behavior when CUDA writes to the image.
  VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &barrier);

  ctx_->endSingleTimeCommands(cmd);

  LOG(INFO) << "Created shared texture: " << width << "x" << height;
  return true;
}

bool SharedTexture::resize(uint32_t new_width, uint32_t new_height) {
  if (new_width == width_ && new_height == height_) {
    return true;  // No change needed
  }
  if (ctx_ == nullptr) {
    LOG(ERROR) << "Cannot resize uninitialized texture";
    return false;
  }

  // Create-before-destroy pattern: Create new texture first, then swap.
  // This ensures we keep the old texture if creation fails.
  SharedTexture new_texture;
  if (!new_texture.create(ctx_, new_width, new_height, format_)) {
    LOG(ERROR) << "Failed to create new texture for resize";
    return false;  // Keep old texture intact
  }

  // Success: swap with new texture. Old resources destroyed when new_texture
  // goes out of scope.
  *this = std::move(new_texture);
  return true;
}

bool SharedTexture::copyFromCuda(const void* src, const CudaStream& stream) {
  if (!cuda_array_) {
    LOG(ERROR) << "SharedTexture not initialized";
    return false;
  }

  const size_t num_pixels = static_cast<size_t>(width_) * height_;
  const size_t row_bytes = width_ * bytesPerPixel(format_);
  cudaStream_t cuda_stream = stream;

  if (format_ == Format::kRGB8) {
    auto src_ptr = thrust::device_pointer_cast(static_cast<const uchar3*>(src));
    auto dst_ptr =
        thrust::device_pointer_cast(static_cast<uchar4*>(rgba_staging_buffer_));
    thrust::transform(thrust::cuda::par.on(cuda_stream), src_ptr,
                      src_ptr + num_pixels, dst_ptr, RgbToRgba{});

    cudaError_t err = cudaMemcpy2DToArrayAsync(
        cuda_array_, 0, 0, rgba_staging_buffer_, width_ * 4, width_ * 4,
        height_, cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMemcpy2DToArrayAsync (RGBA staging) failed: "
                 << cudaGetErrorString(err);
      return false;
    }
  } else {
    cudaError_t err = cudaMemcpy2DToArrayAsync(
        cuda_array_, 0, 0, src, row_bytes, row_bytes, height_,
        cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMemcpy2DToArrayAsync failed: "
                 << cudaGetErrorString(err);
      return false;
    }
  }

  return true;
}

}  // namespace renderer
}  // namespace nvblox
