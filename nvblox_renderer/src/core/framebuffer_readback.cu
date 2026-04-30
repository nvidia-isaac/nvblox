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
#include "nvblox/renderer/core/framebuffer_readback.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "nvblox/renderer/core/vk_context.h"

namespace nvblox {
namespace renderer {

FramebufferReadback::~FramebufferReadback() { destroy(); }

bool FramebufferReadback::init(VkContext* ctx, uint32_t width,
                               uint32_t height) {
  if (!ctx || width == 0 || height == 0) {
    LOG(ERROR) << "FramebufferReadback::init: invalid parameters";
    return false;
  }

  ctx_ = ctx;
  width_ = width;
  height_ = height;

  if (!color_texture_.create(ctx_, width_, height_,
                             SharedTexture::Format::kRGBA8)) {
    LOG(ERROR) << "FramebufferReadback: failed to create color SharedTexture";
    destroy();
    return false;
  }

  if (!depth_texture_.create(ctx_, width_, height_,
                             SharedTexture::Format::kR32F)) {
    LOG(ERROR) << "FramebufferReadback: failed to create depth SharedTexture";
    destroy();
    return false;
  }

  return true;
}

void FramebufferReadback::destroy() {
  depth_texture_.destroy();
  color_texture_.destroy();
  ctx_ = nullptr;
  width_ = 0;
  height_ = 0;
}

void FramebufferReadback::recordCopyCommands(VkCommandBuffer cmd,
                                             VkImage src_color,
                                             VkImage src_depth, uint32_t width,
                                             uint32_t height) {
  CHECK(isInitialized()) << "recordCopyCommands called before init()";
  CHECK(width <= width_ && height <= height_)
      << "Copy dimensions (" << width << "x" << height
      << ") exceed SharedTexture dimensions (" << width_ << "x" << height_
      << ")";

  // --- Barrier: ensure color attachment writes are visible to transfer ---
  // The render pass transitions the color image to TRANSFER_SRC_OPTIMAL, but
  // the implicit external subpass dependency does not guarantee visibility to
  // transfer reads. Insert an explicit memory barrier (no layout transition).
  VkImageMemoryBarrier color_barrier{};
  color_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  color_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  color_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  color_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  color_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  color_barrier.image = src_color;
  color_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  color_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  color_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &color_barrier);

  // --- Copy color image to SharedTexture ---
  VkImageCopy color_region{};
  color_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  color_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  color_region.extent = {width, height, 1};

  vkCmdCopyImage(cmd, src_color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 color_texture_.image(), VK_IMAGE_LAYOUT_GENERAL, 1,
                 &color_region);

  if (src_depth != VK_NULL_HANDLE) {
    // --- Transition depth from DEPTH_STENCIL_ATTACHMENT to TRANSFER_SRC ---
    VkImageMemoryBarrier depth_barrier{};
    depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    depth_barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_barrier.image = src_depth;
    depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_barrier.subresourceRange.baseMipLevel = 0;
    depth_barrier.subresourceRange.levelCount = 1;
    depth_barrier.subresourceRange.baseArrayLayer = 0;
    depth_barrier.subresourceRange.layerCount = 1;
    depth_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depth_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &depth_barrier);

    VkImageCopy depth_region{};
    depth_region.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    depth_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    depth_region.extent = {width, height, 1};

    vkCmdCopyImage(cmd, src_depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   depth_texture_.image(), VK_IMAGE_LAYOUT_GENERAL, 1,
                   &depth_region);
  }
}

bool FramebufferReadback::readToCuda(void* dst_color_rgba8,
                                     void* dst_depth_d32f,
                                     const CudaStream& stream) {
  CHECK(isInitialized()) << "readToCuda called before init()";
  if (!dst_color_rgba8) {
    LOG(ERROR) << "FramebufferReadback::readToCuda: null color dst pointer";
    return false;
  }

  cudaStream_t cuda_stream = stream;

  {
    size_t row_bytes = width_ * sizeof(uint32_t);
    cudaError_t err = cudaMemcpy2DFromArrayAsync(
        dst_color_rgba8, row_bytes, color_texture_.cudaArray(), 0, 0, row_bytes,
        height_, cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "FramebufferReadback: color readback failed: "
                 << cudaGetErrorString(err);
      return false;
    }
  }

  if (dst_depth_d32f) {
    size_t row_bytes = width_ * sizeof(float);
    cudaError_t err = cudaMemcpy2DFromArrayAsync(
        dst_depth_d32f, row_bytes, depth_texture_.cudaArray(), 0, 0, row_bytes,
        height_, cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      LOG(ERROR) << "FramebufferReadback: depth readback failed: "
                 << cudaGetErrorString(err);
      return false;
    }
  }

  return true;
}

}  // namespace renderer
}  // namespace nvblox
