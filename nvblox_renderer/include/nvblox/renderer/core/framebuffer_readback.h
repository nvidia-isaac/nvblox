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

#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_texture.h"

namespace nvblox {
namespace renderer {

class VkContext;

/// Reads back Vulkan framebuffer color and depth attachments to CUDA memory.
///
/// Owns two SharedTextures (CUDA-exported via external memory) that serve as
/// copy destinations. The workflow is:
///   1. After the render pass ends, call recordCopyCommands() to record
///      vkCmdCopyImage from the framebuffer attachments into the internal
///      SharedTextures.
///   2. After the command buffer executes (endFrame), call readToCuda() to
///      copy from the SharedTextures' cudaArrays to user-provided linear
///      CUDA device buffers.
class FramebufferReadback {
 public:
  FramebufferReadback() = default;
  ~FramebufferReadback();

  FramebufferReadback(const FramebufferReadback&) = delete;
  FramebufferReadback& operator=(const FramebufferReadback&) = delete;
  FramebufferReadback(FramebufferReadback&&) = delete;
  FramebufferReadback& operator=(FramebufferReadback&&) = delete;

  /// Initialize with the framebuffer dimensions.
  /// @param ctx Vulkan context.
  /// @param width Framebuffer width in pixels.
  /// @param height Framebuffer height in pixels.
  /// @return True if initialization succeeded.
  bool init(VkContext* ctx, uint32_t width, uint32_t height);

  /// Destroy resources.
  void destroy();

  /// Record Vulkan commands to copy color and depth images into internal
  /// SharedTextures. Must be called after the render pass ends and before
  /// the command buffer is submitted.
  ///
  /// When src_depth is provided, inserts a pipeline barrier to transition the
  /// depth image from DEPTH_STENCIL_ATTACHMENT_OPTIMAL to
  /// TRANSFER_SRC_OPTIMAL before the copy. The color image is assumed to
  /// already be in TRANSFER_SRC_OPTIMAL (headless target final layout).
  ///
  /// @param cmd Active command buffer (recording state, outside render pass).
  /// @param src_color Source color image (VK_FORMAT_R8G8B8A8_SRGB).
  /// @param src_depth Source depth image (VK_FORMAT_D32_SFLOAT), or
  ///        VK_NULL_HANDLE to skip depth copy.
  /// @param width Image width in pixels.
  /// @param height Image height in pixels.
  void recordCopyCommands(VkCommandBuffer cmd, VkImage src_color,
                          VkImage src_depth, uint32_t width, uint32_t height);

  /// Copy from internal SharedTextures to user-provided linear CUDA buffers.
  /// Must be called after the command buffer that contains the copy commands
  /// has finished executing.
  ///
  /// @param dst_color_rgba8 Device pointer for RGBA8 output
  ///        (width * height * 4 bytes).
  /// @param dst_depth_d32f Device pointer for D32F output
  ///        (width * height * 4 bytes), or nullptr to skip depth readback.
  /// @param stream CUDA stream for async copies.
  /// @return True if copies succeeded.
  bool readToCuda(void* dst_color_rgba8, void* dst_depth_d32f,
                  const CudaStream& stream);

  bool isInitialized() const { return ctx_ != nullptr; }

 private:
  SharedTexture color_texture_;
  SharedTexture depth_texture_;
  VkContext* ctx_ = nullptr;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

}  // namespace renderer
}  // namespace nvblox
