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

#include <vector>

#include "nvblox/renderer/core/vk_render_target.h"

namespace nvblox {
namespace renderer {

// TODO(Tao Jin): Add unit tests for render target classes (VkRenderTargetBase,
//               VkHeadlessTarget, VkWindowTarget).

/// Base class for render targets providing shared depth buffer, render pass,
/// and framebuffer management.
///
/// Derived classes must implement:
/// - colorFinalLayout(): The final image layout for color attachments
/// - imageViews(): Access to the color image views for framebuffer creation
/// - acquireImage(), presentImage(): Target-specific image
/// acquisition/presentation
/// - imageCount(), requiresPresentation(): Target-specific properties
/// - resize(), destroy(): Target-specific lifecycle (may call base helpers)
class VkRenderTargetBase : public IVkRenderTarget {
 public:
  VkRenderTargetBase() = default;
  ~VkRenderTargetBase() override = default;

  // Non-copyable
  VkRenderTargetBase(const VkRenderTargetBase&) = delete;
  VkRenderTargetBase& operator=(const VkRenderTargetBase&) = delete;

  // IVkRenderTarget interface - common implementations
  VkRenderPass renderPass() const override final { return render_pass_; }
  VkFramebuffer framebuffer(uint32_t index) const override final;
  VkExtent2D extent() const override final { return extent_; }
  VkFormat colorFormat() const override final { return color_format_; }

  /// Get the depth image handle (for readback after rendering).
  VkImage depthImage() const { return depth_image_; }

  /// Get the depth buffer format.
  VkFormat depthFormat() const { return depth_format_; }

 protected:
  /// Get the final image layout for color attachments in the render pass.
  /// @return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for window targets,
  ///         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for headless targets.
  virtual VkImageLayout colorFinalLayout() const = 0;

  /// Whether the depth buffer needs to be readable after the render pass
  /// (e.g. for readback to CUDA). When true, the depth image is created with
  /// TRANSFER_SRC usage and the render pass stores depth data.
  virtual bool needsDepthReadback() const { return false; }

  /// Get the color image views for framebuffer creation.
  /// @return Reference to the vector of color image views.
  virtual const std::vector<VkImageView>& imageViews() const = 0;

  /// Create depth buffer resources.
  /// @return True if creation succeeded.
  bool createDepthResources();

  /// Create the render pass with color and depth attachments.
  /// Uses colorFinalLayout() for the color attachment final layout.
  /// @return True if creation succeeded.
  bool createRenderPass();

  /// Create framebuffers for all images.
  /// Uses imageViews() to get the color attachments.
  /// @return True if creation succeeded.
  bool createFramebuffers();

  /// Destroy depth buffer resources.
  void destroyDepthResources();

  /// Destroy framebuffers.
  void destroyFramebuffers();

  /// Destroy render pass.
  void destroyRenderPass();

  /// Initialize base class with device handles.
  /// @param device Vulkan logical device.
  /// @param physical_device Vulkan physical device.
  void initBase(VkDevice device, VkPhysicalDevice physical_device);

  // Common members
  VkDevice device_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

  VkExtent2D extent_ = {0, 0};
  VkFormat color_format_ = VK_FORMAT_R8G8B8A8_SRGB;
  VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;

  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers_;

  // Depth buffer
  VkImage depth_image_ = VK_NULL_HANDLE;
  VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;
  VkImageView depth_view_ = VK_NULL_HANDLE;
};

}  // namespace renderer
}  // namespace nvblox
