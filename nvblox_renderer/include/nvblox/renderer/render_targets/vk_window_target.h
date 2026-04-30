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

#include "nvblox/renderer/render_targets/vk_render_target_base.h"

namespace nvblox {
namespace renderer {

/// Window-based render target using VkSwapchainKHR.
/// Creates and manages a swapchain for window presentation.
class VkWindowTarget : public VkRenderTargetBase {
 public:
  VkWindowTarget() = default;
  ~VkWindowTarget() override;

  // Non-copyable
  VkWindowTarget(const VkWindowTarget&) = delete;
  VkWindowTarget& operator=(const VkWindowTarget&) = delete;

  /// Create window render target with swapchain.
  /// @param device Vulkan logical device.
  /// @param physical_device Vulkan physical device.
  /// @param surface Window surface.
  /// @param graphics_queue_family Graphics queue family index.
  /// @param present_queue Queue for presentation.
  /// @param width Initial width.
  /// @param height Initial height.
  /// @return True if creation succeeded.
  bool create(VkDevice device, VkPhysicalDevice physical_device,
              VkSurfaceKHR surface, uint32_t graphics_queue_family,
              VkQueue present_queue, uint32_t width, uint32_t height);

  // IVkRenderTarget interface
  bool resize(uint32_t width, uint32_t height) override;
  void destroy() override;
  bool acquireImage(VkSemaphore semaphore, uint32_t* image_index) override;
  bool presentImage(VkSemaphore wait_semaphore, uint32_t image_index) override;

  uint32_t imageCount() const override {
    return static_cast<uint32_t>(images_.size());
  }
  bool requiresPresentation() const override { return true; }

  /// Get the swapchain handle (for sync operations).
  VkSwapchainKHR swapchain() const { return swapchain_; }

 protected:
  // VkRenderTargetBase hooks
  VkImageLayout colorFinalLayout() const override {
    return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  }
  const std::vector<VkImageView>& imageViews() const override {
    return image_views_;
  }

 private:
  bool createSwapchain(uint32_t width, uint32_t height);
  void destroySwapchainResources();

  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  uint32_t graphics_queue_family_ = 0;
  VkQueue present_queue_ = VK_NULL_HANDLE;

  // Swapchain
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
};

}  // namespace renderer
}  // namespace nvblox
