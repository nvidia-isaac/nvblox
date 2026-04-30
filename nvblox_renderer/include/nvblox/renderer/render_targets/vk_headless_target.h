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

/// Headless render target for offscreen rendering.
/// Creates VkImages directly instead of using a swapchain.
/// Useful for headless rendering, testing, or CUDA interop without display.
class VkHeadlessTarget : public VkRenderTargetBase {
 public:
  /// Default number of images for double-buffering.
  static constexpr uint32_t kDefaultImageCount = 2;

  VkHeadlessTarget() = default;
  ~VkHeadlessTarget() override;

  // Non-copyable
  VkHeadlessTarget(const VkHeadlessTarget&) = delete;
  VkHeadlessTarget& operator=(const VkHeadlessTarget&) = delete;

  /// Create the headless render target.
  /// @param device Vulkan logical device.
  /// @param physical_device Vulkan physical device (for memory allocation).
  /// @param width Initial width in pixels.
  /// @param height Initial height in pixels.
  /// @param image_count Number of images for buffering (default 2).
  /// @return True if creation succeeded.
  bool create(VkDevice device, VkPhysicalDevice physical_device, uint32_t width,
              uint32_t height, uint32_t image_count = kDefaultImageCount);

  // IVkRenderTarget interface
  bool resize(uint32_t width, uint32_t height) override;
  void destroy() override;
  bool acquireImage(VkSemaphore semaphore, uint32_t* image_index) override;
  bool presentImage(VkSemaphore wait_semaphore, uint32_t image_index) override;

  uint32_t imageCount() const override {
    return static_cast<uint32_t>(images_.size());
  }
  bool requiresPresentation() const override { return false; }

  /// Get color image for a given index (useful for reading back rendered data).
  VkImage colorImage(uint32_t index) const {
    if (index >= images_.size()) return VK_NULL_HANDLE;
    return images_[index];
  }

 protected:
  // VkRenderTargetBase hooks
  VkImageLayout colorFinalLayout() const override {
    return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  }
  bool needsDepthReadback() const override { return true; }
  const std::vector<VkImageView>& imageViews() const override {
    return image_views_;
  }

 private:
  bool createImages(uint32_t width, uint32_t height);
  void destroyImages();

  // Color images (owned by us, not a swapchain)
  std::vector<VkImage> images_;
  std::vector<VkDeviceMemory> image_memories_;
  std::vector<VkImageView> image_views_;
  uint32_t image_count_ = kDefaultImageCount;

  // Current image index for round-robin acquisition
  uint32_t current_image_ = 0;
};

}  // namespace renderer
}  // namespace nvblox
