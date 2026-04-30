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

namespace nvblox {
namespace renderer {

/// Abstract interface for render targets.
/// Supports window swapchains, headless rendering, and OpenXR targets.
/// Each implementation handles its own image management and presentation.
class IVkRenderTarget {
 public:
  virtual ~IVkRenderTarget() = default;

  /// Resize the render target.
  /// @param width New width in pixels.
  /// @param height New height in pixels.
  /// @return True if resize succeeded.
  virtual bool resize(uint32_t width, uint32_t height) = 0;

  /// Destroy all resources.
  virtual void destroy() = 0;

  /// Acquire the next image to render to.
  /// @param semaphore Semaphore to signal when image is ready.
  /// @param image_index Output: index of acquired image.
  /// @return True if acquired, false if resize needed or error.
  virtual bool acquireImage(VkSemaphore semaphore, uint32_t* image_index) = 0;

  /// Present/submit the rendered image.
  /// @param wait_semaphore Semaphore to wait on before present.
  /// @param image_index Index of image to present.
  /// @return True if presented, false if resize needed or error.
  virtual bool presentImage(VkSemaphore wait_semaphore,
                            uint32_t image_index) = 0;

  /// Get the render pass for this target.
  virtual VkRenderPass renderPass() const = 0;

  /// Get framebuffer for the given image index.
  virtual VkFramebuffer framebuffer(uint32_t index) const = 0;

  /// Get render target extent (width/height).
  virtual VkExtent2D extent() const = 0;

  /// Get color image format.
  virtual VkFormat colorFormat() const = 0;

  /// Get number of images in the target.
  virtual uint32_t imageCount() const = 0;

  /// Returns true if this target requires presentation.
  /// Window and XR targets return true, headless returns false.
  virtual bool requiresPresentation() const = 0;
};

}  // namespace renderer
}  // namespace nvblox
