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
#include "nvblox/renderer/render_targets/vk_headless_target.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {

VkHeadlessTarget::~VkHeadlessTarget() { destroy(); }

bool VkHeadlessTarget::create(VkDevice device, VkPhysicalDevice physical_device,
                              uint32_t width, uint32_t height,
                              uint32_t image_count) {
  initBase(device, physical_device);
  image_count_ = image_count;

  if (!createImages(width, height)) {
    return false;
  }

  if (!createRenderPass()) {
    destroyImages();
    return false;
  }

  if (!createDepthResources()) {
    destroyRenderPass();
    destroyImages();
    return false;
  }

  if (!createFramebuffers()) {
    destroyDepthResources();
    destroyRenderPass();
    destroyImages();
    return false;
  }

  LOG(INFO) << "Created headless render target: " << width << "x" << height
            << " with " << image_count_ << " images";
  return true;
}

bool VkHeadlessTarget::createImages(uint32_t width, uint32_t height) {
  extent_.width = width;
  extent_.height = height;

  // Pre-allocate and zero-initialize to VK_NULL_HANDLE for safe cleanup on
  // error
  images_.resize(image_count_, VK_NULL_HANDLE);
  image_memories_.resize(image_count_, VK_NULL_HANDLE);
  image_views_.resize(image_count_, VK_NULL_HANDLE);

  // Configure image creation (same for all images)
  Image2DCreateInfo create_info;
  create_info.width = width;
  create_info.height = height;
  create_info.format = color_format_;
  create_info.usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  create_info.memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  for (uint32_t i = 0; i < image_count_; ++i) {
    Image2DResult result;
    if (!createImage2D(device_, physical_device_, create_info,
                       VK_IMAGE_ASPECT_COLOR_BIT, &result)) {
      LOG(ERROR) << "Failed to create color image " << i;
      // Clean up previously created images
      for (uint32_t j = 0; j < i; ++j) {
        Image2DResult prev = {images_[j], image_memories_[j], image_views_[j]};
        destroyImage2D(device_, &prev);
      }
      images_.clear();
      image_memories_.clear();
      image_views_.clear();
      return false;
    }

    images_[i] = result.image;
    image_memories_[i] = result.memory;
    image_views_[i] = result.view;
  }

  return true;
}

void VkHeadlessTarget::destroyImages() {
  // Destroy color image views
  for (auto view : image_views_) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device_, view, nullptr);
    }
  }
  image_views_.clear();

  // Destroy color images and memory
  for (size_t i = 0; i < images_.size(); ++i) {
    if (images_[i] != VK_NULL_HANDLE) {
      vkDestroyImage(device_, images_[i], nullptr);
    }
    if (i < image_memories_.size() && image_memories_[i] != VK_NULL_HANDLE) {
      vkFreeMemory(device_, image_memories_[i], nullptr);
    }
  }
  images_.clear();
  image_memories_.clear();
}

bool VkHeadlessTarget::resize(uint32_t width, uint32_t height) {
  // Validate dimensions
  if (width == 0 || height == 0) {
    LOG(WARNING) << "VkHeadlessTarget::resize: invalid dimensions " << width
                 << "x" << height;
    return false;
  }

  if (width == extent_.width && height == extent_.height) {
    return true;
  }

  if (!checkVkResult(vkDeviceWaitIdle(device_),
                     "vkDeviceWaitIdle (before resize)")) {
    return false;
  }

  // Destroy in reverse order of creation
  destroyFramebuffers();
  destroyDepthResources();
  destroyRenderPass();
  destroyImages();

  // Recreate
  if (!createImages(width, height)) {
    return false;
  }

  if (!createRenderPass()) {
    destroyImages();
    return false;
  }

  if (!createDepthResources()) {
    destroyRenderPass();
    destroyImages();
    return false;
  }

  if (!createFramebuffers()) {
    destroyDepthResources();
    destroyRenderPass();
    destroyImages();
    return false;
  }

  LOG(INFO) << "Resized headless render target to " << width << "x" << height;
  return true;
}

void VkHeadlessTarget::destroy() {
  if (device_ == VK_NULL_HANDLE) {
    return;
  }

  checkVkResult(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (destroy)");

  destroyFramebuffers();
  destroyDepthResources();
  destroyRenderPass();
  destroyImages();
  device_ = VK_NULL_HANDLE;
}

bool VkHeadlessTarget::acquireImage(VkSemaphore /*semaphore*/,
                                    uint32_t* image_index) {
  if (!image_index) {
    LOG(ERROR) << "image_index pointer is null";
    return false;
  }
  if (image_count_ == 0) {
    LOG(ERROR) << "Headless target has no images";
    return false;
  }
  // For headless, we simply cycle through images in round-robin fashion.
  // No semaphore signaling needed since we don't have a presentation engine.
  *image_index = current_image_;
  current_image_ = (current_image_ + 1) % image_count_;
  return true;
}

bool VkHeadlessTarget::presentImage(VkSemaphore /*wait_semaphore*/,
                                    uint32_t /*image_index*/) {
  // No presentation for headless rendering.
  // This could be extended to copy the image to host memory if needed.
  return true;
}

}  // namespace renderer
}  // namespace nvblox
