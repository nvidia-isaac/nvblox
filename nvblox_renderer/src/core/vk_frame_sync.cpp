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
#include "nvblox/renderer/core/vk_frame_sync.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"

namespace nvblox {
namespace renderer {

VkFrameSync::~VkFrameSync() { destroy(); }

bool VkFrameSync::create(VkDevice device) {
  if (device == VK_NULL_HANDLE) {
    LOG(ERROR) << "Invalid device for VkFrameSync";
    return false;
  }
  device_ = device;

  // Create fences per frame-in-flight (limits CPU ahead of GPU)
  in_flight_fences_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled

  for (int i = 0; i < kMaxFramesInFlight; i++) {
    // Fence creation is fatal - use hard check
    checkVkErrors(
        vkCreateFence(device_, &fence_info, nullptr, &in_flight_fences_[i]));
  }

  LOG(INFO) << "VkFrameSync created with " << kMaxFramesInFlight
            << " frames in flight";
  return true;
}

bool VkFrameSync::createRenderTargetSemaphores(uint32_t image_count) {
  if (device_ == VK_NULL_HANDLE) {
    LOG(ERROR) << "VkFrameSync not initialized";
    return false;
  }

  // Destroy old semaphores if any
  for (auto& sem : image_available_semaphores_) {
    if (sem != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, sem, nullptr);
    }
  }
  for (auto& sem : render_finished_semaphores_) {
    if (sem != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, sem, nullptr);
    }
  }

  // image_available: one per frame-in-flight (used at acquire time, before we
  // know which image we'll get). Indexed by current_frame_.
  // render_finished: one per render target image (used at present time, indexed
  // by the acquired image_index). This avoids reuse conflicts when
  // frames-in-flight != render target image count.
  image_available_semaphores_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);
  render_finished_semaphores_.resize(image_count, VK_NULL_HANDLE);
  images_in_flight_.resize(image_count, VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  // Create image_available semaphores (per frame-in-flight)
  // Semaphore creation is fatal - use hard check
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    checkVkErrors(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                    &image_available_semaphores_[i]));
  }

  // Create render_finished semaphores (per swapchain image)
  for (uint32_t i = 0; i < image_count; i++) {
    checkVkErrors(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                    &render_finished_semaphores_[i]));
  }

  LOG(INFO) << "Created " << kMaxFramesInFlight << " image_available and "
            << image_count << " render_finished semaphores";
  return true;
}

void VkFrameSync::destroy() {
  if (device_ == VK_NULL_HANDLE) {
    return;
  }

  // Wait for device to be idle before destroying sync objects
  checkVkResult(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (destroy)");

  for (auto& fence : in_flight_fences_) {
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(device_, fence, nullptr);
      fence = VK_NULL_HANDLE;
    }
  }
  in_flight_fences_.clear();

  for (auto& sem : image_available_semaphores_) {
    if (sem != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, sem, nullptr);
      sem = VK_NULL_HANDLE;
    }
  }
  image_available_semaphores_.clear();

  for (auto& sem : render_finished_semaphores_) {
    if (sem != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, sem, nullptr);
      sem = VK_NULL_HANDLE;
    }
  }
  render_finished_semaphores_.clear();

  images_in_flight_.clear();
  device_ = VK_NULL_HANDLE;
  current_frame_.store(0);
}

bool VkFrameSync::waitForCurrentFrame() {
  const uint32_t frame = current_frame_.load();
  if (frame >= in_flight_fences_.size()) {
    LOG(ERROR) << "Invalid frame index: " << frame;
    return false;
  }

  VkResult result = vkWaitForFences(device_, 1, &in_flight_fences_[frame],
                                    VK_TRUE, kFenceTimeoutNs);
  if (result == VK_TIMEOUT) {
    LOG(ERROR) << "Fence wait timed out - GPU may be hung";
    return false;
  }
  return checkVkResult(result, "vkWaitForFences (current frame)");
}

bool VkFrameSync::resetCurrentFence() {
  const uint32_t frame = current_frame_.load();
  if (frame >= in_flight_fences_.size()) {
    LOG(ERROR) << "Invalid frame index: " << frame;
    return false;
  }
  return checkVkResult(vkResetFences(device_, 1, &in_flight_fences_[frame]),
                       "vkResetFences");
}

void VkFrameSync::markImageInFlight(uint32_t image_index) {
  const uint32_t frame = current_frame_.load();
  CHECK_LT(image_index, images_in_flight_.size()) << "Invalid image index";
  CHECK_LT(frame, in_flight_fences_.size()) << "Invalid frame index";
  images_in_flight_[image_index] = in_flight_fences_[frame];
}

bool VkFrameSync::waitForImageInFlight(uint32_t image_index) {
  if (image_index >= images_in_flight_.size()) {
    return true;  // No fence to wait on
  }
  if (images_in_flight_[image_index] == VK_NULL_HANDLE) {
    return true;  // No fence to wait on
  }
  VkResult result = vkWaitForFences(device_, 1, &images_in_flight_[image_index],
                                    VK_TRUE, kFenceTimeoutNs);
  if (result == VK_TIMEOUT) {
    LOG(ERROR) << "waitForImageInFlight timed out for image " << image_index;
    return false;
  }
  return checkVkResult(result, "vkWaitForFences (image in flight)");
}

void VkFrameSync::advanceFrame() {
  // Single-threaded write with atomic store for visibility to reader threads.
  uint32_t next = (current_frame_.load() + 1) % kMaxFramesInFlight;
  current_frame_.store(next);
}

}  // namespace renderer
}  // namespace nvblox
