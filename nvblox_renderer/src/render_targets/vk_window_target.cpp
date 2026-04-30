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
#include "nvblox/renderer/render_targets/vk_window_target.h"

#include <algorithm>

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_utils.h"
#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {

VkWindowTarget::~VkWindowTarget() { destroy(); }

bool VkWindowTarget::create(VkDevice device, VkPhysicalDevice physical_device,
                            VkSurfaceKHR surface,
                            uint32_t graphics_queue_family,
                            VkQueue present_queue, uint32_t width,
                            uint32_t height) {
  if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE ||
      surface == VK_NULL_HANDLE) {
    LOG(ERROR) << "Invalid parameters for VkWindowTarget";
    return false;
  }

  initBase(device, physical_device);
  surface_ = surface;
  graphics_queue_family_ = graphics_queue_family;
  present_queue_ = present_queue;

  VkBool32 present_support = VK_FALSE;
  if (!checkVkResult(vkGetPhysicalDeviceSurfaceSupportKHR(
                         physical_device_, graphics_queue_family_, surface_,
                         &present_support),
                     "vkGetPhysicalDeviceSurfaceSupportKHR")) {
    return false;
  }
  if (!present_support) {
    LOG(ERROR) << "Graphics queue family does not support presentation";
    return false;
  }

  return createSwapchain(width, height);
}

bool VkWindowTarget::createSwapchain(uint32_t width, uint32_t height) {
  VkSurfaceCapabilitiesKHR capabilities;
  if (!checkVkResult(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                         physical_device_, surface_, &capabilities),
                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) {
    return false;
  }

  uint32_t format_count = 0;
  if (!checkVkResult(vkGetPhysicalDeviceSurfaceFormatsKHR(
                         physical_device_, surface_, &format_count, nullptr),
                     "vkGetPhysicalDeviceSurfaceFormatsKHR (count)")) {
    return false;
  }
  if (format_count == 0) {
    LOG(ERROR) << "No surface formats available";
    return false;
  }
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  if (!checkVkResult(
          vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_,
                                               &format_count, formats.data()),
          "vkGetPhysicalDeviceSurfaceFormatsKHR (data)")) {
    return false;
  }

  VkSurfaceFormatKHR surface_format = formats[0];
  for (const auto& fmt : formats) {
    if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
        fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      surface_format = fmt;
      break;
    }
  }
  color_format_ = surface_format.format;

  if (capabilities.currentExtent.width != UINT32_MAX) {
    extent_ = capabilities.currentExtent;
  } else {
    extent_.width =
        std::max(capabilities.minImageExtent.width,
                 std::min(capabilities.maxImageExtent.width, width));
    extent_.height =
        std::max(capabilities.minImageExtent.height,
                 std::min(capabilities.maxImageExtent.height, height));
  }

  uint32_t image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 &&
      image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }

  // Save old swapchain for potential reuse by driver (performance optimization)
  VkSwapchainKHR old_swapchain = swapchain_;

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent_;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  create_info.preTransform = capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = old_swapchain;  // Allow driver to reuse resources

  // Swapchain creation is critical - use hard check
  checkVkErrors(
      vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_));

  // Destroy old swapchain now that new one is created
  if (old_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_, old_swapchain, nullptr);
  }

  if (!checkVkResult(
          vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr),
          "vkGetSwapchainImagesKHR (count)")) {
    destroySwapchainResources();
    return false;
  }
  if (image_count == 0) {
    LOG(ERROR) << "No swapchain images available";
    destroySwapchainResources();
    return false;
  }
  images_.resize(image_count);
  if (!checkVkResult(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count,
                                             images_.data()),
                     "vkGetSwapchainImagesKHR (data)")) {
    destroySwapchainResources();
    return false;
  }

  image_views_.resize(image_count);
  for (uint32_t i = 0; i < image_count; i++) {
    // Create image view using helper
    checkVkErrors(createImageView2D(device_, images_[i], color_format_,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    &image_views_[i]));
  }

  if (!createDepthResources()) {
    destroySwapchainResources();
    return false;
  }
  if (!createRenderPass()) {
    destroyDepthResources();
    destroySwapchainResources();
    return false;
  }
  if (!createFramebuffers()) {
    destroyRenderPass();
    destroyDepthResources();
    destroySwapchainResources();
    return false;
  }

  LOG(INFO) << "VkWindowTarget created: " << extent_.width << "x"
            << extent_.height << " with " << images_.size() << " images";
  return true;
}

void VkWindowTarget::destroySwapchainResources() {
  destroyFramebuffers();
  destroyDepthResources();
  destroyRenderPass();

  for (auto iv : image_views_) {
    if (iv != VK_NULL_HANDLE) vkDestroyImageView(device_, iv, nullptr);
  }
  image_views_.clear();
  images_.clear();

  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
}

bool VkWindowTarget::resize(uint32_t width, uint32_t height) {
  // Validate dimensions
  if (width == 0 || height == 0) {
    LOG(WARNING) << "VkWindowTarget::resize: invalid dimensions " << width
                 << "x" << height;
    return false;
  }

  if (swapchain_ != VK_NULL_HANDLE && extent_.width == width &&
      extent_.height == height) {
    return true;
  }
  if (!checkVkResult(vkDeviceWaitIdle(device_),
                     "vkDeviceWaitIdle (before resize)")) {
    return false;
  }

  // Destroy resources that depend on swapchain images, but NOT the swapchain
  // itself. createSwapchain will pass the old swapchain to vkCreateSwapchainKHR
  // via oldSwapchain for potential driver optimization, then destroy it.
  destroyFramebuffers();
  destroyDepthResources();
  destroyRenderPass();

  for (auto iv : image_views_) {
    if (iv != VK_NULL_HANDLE) vkDestroyImageView(device_, iv, nullptr);
  }
  image_views_.clear();
  images_.clear();

  return createSwapchain(width, height);
}

void VkWindowTarget::destroy() {
  if (device_ == VK_NULL_HANDLE) return;
  checkVkResult(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (destroy)");
  destroySwapchainResources();
  device_ = VK_NULL_HANDLE;
}

bool VkWindowTarget::acquireImage(VkSemaphore semaphore,
                                  uint32_t* image_index) {
  if (!image_index) {
    LOG(ERROR) << "image_index pointer is null";
    return false;
  }
  VkResult result =
      vkAcquireNextImageKHR(device_, swapchain_, kAcquireImageTimeoutNs,
                            semaphore, VK_NULL_HANDLE, image_index);
  // Handle expected recoverable cases
  if (result == VK_TIMEOUT) {
    LOG(WARNING) << "Swapchain image acquire timed out";
    return false;
  }
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    return false;  // Caller should handle swapchain recreation
  }
  if (result == VK_ERROR_SURFACE_LOST_KHR) {
    LOG(ERROR) << "Surface lost - window may have been destroyed";
    return false;  // Caller should recreate surface and swapchain
  }
  return checkVkResult(result, "vkAcquireNextImageKHR");
}

bool VkWindowTarget::presentImage(VkSemaphore wait_semaphore,
                                  uint32_t image_index) {
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &wait_semaphore;
  VkSwapchainKHR swapchains[] = {swapchain_};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_index;

  VkResult result = vkQueuePresentKHR(present_queue_, &present_info);
  // Handle expected recoverable cases
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    return false;  // Caller should handle swapchain recreation
  }
  if (result == VK_ERROR_SURFACE_LOST_KHR) {
    LOG(ERROR)
        << "Surface lost during present - window may have been destroyed";
    return false;  // Caller should recreate surface and swapchain
  }
  return checkVkResult(result, "vkQueuePresentKHR");
}

}  // namespace renderer
}  // namespace nvblox
