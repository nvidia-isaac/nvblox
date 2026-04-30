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
#include "nvblox/renderer/core/vk_utils.h"

#include <glog/logging.h>

namespace nvblox {
namespace renderer {

uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  LOG(ERROR) << "Failed to find suitable memory type";
  return kMemoryTypeNotFound;
}

VkResult createImageView2D(VkDevice device, VkImage image, VkFormat format,
                           VkImageAspectFlags aspect_mask,
                           VkImageView* out_view) {
  if (!out_view) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (device == VK_NULL_HANDLE) {
    LOG(ERROR) << "Cannot create image view with null device";
    return VK_ERROR_DEVICE_LOST;
  }

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask = aspect_mask;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  return vkCreateImageView(device, &view_info, nullptr, out_view);
}

VkResult createDefaultSampler2D(VkDevice device, VkSampler* out_sampler) {
  if (!out_sampler) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (device == VK_NULL_HANDLE) {
    LOG(ERROR) << "Cannot create sampler with null device";
    return VK_ERROR_DEVICE_LOST;
  }

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  return vkCreateSampler(device, &sampler_info, nullptr, out_sampler);
}

bool createImage2D(VkDevice device, VkPhysicalDevice physical_device,
                   const Image2DCreateInfo& create_info,
                   VkImageAspectFlags aspect_mask, Image2DResult* result) {
  if (!result) {
    LOG(ERROR) << "createImage2D: null result pointer";
    return false;
  }

  *result = {};  // Zero-initialize

  // Validate dimensions
  if (create_info.width == 0 || create_info.height == 0) {
    LOG(ERROR) << "createImage2D: invalid dimensions (" << create_info.width
               << "x" << create_info.height << ")";
    return false;
  }

  // Create image
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = create_info.width;
  image_info.extent.height = create_info.height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = create_info.format;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = create_info.usage;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult vk_result =
      vkCreateImage(device, &image_info, nullptr, &result->image);
  if (vk_result != VK_SUCCESS) {
    LOG(ERROR) << "vkCreateImage failed with result: " << vk_result;
    return false;
  }

  // Get memory requirements
  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(device, result->image, &mem_req);

  // Find suitable memory type
  uint32_t memory_type = findMemoryType(physical_device, mem_req.memoryTypeBits,
                                        create_info.memory_properties);
  if (memory_type == kMemoryTypeNotFound) {
    LOG(ERROR) << "Failed to find suitable memory type for image";
    destroyImage2D(device, result);
    return false;
  }

  // Allocate memory
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = memory_type;

  vk_result = vkAllocateMemory(device, &alloc_info, nullptr, &result->memory);
  if (vk_result != VK_SUCCESS) {
    LOG(ERROR) << "vkAllocateMemory failed with result: " << vk_result;
    destroyImage2D(device, result);
    return false;
  }

  // Bind memory
  vk_result = vkBindImageMemory(device, result->image, result->memory, 0);
  if (vk_result != VK_SUCCESS) {
    LOG(ERROR) << "vkBindImageMemory failed with result: " << vk_result;
    destroyImage2D(device, result);
    return false;
  }

  // Create image view if aspect mask is provided
  if (aspect_mask != 0) {
    vk_result = createImageView2D(device, result->image, create_info.format,
                                  aspect_mask, &result->view);
    if (vk_result != VK_SUCCESS) {
      LOG(ERROR) << "createImageView2D failed with result: " << vk_result;
      destroyImage2D(device, result);
      return false;
    }
  }

  return true;
}

void destroyImage2D(VkDevice device, Image2DResult* result) {
  if (!result) {
    return;
  }

  if (result->view != VK_NULL_HANDLE) {
    vkDestroyImageView(device, result->view, nullptr);
    result->view = VK_NULL_HANDLE;
  }
  if (result->image != VK_NULL_HANDLE) {
    vkDestroyImage(device, result->image, nullptr);
    result->image = VK_NULL_HANDLE;
  }
  if (result->memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, result->memory, nullptr);
    result->memory = VK_NULL_HANDLE;
  }
}

VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* create_info,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* debug_messenger) {
  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  if (func) {
    return func(instance, create_info, allocator, debug_messenger);
  }
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debug_messenger,
                                   const VkAllocationCallbacks* allocator) {
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (func) {
    func(instance, debug_messenger, allocator);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
defaultDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                     VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                     const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                     void* /*user_data*/) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    LOG(WARNING) << "Vulkan: " << callback_data->pMessage;
  }
  return VK_FALSE;
}

}  // namespace renderer
}  // namespace nvblox
