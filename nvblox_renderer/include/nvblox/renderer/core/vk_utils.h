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
#include <limits>

#include <vulkan/vulkan.h>

namespace nvblox {
namespace renderer {

/// NVIDIA vendor ID for identifying NVIDIA GPUs.
constexpr uint32_t kNvidiaVendorId = 0x10DE;

/// Sentinel value indicating no suitable memory type was found.
constexpr uint32_t kMemoryTypeNotFound = std::numeric_limits<uint32_t>::max();

/// Find a suitable memory type for allocation.
/// @param physical_device Vulkan physical device.
/// @param type_filter Bitmask of acceptable memory types.
/// @param properties Required memory property flags.
/// @return Memory type index, or kMemoryTypeNotFound if not found.
uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter,
                        VkMemoryPropertyFlags properties);

/// Create a 2D image view for a given image.
/// @param device Vulkan logical device.
/// @param image VkImage to create the view for.
/// @param format Format of the image view.
/// @param aspect_mask Aspect flags (e.g., VK_IMAGE_ASPECT_COLOR_BIT).
/// @param out_view Output image view (must not be nullptr).
/// @return VK_SUCCESS on success, error code otherwise.
VkResult createImageView2D(VkDevice device, VkImage image, VkFormat format,
                           VkImageAspectFlags aspect_mask,
                           VkImageView* out_view);

/// Create a 2D sampler with common default settings.
/// Default: linear filtering, clamp-to-edge addressing.
/// @param device Vulkan logical device.
/// @param out_sampler Output sampler (must not be nullptr).
/// @return VK_SUCCESS on success, error code otherwise.
VkResult createDefaultSampler2D(VkDevice device, VkSampler* out_sampler);

/// Configuration for creating a 2D image with memory.
struct Image2DCreateInfo {
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageUsageFlags usage = 0;
  VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};

/// Result of creating a 2D image with memory.
struct Image2DResult {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
};

/// Create a 2D image with allocated memory and optional image view.
/// @param device Vulkan logical device.
/// @param physical_device Physical device for memory type lookup.
/// @param create_info Image configuration.
/// @param aspect_mask Aspect flags for image view (set to 0 to skip view
/// creation).
/// @param result Output structure containing created resources.
/// @return true on success, false on failure (resources cleaned up on failure).
bool createImage2D(VkDevice device, VkPhysicalDevice physical_device,
                   const Image2DCreateInfo& create_info,
                   VkImageAspectFlags aspect_mask, Image2DResult* result);

/// Destroy resources created by createImage2D.
/// Safe to call with partially initialized result (handles VK_NULL_HANDLE).
void destroyImage2D(VkDevice device, Image2DResult* result);

/// Create debug utils messenger (loads function dynamically).
VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* create_info,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* debug_messenger);

/// Destroy debug utils messenger (loads function dynamically).
void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debug_messenger,
                                   const VkAllocationCallbacks* allocator);

/// Default validation layer debug callback.
/// Logs warnings and errors via glog.
VKAPI_ATTR VkBool32 VKAPI_CALL defaultDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);

}  // namespace renderer
}  // namespace nvblox
