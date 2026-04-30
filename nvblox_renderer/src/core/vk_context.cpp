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

#include "nvblox/renderer/core/vk_context.h"

#include <cstring>
#include <set>
#include <vector>

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {

VkContext::~VkContext() {
  waitIdle();

  render_target_.reset();
  frame_sync_.reset();

  if (command_pool_) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
  }

  if (pipeline_cache_) {
    vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
  }

  if (device_) {
    vkDestroyDevice(device_, nullptr);
  }

  if (debug_messenger_) {
    destroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
  }

  if (instance_) {
    vkDestroyInstance(instance_, nullptr);
  }
}

bool VkContext::init(const std::string& app_name,
                     const std::vector<const char*>& required_extensions,
                     bool enable_validation) {
  if (!createInstance(app_name, required_extensions, enable_validation)) {
    return false;
  }

  if (!selectPhysicalDevice()) {
    return false;
  }

  return true;
}

bool VkContext::createInstance(
    const std::string& app_name,
    const std::vector<const char*>& required_extensions,
    bool enable_validation) {
  // Check validation layer support
  const std::vector<const char*> validation_layers = {
      "VK_LAYER_KHRONOS_validation"};

  if (enable_validation) {
    uint32_t layer_count;
    VkResult result = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    if (result != VK_SUCCESS) {
      LOG(WARNING) << "Failed to enumerate instance layer properties";
      enable_validation = false;
    } else {
      std::vector<VkLayerProperties> available_layers(layer_count);
      result = vkEnumerateInstanceLayerProperties(&layer_count,
                                                  available_layers.data());
      if (result != VK_SUCCESS) {
        LOG(WARNING) << "Failed to enumerate instance layer properties";
        enable_validation = false;
      } else {
        for (const char* layer_name : validation_layers) {
          bool found = false;
          for (const auto& layer : available_layers) {
            if (strcmp(layer_name, layer.layerName) == 0) {
              found = true;
              break;
            }
          }
          if (!found) {
            LOG(WARNING) << "Validation layer not available: " << layer_name;
            enable_validation = false;
            break;
          }
        }
      }
    }
  }

  // Application info
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = app_name.c_str();
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "nvblox_renderer";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  // Start with required extensions (e.g., from GLFW for window presentation)
  std::vector<const char*> extensions = required_extensions;

  // Required for querying device UUID (CUDA device matching) and as a
  // dependency of the external memory/semaphore capabilities extensions.
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  // Instance-level capabilities for CUDA interop (external memory/semaphore).
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

  if (enable_validation) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // Create instance
  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
  if (enable_validation) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();

    debug_create_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_create_info.pfnUserCallback = defaultDebugCallback;
    create_info.pNext = &debug_create_info;
  }

  checkVkErrors(vkCreateInstance(&create_info, nullptr, &instance_));

  // Create debug messenger (optional, soft check)
  if (enable_validation) {
    checkVkResult(createDebugUtilsMessengerEXT(instance_, &debug_create_info,
                                               nullptr, &debug_messenger_),
                  "createDebugUtilsMessengerEXT");
  }

  LOG(INFO) << "Created Vulkan instance";
  return true;
}

bool VkContext::selectPhysicalDevice() {
  // Get the active CUDA device - this ensures we match whatever device
  // nvblox and other CUDA code is using
  int cuda_device;
  cudaError_t cuda_err = cudaGetDevice(&cuda_device);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "Failed to get current CUDA device: "
               << cudaGetErrorString(cuda_err);
    return false;
  }

  // Get CUDA device properties (includes UUID for matching)
  cudaDeviceProp cuda_props;
  cuda_err = cudaGetDeviceProperties(&cuda_props, cuda_device);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "Failed to get CUDA device properties: "
               << cudaGetErrorString(cuda_err);
    return false;
  }

  LOG(INFO) << "CUDA active device " << cuda_device << ": " << cuda_props.name;

  // Enumerate Vulkan physical devices
  uint32_t device_count = 0;
  if (!checkVkResult(
          vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
          "vkEnumeratePhysicalDevices (count)")) {
    return false;
  }

  if (device_count == 0) {
    LOG(ERROR) << "No Vulkan-capable GPUs found";
    return false;
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  if (!checkVkResult(
          vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices (data)")) {
    return false;
  }

  LOG(INFO) << "Found " << device_count << " Vulkan physical device(s):";

  // Find the Vulkan device that matches the active CUDA device by UUID
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    // Skip non-NVIDIA devices (can't do CUDA interop)
    if (props.vendorID != kNvidiaVendorId) {
      LOG(INFO) << "  - " << props.deviceName << " (non-NVIDIA, skipped)";
      continue;
    }

    // Get device UUID for comparison with CUDA
    VkPhysicalDeviceIDProperties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(device, &props2);

    // Compare UUIDs - cuda_props.uuid is a cudaUUID_t (16 bytes)
    bool uuid_match =
        (std::memcmp(id_props.deviceUUID, &cuda_props.uuid, 16) == 0);

    if (!uuid_match) {
      LOG(INFO) << "  - " << props.deviceName
                << " (UUID mismatch with CUDA device)";
      continue;
    }

    LOG(INFO) << "  - " << props.deviceName << " [MATCHES CUDA device "
              << cuda_device << "]";

    // Found matching device - verify required extensions
    uint32_t ext_count;
    if (!checkVkResult(vkEnumerateDeviceExtensionProperties(
                           device, nullptr, &ext_count, nullptr),
                       "vkEnumerateDeviceExtensionProperties (count)")) {
      continue;
    }
    std::vector<VkExtensionProperties> available_exts(ext_count);
    if (!checkVkResult(vkEnumerateDeviceExtensionProperties(
                           device, nullptr, &ext_count, available_exts.data()),
                       "vkEnumerateDeviceExtensionProperties (data)")) {
      continue;
    }

    std::set<std::string> required_exts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };

    for (const auto& ext : available_exts) {
      required_exts.erase(ext.extensionName);
    }

    if (!required_exts.empty()) {
      LOG(ERROR) << "CUDA-matched device " << props.deviceName
                 << " missing required Vulkan extensions for interop";
      for (const auto& missing : required_exts) {
        LOG(ERROR) << "  Missing: " << missing;
      }
      return false;
    }

    // Success - use this device
    physical_device_ = device;
    cuda_device_index_ = cuda_device;
    std::memcpy(cuda_device_uuid_, id_props.deviceUUID,
                sizeof(cuda_device_uuid_));

    LOG(INFO) << "Selected GPU: " << props.deviceName << " (CUDA device "
              << cuda_device << ")";
    return true;
  }

  LOG(ERROR) << "No Vulkan device found matching CUDA device " << cuda_device
             << " (" << cuda_props.name << ")";
  LOG(ERROR) << "Ensure the GPU is visible to both Vulkan and CUDA";
  return false;
}

bool VkContext::createDevice() {
  // Find graphics queue family
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                           &queue_family_count, nullptr);
  if (queue_family_count == 0) {
    LOG(ERROR) << "No queue families found on device";
    return false;
  }
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &queue_family_count, queue_families.data());

  // Find a graphics-capable queue family
  bool found_graphics_queue = false;
  for (uint32_t i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      graphics_queue_family_ = i;
      found_graphics_queue = true;
      break;
    }
  }

  if (!found_graphics_queue) {
    LOG(ERROR) << "No graphics queue family found on device";
    return false;
  }

  // Queue create info
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = graphics_queue_family_;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  // Swapchain for presentation; external memory/semaphore extensions for
  // CUDA interop (importing Vulkan buffers and sync primitives into CUDA).
  std::vector<const char*> device_extensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
  };

  // Device features
  VkPhysicalDeviceFeatures device_features{};
  device_features.wideLines = VK_TRUE;    // For line width > 1
  device_features.largePoints = VK_TRUE;  // For point size > 1
  device_features.fillModeNonSolid =
      VK_TRUE;  // For wireframe rendering (VK_POLYGON_MODE_LINE)

  // Create device
  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount = 1;
  create_info.pQueueCreateInfos = &queue_create_info;
  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();

  checkVkErrors(
      vkCreateDevice(physical_device_, &create_info, nullptr, &device_));

  vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
  if (graphics_queue_ == VK_NULL_HANDLE) {
    LOG(ERROR) << "Failed to get graphics queue";
    return false;
  }

  if (!createCommandPool()) {
    return false;
  }

  // Create pipeline cache for faster pipeline creation (optional, soft check)
  VkPipelineCacheCreateInfo cache_info{};
  cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  cache_info.initialDataSize = 0;
  cache_info.pInitialData = nullptr;

  // Not a fatal error - we can continue without the cache
  checkVkResult(
      vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_),
      "vkCreatePipelineCache");

  LOG(INFO) << "Created Vulkan logical device";
  return true;
}

bool VkContext::setRenderTarget(std::unique_ptr<IVkRenderTarget> target) {
  if (!target) {
    LOG(ERROR) << "Cannot set null render target";
    return false;
  }

  render_target_ = std::move(target);

  // Create sync objects for the render target
  frame_sync_ = std::make_unique<VkFrameSync>();
  if (!frame_sync_->create(device_)) {
    LOG(ERROR) << "Failed to create frame sync objects";
    frame_sync_.reset();
    render_target_.reset();
    return false;
  }
  if (!frame_sync_->createRenderTargetSemaphores(
          render_target_->imageCount())) {
    LOG(ERROR) << "Failed to create swapchain semaphores";
    frame_sync_.reset();
    render_target_.reset();
    return false;
  }

  LOG(INFO) << "Render target set with " << render_target_->imageCount()
            << " images";
  return true;
}

bool VkContext::resizeRenderTarget(uint32_t width, uint32_t height) {
  if (!frame_sync_ || !render_target_) {
    LOG(ERROR) << "Cannot resize: render target not set";
    return false;
  }
  if (!waitIdle()) {
    return false;
  }

  if (!render_target_->resize(width, height)) {
    return false;
  }

  return frame_sync_->createRenderTargetSemaphores(
      render_target_->imageCount());
}

void VkContext::destroyRenderTarget() {
  // Helpers handle their own cleanup
  render_target_.reset();
  frame_sync_.reset();
}

bool VkContext::beginFrame(uint32_t* image_index) {
  if (!image_index) {
    LOG(ERROR) << "image_index pointer is null";
    return false;
  }
  if (!frame_sync_ || !render_target_) {
    LOG(ERROR) << "VkContext not initialized";
    return false;
  }

  if (!frame_sync_->waitForCurrentFrame()) {
    return false;
  }

  if (!render_target_->acquireImage(
          frame_sync_->currentImageAvailableSemaphore(), image_index)) {
    return false;  // Need resize
  }

  if (!frame_sync_->waitForImageInFlight(*image_index)) {
    LOG(ERROR) << "Failed to wait for image in flight";
    return false;
  }
  frame_sync_->markImageInFlight(*image_index);
  if (!frame_sync_->resetCurrentFence()) {
    LOG(ERROR) << "Failed to reset current fence";
    return false;
  }
  return true;
}

bool VkContext::endFrame(uint32_t image_index, VkCommandBuffer cmd) {
  if (!frame_sync_ || !render_target_) {
    LOG(ERROR) << "VkContext not initialized";
    return false;
  }
  uint32_t image_count = frame_sync_->renderTargetImageCount();
  if (image_count == 0) {
    LOG(ERROR) << "No render target images available";
    return false;
  }
  if (image_index >= image_count) {
    LOG(ERROR) << "Invalid image_index " << image_index
               << " (max: " << image_count - 1 << ")";
    return false;
  }

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  // For targets that require presentation (window), use semaphore
  // synchronization. For headless targets, skip semaphores since there's no
  // presentation engine.
  VkSemaphore wait_semaphores[] = {
      frame_sync_->currentImageAvailableSemaphore()};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signal_semaphores[] = {
      frame_sync_->renderFinishedSemaphore(image_index)};

  if (render_target_->requiresPresentation()) {
    // Wait on the image_available semaphore that was signaled during acquire
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;

    // Signal the render_finished semaphore for THIS IMAGE (not current frame).
    // This ensures each swapchain image has its own semaphore, preventing
    // reuse conflicts when frames-in-flight != swapchain image count.
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;
  }

  if (!checkVkResult(vkQueueSubmit(graphics_queue_, 1, &submit_info,
                                   frame_sync_->currentInFlightFence()),
                     "vkQueueSubmit (draw command buffer)")) {
    return false;
  }

  // Present via render target (waits on same semaphore indexed by image_index)
  // Headless targets treat this as a no-op.
  bool success = render_target_->presentImage(
      frame_sync_->renderFinishedSemaphore(image_index), image_index);

  // Always advance even on present failure to keep frame sync state consistent.
  frame_sync_->advanceFrame();
  return success;
}

bool VkContext::waitIdle() {
  if (!device_) {
    return true;
  }
  return checkVkResult(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
}

bool VkContext::createCommandPool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = graphics_queue_family_;

  checkVkErrors(
      vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));
  return true;
}

VkCommandBuffer VkContext::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool_;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (!checkVkResult(vkAllocateCommandBuffers(device_, &alloc_info, &cmd),
                     "vkAllocateCommandBuffers (single-time)")) {
    return VK_NULL_HANDLE;
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (!checkVkResult(vkBeginCommandBuffer(cmd, &begin_info),
                     "vkBeginCommandBuffer (single-time)")) {
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
    return VK_NULL_HANDLE;
  }
  return cmd;
}

void VkContext::endSingleTimeCommands(VkCommandBuffer cmd) {
  if (cmd == VK_NULL_HANDLE) {
    return;
  }

  if (!checkVkResult(vkEndCommandBuffer(cmd),
                     "vkEndCommandBuffer (single-time)")) {
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
    return;
  }

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  checkVkErrors(
      vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE));

  // Wait for the queue to finish - soft check, just log warning on failure
  checkVkResult(vkQueueWaitIdle(graphics_queue_),
                "vkQueueWaitIdle (single-time)");

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

}  // namespace renderer
}  // namespace nvblox
