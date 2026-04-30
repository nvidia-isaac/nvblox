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

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <vulkan/vulkan.h>

#include "nvblox/renderer/core/vk_frame_sync.h"
#include "nvblox/renderer/core/vk_render_target.h"

namespace nvblox {
namespace renderer {

/// Vulkan context managing instance, device, and render target.
/// Configured with extensions for CUDA interop.
/// Delegates render target/framebuffer management to IVkRenderTarget
/// and synchronization to VkFrameSync.
///
/// @note Thread Safety: VkContext is NOT thread-safe. All methods must be
/// called from the same thread. CUDA interop resources (UUID, device index)
/// are read-only after initialization and safe to access from any thread.
class VkContext {
 public:
  VkContext() = default;
  ~VkContext();

  // Non-copyable, non-movable (holds Vulkan handles)
  VkContext(const VkContext&) = delete;
  VkContext& operator=(const VkContext&) = delete;
  VkContext(VkContext&&) = delete;
  VkContext& operator=(VkContext&&) = delete;

  /// Initialize Vulkan instance and find a suitable physical device.
  /// @param app_name Application name for Vulkan instance.
  /// @param required_extensions Instance extensions required by the backend
  ///        (e.g., window presentation extensions from GLFW).
  /// @param enable_validation Enable Vulkan validation layers at runtime to
  ///        catch API misuse and report errors via debug callbacks.
  ///        Adds significant runtime overhead; recommended only for debugging.
  /// @return True if initialization succeeded.
  bool init(const std::string& app_name,
            const std::vector<const char*>& required_extensions = {},
            bool enable_validation = false);

  /// Create logical device and queues.
  /// Must be called after init() and before setRenderTarget().
  /// @return True if device creation succeeded.
  bool createDevice();

  /// Set a render target (generic - supports window, headless, XR backends).
  /// This is the preferred method for setting up rendering.
  /// @param target Render target implementing IVkRenderTarget.
  /// @return True if setup succeeded.
  bool setRenderTarget(std::unique_ptr<IVkRenderTarget> target);

  /// Resize the render target (e.g., after window resize).
  /// @param width New width.
  /// @param height New height.
  /// @return True if resize succeeded.
  bool resizeRenderTarget(uint32_t width, uint32_t height);

  /// Clean up render target resources.
  void destroyRenderTarget();

  /// Begin a frame - acquire render target image.
  /// @param image_index Output: index of acquired image.
  /// @return True if image acquired successfully.
  bool beginFrame(uint32_t* image_index);

  /// End a frame - submit the command buffer and display the rendered image.
  /// The display mechanism is determined by the active IVkRenderTarget backend.
  /// @param image_index Index of the render target image to display.
  /// @param cmd Command buffer to submit for execution before display.
  /// @return True if the frame was submitted and displayed successfully.
  bool endFrame(uint32_t image_index, VkCommandBuffer cmd);

  /// Wait for device to be idle.
  /// @return True if the device became idle successfully.
  bool waitIdle();

  // Accessors - core Vulkan objects
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physicalDevice() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue graphicsQueue() const { return graphics_queue_; }
  uint32_t graphicsQueueFamily() const { return graphics_queue_family_; }
  VkCommandPool commandPool() const { return command_pool_; }
  VkPipelineCache pipelineCache() const { return pipeline_cache_; }

  // Render target accessors - delegate to the active render target.
  // These require hasRenderTarget() == true; will abort via CHECK if not set.
  VkRenderPass renderPass() const {
    CHECK(hasRenderTarget()) << "renderPass() called without render target";
    return render_target_->renderPass();
  }
  VkFramebuffer framebuffer(uint32_t index) const {
    CHECK(hasRenderTarget()) << "framebuffer() called without render target";
    return render_target_->framebuffer(index);
  }
  VkExtent2D renderTargetExtent() const {
    CHECK(hasRenderTarget())
        << "renderTargetExtent() called without render target";
    return render_target_->extent();
  }
  VkFormat renderTargetFormat() const {
    CHECK(hasRenderTarget())
        << "renderTargetFormat() called without render target";
    return render_target_->colorFormat();
  }
  uint32_t renderTargetImageCount() const {
    CHECK(hasRenderTarget())
        << "renderTargetImageCount() called without render target";
    return render_target_->imageCount();
  }

  /// Allocate a one-time command buffer for immediate execution.
  /// @return A command buffer in the recording state, or VK_NULL_HANDLE on
  /// failure.
  /// @note The caller must submit the command buffer using
  /// endSingleTimeCommands() when finished recording commands.
  VkCommandBuffer beginSingleTimeCommands();

  /// End recording and submit a one-time command buffer.
  /// @param cmd Command buffer allocated by beginSingleTimeCommands().
  ///        VK_NULL_HANDLE is safely ignored.
  /// @note This function waits for the queue to become idle after submission.
  ///       The command buffer is automatically freed after execution completes.
  void endSingleTimeCommands(VkCommandBuffer cmd);

  /// Get the CUDA device UUID for interop.
  /// @return Pointer to 16-byte UUID, or nullptr if not available.
  const uint8_t* cudaDeviceUuid() const { return cuda_device_uuid_; }

  /// Get the matched CUDA device index.
  /// @return CUDA device index, or -1 if not initialized.
  int cudaDeviceIndex() const { return cuda_device_index_; }

  /// Check if a render target has been set.
  bool hasRenderTarget() const { return render_target_ != nullptr; }

  /// Get the underlying render target, e.g. to query backend-specific
  /// properties like requiresPresentation()
  IVkRenderTarget* renderTarget() const { return render_target_.get(); }

 private:
  bool createInstance(const std::string& app_name,
                      const std::vector<const char*>& required_extensions,
                      bool enable_validation);
  bool selectPhysicalDevice();
  bool createCommandPool();

  // Instance
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

  // Physical device
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  uint8_t cuda_device_uuid_[16] = {};
  int cuda_device_index_ = -1;

  // Logical device
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  uint32_t graphics_queue_family_ = 0;

  // Command pool
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  // Pipeline cache for faster pipeline creation
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;

  // Delegated helpers
  std::unique_ptr<IVkRenderTarget> render_target_;
  std::unique_ptr<VkFrameSync> frame_sync_;
};

}  // namespace renderer
}  // namespace nvblox
