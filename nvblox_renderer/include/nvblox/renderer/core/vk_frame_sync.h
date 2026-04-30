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

#include <atomic>
#include <cstdint>
#include <vector>

#include <glog/logging.h>
#include <vulkan/vulkan.h>

namespace nvblox {
namespace renderer {

/// Manages frame synchronization primitives (semaphores and fences).
///
/// @note Thread Safety: NOT thread-safe. All methods must be called from the
/// render thread, except currentFrame() which is safe to read from any thread
/// (e.g., for debugging/diagnostics). The other accessors that read
/// current_frame_ (currentImageAvailableSemaphore, currentInFlightFence) are
/// NOT thread-safe because they also access non-atomic vectors.
///
/// Usage: waitForCurrentFrame -> resetCurrentFence -> waitForImageInFlight
///        -> markImageInFlight -> submit commands -> advanceFrame.
class VkFrameSync {
 public:
  /// Maximum number of frames that can be in-flight simultaneously.
  /// This limits how far the CPU can get ahead of the GPU.
  static constexpr int kMaxFramesInFlight = 2;

  /// Timeout for fence waits in nanoseconds (5 seconds).
  static constexpr uint64_t kFenceTimeoutNs = 5'000'000'000;

  VkFrameSync() = default;
  ~VkFrameSync();

  // Non-copyable
  VkFrameSync(const VkFrameSync&) = delete;
  VkFrameSync& operator=(const VkFrameSync&) = delete;

  /// Create synchronization objects.
  /// @param device Vulkan logical device.
  /// @return True if creation succeeded.
  bool create(VkDevice device);

  /// Create or recreate semaphores for render target images.
  /// Call this after render target creation/recreation.
  /// @param image_count Number of render target images.
  /// @return True if creation succeeded.
  bool createRenderTargetSemaphores(uint32_t image_count);

  /// Destroy all synchronization objects.
  void destroy();

  /// Wait for the current frame-in-flight's fence to be signaled.
  /// "Current frame" refers to the frame-in-flight slot (cycling 0 to
  /// kMaxFramesInFlight-1), not a specific rendered frame.
  /// @return True if fence is ready, false on timeout or error.
  bool waitForCurrentFrame();

  /// Reset the current frame-in-flight's fence for reuse.
  /// @return True if reset succeeded, false on error.
  bool resetCurrentFence();

  /// Mark that a render target image is now being used by current frame.
  /// @param image_index Index of the render target image being used.
  void markImageInFlight(uint32_t image_index);

  /// Wait for any previous frame that was using this image.
  /// @param image_index Index of the render target image.
  /// @return True if wait succeeded, false on timeout or error.
  bool waitForImageInFlight(uint32_t image_index);

  /// Advance to the next frame index.
  void advanceFrame();

  /// Get current frame index. Thread-safe (atomic read).
  uint32_t currentFrame() const { return current_frame_.load(); }

  /// Get image available semaphore for current frame.
  /// Signaled when render target image is available (used in acquire).
  /// Indexed by current_frame_ (frame-in-flight), not by image_index.
  VkSemaphore currentImageAvailableSemaphore() const {
    const auto frame = current_frame_.load();
    CHECK_LT(frame, image_available_semaphores_.size())
        << "Frame index out of bounds";
    return image_available_semaphores_[frame];
  }

  /// Get render finished semaphore for a specific render target image.
  /// Use this for signaling in submit and waiting in present.
  /// Indexed by acquired image_index to avoid reuse conflicts when
  /// frames-in-flight != render target image count.
  /// @param image_index Index of the render target image.
  /// @return Semaphore for the specified image index.
  VkSemaphore renderFinishedSemaphore(uint32_t image_index) const {
    CHECK_LT(image_index, render_finished_semaphores_.size())
        << "Invalid image_index";
    return render_finished_semaphores_[image_index];
  }

  /// Get in-flight fence for current frame.
  VkFence currentInFlightFence() const {
    const auto frame = current_frame_.load();
    CHECK_LT(frame, in_flight_fences_.size()) << "Frame index out of bounds";
    return in_flight_fences_[frame];
  }

  /// Check if sync objects are valid.
  bool isValid() const { return device_ != VK_NULL_HANDLE; }

  /// Get the number of render target images (size of
  /// render_finished_semaphores_).
  uint32_t renderTargetImageCount() const {
    return static_cast<uint32_t>(render_finished_semaphores_.size());
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;

  // Fences per frame-in-flight (limits CPU ahead of GPU)
  std::vector<VkFence> in_flight_fences_;

  // Semaphores per render target image (avoids reuse conflicts)
  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;

  // Track which fence is using each render target image
  std::vector<VkFence> images_in_flight_;

  // Atomic for safe reads from other threads (e.g., debugging).
  // Writes are single-threaded from the render loop.
  std::atomic<uint32_t> current_frame_{0};
};

}  // namespace renderer
}  // namespace nvblox
