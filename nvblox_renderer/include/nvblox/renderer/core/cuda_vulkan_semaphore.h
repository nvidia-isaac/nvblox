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

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"

namespace nvblox {
namespace renderer {

class VkContext;

/// Semaphore for GPU-GPU synchronization between CUDA and Vulkan.
/// This allows CUDA to signal when a write is complete, and Vulkan to wait
/// on that signal before reading the data - all on the GPU without CPU sync.
///
/// Usage pattern:
/// 1. Create semaphore: sem.create(ctx)
/// 2. CUDA writes data, then signals: sem.signalFromCuda(stream)
/// 3. Vulkan command buffer waits on sem.vulkanSemaphore() before reading
///
/// @note This class uses timeline semaphores for robust multi-frame operation.
class CudaVulkanSemaphore {
 public:
  CudaVulkanSemaphore() = default;
  ~CudaVulkanSemaphore();

  // Non-copyable
  CudaVulkanSemaphore(const CudaVulkanSemaphore&) = delete;
  CudaVulkanSemaphore& operator=(const CudaVulkanSemaphore&) = delete;

  // Movable
  CudaVulkanSemaphore(CudaVulkanSemaphore&& other) noexcept;
  CudaVulkanSemaphore& operator=(CudaVulkanSemaphore&& other) noexcept;

  /// Create the semaphore with CUDA interop.
  /// @param ctx Vulkan context.
  /// @return True if creation succeeded.
  bool create(VkContext* ctx);

  /// Destroy the semaphore.
  void destroy();

  /// Signal the semaphore from CUDA after a write operation.
  /// Call this after copying data to shared resources.
  /// @param stream CUDA stream that performed the write.
  /// @param[out] signaled_value If non-null, receives the signaled value for
  /// use
  ///             with waitFromCuda(). Pass this value to ensure correct
  ///             synchronization.
  /// @return True if signal succeeded, false if CUDA signaling failed.
  /// @note On failure, the data copy may still have completed. The caller
  /// should
  ///       use CPU synchronization (stream.synchronize()) as a fallback before
  ///       rendering. This method logs a warning on failure but does not abort.
  /// @note The signal value is incremented atomically before the CUDA signal
  ///       operation. If signaling fails, a "gap" in the timeline is created,
  ///       but timeline semaphores handle gaps gracefully (waiting for N
  ///       succeeds when any value >= N is signaled).
  bool signalFromCuda(const CudaStream& stream,
                      uint64_t* signaled_value = nullptr);

  /// Wait on the semaphore from CUDA before a read operation.
  /// @param stream CUDA stream that will read the data.
  /// @param value The specific signal value to wait for. Use the value returned
  ///              by signalFromCuda() to ensure waiting for the correct signal.
  /// @return True if wait succeeded, false if CUDA wait failed.
  /// @note On failure, a warning is logged but the program does not abort.
  ///       The caller may need to use alternative synchronization.
  /// @warning Passing value=0 uses the current signal value at call time, which
  ///          can race with concurrent signalFromCuda() calls. Always prefer
  ///          passing the explicit value returned by signalFromCuda().
  bool waitFromCuda(const CudaStream& stream, uint64_t value = 0);

  /// Get the Vulkan semaphore for use in command buffer submission.
  /// The semaphore should be waited on before rendering.
  VkSemaphore vulkanSemaphore() const { return semaphore_; }

  /// Get the current signal value (for timeline semaphore wait).
  uint64_t currentSignalValue() const { return signal_value_.load(); }

  /// Check if semaphore is valid.
  bool isValid() const { return semaphore_ != VK_NULL_HANDLE; }

 private:
  VkContext* ctx_ = nullptr;

  // Vulkan timeline semaphore
  VkSemaphore semaphore_ = VK_NULL_HANDLE;

  // CUDA external semaphore handle
  cudaExternalSemaphore_t cuda_semaphore_ = nullptr;

  // Timeline semaphore value (monotonically increasing).
  // Atomic to prevent lost increments if signalFromCuda() is called
  // concurrently (though single-threaded use is expected).
  std::atomic<uint64_t> signal_value_{0};
};

}  // namespace renderer
}  // namespace nvblox
