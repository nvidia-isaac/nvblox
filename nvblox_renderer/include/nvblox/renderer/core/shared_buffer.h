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

#include <cstddef>

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_resource_base.h"

namespace nvblox {
namespace renderer {

class VkContext;

/// Shared buffer for zero-copy CUDA-Vulkan interop.
/// Used for vertex buffers, index buffers, and other GPU data.
///
/// Synchronization:
/// The caller must ensure CUDA writes are complete before Vulkan reads.
/// Currently uses cudaStreamSynchronize (CPU-blocking).
/// Future: CudaVulkanSemaphore will enable GPU-GPU sync without CPU stall.
class SharedBuffer : public SharedResourceBase<SharedBuffer> {
 public:
  /// Type aliases for type introspection (following nvblox pattern)
  using VkResourceType = VkBuffer;
  using CudaResourceType = void*;

  /// Buffer usage hints.
  enum class Usage {
    kVertex,   // Vertex buffer
    kIndex,    // Index buffer
    kStorage,  // Storage buffer (compute)
  };

  SharedBuffer() = default;
  ~SharedBuffer();

  // Non-copyable (inherited from base)
  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;

  // Movable
  SharedBuffer(SharedBuffer&& other) noexcept;
  SharedBuffer& operator=(SharedBuffer&& other) noexcept;

  /// Create a shared buffer.
  /// @param ctx Vulkan context.
  /// @param size Buffer size in bytes.
  /// @param usage Buffer usage.
  /// @return True if creation succeeded.
  bool create(VkContext* ctx, size_t size, Usage usage);

  /// Resize the buffer.
  /// @param new_size New buffer size in bytes.
  /// @return True if resize succeeded.
  bool resize(size_t new_size);

  /// Destroy the buffer.
  void destroy();

  /// Copy data from CUDA memory to the buffer.
  /// The caller is responsible for synchronization before Vulkan reads.
  /// @param src Source CUDA device pointer.
  /// @param size Number of bytes to copy.
  /// @param stream CUDA stream.
  /// @return True if copy succeeded, false on error.
  bool copyFromCuda(const void* src, size_t size, const CudaStream& stream);

  /// Get Vulkan buffer handle.
  VkBuffer buffer() const { return buffer_; }

  /// Get buffer size in bytes.
  size_t size() const { return size_; }

  /// Get CUDA device pointer for direct writes.
  /// @return CUDA device pointer, or nullptr if buffer is not valid.
  /// @note The pointer is valid for the lifetime of this SharedBuffer object
  ///       and becomes invalid after destroy() is called or the object is
  ///       destroyed. Use isValid() to check if the pointer can be safely used.
  void* cudaPtr() const { return cuda_ptr_; }

  /// Check if buffer is valid.
  bool isValid() const { return buffer_ != VK_NULL_HANDLE; }

 private:
  static constexpr Usage kDefaultUsage = Usage::kVertex;

  // Buffer-specific Vulkan resource
  VkBuffer buffer_ = VK_NULL_HANDLE;

  // Buffer-specific CUDA mapping
  void* cuda_ptr_ = nullptr;

  // Buffer properties
  size_t size_ = 0;
  Usage usage_ = kDefaultUsage;
};

}  // namespace renderer
}  // namespace nvblox
