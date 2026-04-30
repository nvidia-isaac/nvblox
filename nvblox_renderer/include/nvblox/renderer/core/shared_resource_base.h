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

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"

namespace nvblox {
namespace renderer {

// Forward declaration: full include avoided because vk_context.h pulls in C++17
// headers that CUDA 11.4's nvcc on Jetson JP5 cannot parse.
class VkContext;

/// Base class for CUDA-Vulkan shared resources (SharedBuffer, SharedTexture).
/// Provides interop memory management and CUDA import helpers. Derived classes
/// implement resource-specific creation, mapping, and copy. Resize should
/// destroy() then create() with new dims.
///
/// @tparam Derived The derived class type.
template <typename Derived>
class SharedResourceBase {
 public:
  /// Nested types for type introspection (following nvblox pattern)
  using DerivedType = Derived;

  SharedResourceBase() = default;
  ~SharedResourceBase() = default;

  // Non-copyable (derived classes should also be non-copyable)
  SharedResourceBase(const SharedResourceBase&) = delete;
  SharedResourceBase& operator=(const SharedResourceBase&) = delete;

  // Move operations should be implemented by derived classes
  SharedResourceBase(SharedResourceBase&&) noexcept = default;
  SharedResourceBase& operator=(SharedResourceBase&&) noexcept = default;

  /// Get the Vulkan context.
  VkContext* context() const { return ctx_; }

 protected:
  /// Helper to access derived class.
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  /// Validate VkContext for resource creation.
  /// @param ctx The context to validate.
  /// @return True if context is valid.
  bool validateContext(VkContext* ctx) const;

  /// Find a suitable memory type index.
  /// @param memory_type_bits Bit field of suitable memory types.
  /// @param properties Required memory properties.
  /// @return Memory type index, or kMemoryTypeNotFound if not found.
  uint32_t findMemoryType(uint32_t memory_type_bits,
                          VkMemoryPropertyFlags properties) const;

  /// Allocate Vulkan memory with export capability for CUDA interop.
  /// @param size Allocation size in bytes.
  /// @param memory_type_index Memory type index.
  /// @return True if allocation succeeded.
  bool allocateExportableMemory(VkDeviceSize size, uint32_t memory_type_index);

  /// Import Vulkan memory to CUDA as external memory.
  /// @param size Total memory size for CUDA import.
  /// @return True if import succeeded.
  bool importToCuda(VkDeviceSize size);

  /// Destroy base class resources (CUDA external memory, Vulkan memory).
  /// Derived classes should call this in their destroy() method.
  void destroyBase();

  /// Move base class members from another instance.
  /// @param other Source instance to move from.
  void moveBaseFrom(SharedResourceBase& other);

  /// Reset base class members to default state.
  void resetBase();

  // Common members for all shared resources
  VkContext* ctx_ = nullptr;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  cudaExternalMemory_t cuda_external_memory_ = nullptr;
};

}  // namespace renderer
}  // namespace nvblox

#include "nvblox/renderer/core/cuda/impl/shared_resource_base_impl.cuh"
