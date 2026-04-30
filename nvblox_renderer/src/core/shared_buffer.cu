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

#include "nvblox/renderer/core/shared_buffer.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {

SharedBuffer::~SharedBuffer() { destroy(); }

SharedBuffer::SharedBuffer(SharedBuffer&& other) noexcept {
  *this = std::move(other);
}

SharedBuffer& SharedBuffer::operator=(SharedBuffer&& other) noexcept {
  if (this != &other) {
    destroy();

    // Move base class members
    moveBaseFrom(other);

    // Move buffer-specific members
    buffer_ = other.buffer_;
    cuda_ptr_ = other.cuda_ptr_;
    size_ = other.size_;
    usage_ = other.usage_;

    // Reset other's buffer-specific members
    other.buffer_ = VK_NULL_HANDLE;
    other.cuda_ptr_ = nullptr;
    other.size_ = 0;
    other.usage_ = kDefaultUsage;  // Reset to default
  }
  return *this;
}

bool SharedBuffer::create(VkContext* ctx, size_t size, Usage usage) {
  // Validate context using base class helper
  if (!validateContext(ctx)) {
    return false;
  }
  if (size == 0) {
    LOG(ERROR) << "SharedBuffer size must be positive";
    return false;
  }

  ctx_ = ctx;
  size_ = size;
  usage_ = usage;

  VkDevice device = ctx_->device();

  // Determine Vulkan buffer usage flags
  VkBufferUsageFlags vk_usage = 0;
  switch (usage) {
    case Usage::kVertex:
      vk_usage =
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      break;
    case Usage::kIndex:
      vk_usage =
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      break;
    case Usage::kStorage:
      vk_usage =
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      break;
  }

  // Create buffer with external memory support (uses FD for CUDA interop)
  VkExternalMemoryBufferCreateInfo external_info = {};
  external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = &external_info;
  buffer_info.size = size;
  buffer_info.usage = vk_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  checkVkErrors(vkCreateBuffer(device, &buffer_info, nullptr, &buffer_));

  // Get memory requirements
  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, buffer_, &mem_requirements);

  // Store actual allocated size (may be larger than requested due to alignment)
  VkDeviceSize allocated_size = mem_requirements.size;

  // Find suitable memory type using base class helper
  uint32_t memory_type_index = findMemoryType(
      mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memory_type_index == kMemoryTypeNotFound) {
    LOG(ERROR) << "Failed to find suitable memory type for buffer";
    vkDestroyBuffer(device, buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    return false;
  }

  // Allocate exportable memory using base class helper
  if (!allocateExportableMemory(allocated_size, memory_type_index)) {
    vkDestroyBuffer(device, buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    return false;
  }

  // Bind memory to buffer
  checkVkErrors(vkBindBufferMemory(device, buffer_, memory_, 0));

  // Import to CUDA using base class helper
  if (!importToCuda(allocated_size)) {
    destroy();
    return false;
  }

  // Map the buffer to CUDA (use allocated_size to match imported memory)
  cudaExternalMemoryBufferDesc buffer_desc = {};
  buffer_desc.offset = 0;
  buffer_desc.size = allocated_size;
  buffer_desc.flags = 0;

  cudaError_t err = cudaExternalMemoryGetMappedBuffer(
      &cuda_ptr_, cuda_external_memory_, &buffer_desc);
  if (err != cudaSuccess) {
    LOG(ERROR) << "Failed to get mapped buffer: " << cudaGetErrorString(err);
    destroy();
    return false;
  }

  LOG(INFO) << "Created SharedBuffer: " << size << " bytes";
  return true;
}

bool SharedBuffer::resize(size_t new_size) {
  if (new_size <= size_) {
    return true;  // No resize needed
  }

  // Wait for GPU to finish using the current buffer before destroying it
  // This prevents validation errors when the buffer is still referenced by
  // command buffers
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Create new buffer first to preserve state on failure
  SharedBuffer new_buffer;
  if (!new_buffer.create(ctx_, new_size, usage_)) {
    LOG(ERROR) << "Failed to create resized buffer";
    return false;  // Keep existing buffer intact
  }

  // Success - swap resources
  *this = std::move(new_buffer);
  return true;
}

void SharedBuffer::destroy() {
  // Wait for GPU to finish using the buffer before destroying resources.
  // This prevents use-after-free if GPU operations are still in flight.
  if (ctx_) {
    ctx_->waitIdle();
  }

  // Clean up buffer-specific CUDA mapping (part of external memory, no free
  // needed)
  cuda_ptr_ = nullptr;

  // Destroy Vulkan buffer before base cleanup frees the memory
  if (ctx_ && buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(ctx_->device(), buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
  }

  // Destroy base class resources (sync, CUDA external memory, Vulkan memory)
  destroyBase();

  size_ = 0;
}

bool SharedBuffer::copyFromCuda(const void* src, size_t size,
                                const CudaStream& stream) {
  if (!src) {
    LOG(ERROR) << "Invalid copy: source pointer is null";
    return false;
  }
  if (!cuda_ptr_ || size > size_) {
    LOG(ERROR) << "Invalid copy: buffer not valid or size too large";
    return false;
  }

  cudaStream_t cuda_stream = stream;
  cudaError_t err = cudaMemcpyAsync(cuda_ptr_, src, size,
                                    cudaMemcpyDeviceToDevice, cuda_stream);
  if (err != cudaSuccess) {
    LOG(ERROR) << "cudaMemcpyAsync failed: " << cudaGetErrorString(err);
    return false;
  }

  return true;
}

}  // namespace renderer
}  // namespace nvblox
