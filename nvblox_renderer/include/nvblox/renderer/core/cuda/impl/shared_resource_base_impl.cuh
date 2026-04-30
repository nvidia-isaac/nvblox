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

#include <unistd.h>  // close()

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/shared_resource_base.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {

template <typename Derived>
bool SharedResourceBase<Derived>::validateContext(VkContext* ctx) const {
  if (ctx == nullptr) {
    LOG(ERROR) << "SharedResource requires non-null VkContext";
    return false;
  }
  if (ctx->device() == VK_NULL_HANDLE) {
    LOG(ERROR) << "SharedResource requires VkContext with valid device";
    return false;
  }
  return true;
}

template <typename Derived>
uint32_t SharedResourceBase<Derived>::findMemoryType(
    uint32_t memory_type_bits, VkMemoryPropertyFlags properties) const {
  if (!ctx_) {
    return kMemoryTypeNotFound;
  }

  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(ctx_->physicalDevice(), &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((memory_type_bits & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  return kMemoryTypeNotFound;
}

template <typename Derived>
bool SharedResourceBase<Derived>::allocateExportableMemory(
    VkDeviceSize size, uint32_t memory_type_index) {
  if (!ctx_ || memory_type_index == kMemoryTypeNotFound) {
    LOG(ERROR) << "Invalid context or memory type for allocation";
    return false;
  }

  VkDevice device = ctx_->device();

  VkExportMemoryAllocateInfo export_info = {};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = size;
  alloc_info.memoryTypeIndex = memory_type_index;

  if (!checkVkResult(vkAllocateMemory(device, &alloc_info, nullptr, &memory_),
                     "vkAllocateMemory for exportable memory")) {
    return false;
  }

  return true;
}

template <typename Derived>
bool SharedResourceBase<Derived>::importToCuda(VkDeviceSize size) {
  if (!ctx_ || memory_ == VK_NULL_HANDLE) {
    LOG(ERROR) << "Invalid context or memory for CUDA import";
    return false;
  }

  VkDevice device = ctx_->device();

  VkMemoryGetFdInfoKHR fd_info = {};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.memory = memory_;
  fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  auto vkGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
  if (!vkGetMemoryFdKHR) {
    LOG(ERROR) << "vkGetMemoryFdKHR not available";
    return false;
  }

  int fd;
  if (!checkVkResult(vkGetMemoryFdKHR(device, &fd_info, &fd),
                     "vkGetMemoryFdKHR for CUDA import")) {
    return false;
  }

  cudaExternalMemoryHandleDesc cuda_ext_desc = {};
  cuda_ext_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  cuda_ext_desc.handle.fd = fd;
  cuda_ext_desc.size = size;

  // CUDA takes ownership of the fd on success and will close it when the
  // external memory is destroyed. We only close it on failure.
  cudaError_t err =
      cudaImportExternalMemory(&cuda_external_memory_, &cuda_ext_desc);
  if (err != cudaSuccess) {
    LOG(ERROR) << "Failed to import external memory to CUDA: "
               << cudaGetErrorString(err);
    close(fd);
    return false;
  }

  return true;
}

template <typename Derived>
void SharedResourceBase<Derived>::destroyBase() {
  if (cuda_external_memory_) {
    cudaError_t err = cudaDestroyExternalMemory(cuda_external_memory_);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to destroy CUDA external memory: "
                   << cudaGetErrorString(err);
    }
    cuda_external_memory_ = nullptr;
  }

  if (ctx_ && memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(ctx_->device(), memory_, nullptr);
    memory_ = VK_NULL_HANDLE;
  }
}

template <typename Derived>
void SharedResourceBase<Derived>::moveBaseFrom(SharedResourceBase& other) {
  ctx_ = other.ctx_;
  memory_ = other.memory_;
  cuda_external_memory_ = other.cuda_external_memory_;

  other.resetBase();
}

template <typename Derived>
void SharedResourceBase<Derived>::resetBase() {
  ctx_ = nullptr;
  memory_ = VK_NULL_HANDLE;
  cuda_external_memory_ = nullptr;
}

}  // namespace renderer
}  // namespace nvblox
