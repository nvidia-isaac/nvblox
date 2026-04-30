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

#include "nvblox/renderer/core/cuda_vulkan_semaphore.h"

#include <unistd.h>

#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_context.h"

namespace nvblox {
namespace renderer {

CudaVulkanSemaphore::~CudaVulkanSemaphore() { destroy(); }

CudaVulkanSemaphore::CudaVulkanSemaphore(CudaVulkanSemaphore&& other) noexcept {
  *this = std::move(other);
}

CudaVulkanSemaphore& CudaVulkanSemaphore::operator=(
    CudaVulkanSemaphore&& other) noexcept {
  if (this != &other) {
    destroy();
    ctx_ = other.ctx_;
    semaphore_ = other.semaphore_;
    cuda_semaphore_ = other.cuda_semaphore_;
    signal_value_.store(other.signal_value_.load());

    other.ctx_ = nullptr;
    other.semaphore_ = VK_NULL_HANDLE;
    other.cuda_semaphore_ = nullptr;
    other.signal_value_.store(0);
  }
  return *this;
}

bool CudaVulkanSemaphore::create(VkContext* ctx) {
  // Input validation
  if (ctx == nullptr || ctx->device() == VK_NULL_HANDLE) {
    LOG(ERROR) << "CudaVulkanSemaphore::create requires valid VkContext";
    return false;
  }

  ctx_ = ctx;
  VkDevice device = ctx_->device();

  // Create timeline semaphore with export capability
  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = 0;

  VkExportSemaphoreCreateInfo export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  export_info.pNext = &type_info;
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkSemaphoreCreateInfo sem_info{};
  sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sem_info.pNext = &export_info;

  checkVkErrors(vkCreateSemaphore(device, &sem_info, nullptr, &semaphore_));

  // Export semaphore to file descriptor
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.semaphore = semaphore_;
  fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  auto vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
  if (!vkGetSemaphoreFdKHR) {
    LOG(ERROR) << "vkGetSemaphoreFdKHR not available";
    destroy();
    return false;
  }

  int fd;
  if (!checkVkResult(vkGetSemaphoreFdKHR(device, &fd_info, &fd),
                     "vkGetSemaphoreFdKHR")) {
    destroy();
    return false;
  }

  // Import into CUDA
  cudaExternalSemaphoreHandleDesc cuda_sem_desc{};
  cuda_sem_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  cuda_sem_desc.handle.fd = fd;
  cuda_sem_desc.flags = 0;

  // CUDA takes ownership of fd on success. On failure, we must close the fd
  // ourselves to prevent a leak. Use manual check to ensure fd cleanup.
  cudaError_t cuda_err =
      cudaImportExternalSemaphore(&cuda_semaphore_, &cuda_sem_desc);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "cudaImportExternalSemaphore failed: "
               << cudaGetErrorString(cuda_err);
    close(fd);  // Close fd since CUDA didn't take ownership
    destroy();
    return false;
  }

  signal_value_.store(0);

  LOG(INFO) << "Created CUDA-Vulkan timeline semaphore";
  return true;
}

void CudaVulkanSemaphore::destroy() {
  // Destroy CUDA external semaphore with error checking (warning-level in
  // destructor)
  if (cuda_semaphore_) {
    cudaError_t err = cudaDestroyExternalSemaphore(cuda_semaphore_);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to destroy CUDA external semaphore: "
                   << cudaGetErrorString(err);
    }
    cuda_semaphore_ = nullptr;
  }

  if (ctx_ && ctx_->device() && semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(ctx_->device(), semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
  }

  ctx_ = nullptr;
  signal_value_.store(0);
}

bool CudaVulkanSemaphore::signalFromCuda(const CudaStream& stream,
                                         uint64_t* signaled_value) {
  if (!cuda_semaphore_) {
    LOG(ERROR) << "CudaVulkanSemaphore not initialized";
    return false;
  }

  // Atomically increment and get next value before signaling.
  // This ensures concurrent calls get unique monotonically increasing values.
  // Note: If signaling fails below, we have a "gap" in the timeline
  // (signal_value_ was incremented but no actual signal occurred). Timeline
  // semaphores handle gaps gracefully - waiting for value N succeeds once any
  // value >= N is signaled. This is preferable to trying to "undo" the
  // increment, which would be racy.
  uint64_t next_value = signal_value_.fetch_add(1) + 1;

  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = next_value;
  params.flags = 0;

  cudaStream_t cuda_stream = stream;
  cudaError_t err = cudaSignalExternalSemaphoresAsync(&cuda_semaphore_, &params,
                                                      1, cuda_stream);
  if (err != cudaSuccess) {
    LOG(WARNING) << "Failed to signal CUDA-Vulkan semaphore: "
                 << cudaGetErrorString(err);
    return false;
  }

  // Output the signaled value so caller can use it for waitFromCuda()
  if (signaled_value) {
    *signaled_value = next_value;
  }

  return true;
}

bool CudaVulkanSemaphore::waitFromCuda(const CudaStream& stream,
                                       uint64_t value) {
  if (!cuda_semaphore_) {
    LOG(ERROR) << "CudaVulkanSemaphore not initialized";
    return false;
  }

  // If value is 0, use current signal value. This can race with concurrent
  // signalFromCuda() calls - prefer passing the explicit value from
  // signalFromCuda() for correct synchronization.
  uint64_t wait_value = value;
  if (value == 0) {
    wait_value = signal_value_.load();
    VLOG(1) << "waitFromCuda called with value=0, using current signal value "
            << wait_value << ". Consider passing explicit value for safety.";
  }

  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = wait_value;
  params.flags = 0;

  cudaStream_t cuda_stream = stream;
  cudaError_t err = cudaWaitExternalSemaphoresAsync(&cuda_semaphore_, &params,
                                                    1, cuda_stream);
  if (err != cudaSuccess) {
    LOG(WARNING) << "Failed to wait on CUDA-Vulkan semaphore: "
                 << cudaGetErrorString(err);
    return false;
  }

  return true;
}

}  // namespace renderer
}  // namespace nvblox
