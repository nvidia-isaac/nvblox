/*
Copyright 2025 NVIDIA CORPORATION

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
#include "nvblox/core/internal/error_check.h"
#include "nvblox/tests/increment_on_gpu.h"
#include "nvblox/utils/cuda_kernel_utils.h"

__global__ void incrementKernel(int* number) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    (*number)++;
  }
}

__global__ void incrementKernel(int* number, const int num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    number[idx]++;
  }
}

namespace nvblox {
namespace test_utils {

void incrementOnStream(int* number_ptr, CudaStream* stream_ptr) {
  incrementKernel<<<1, 1, 0, *stream_ptr>>>(number_ptr);
  stream_ptr->synchronize();
}

void incrementOnGPU(int* number_ptr) {
  incrementKernel<<<1, 1>>>(number_ptr);
  checkCudaErrors(cudaDeviceSynchronize());
}

void incrementOnGPU(const int num_elements, int* numbers_ptr) {
  constexpr int kThreadsPerBlock = 32;
  const int num_blocks = divideRoundUp(num_elements, kThreadsPerBlock);
  incrementKernel<<<num_blocks, kThreadsPerBlock>>>(numbers_ptr, num_elements);
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace test_utils
}  // namespace nvblox
