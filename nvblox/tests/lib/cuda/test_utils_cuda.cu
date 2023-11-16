/*
Copyright 2022 NVIDIA CORPORATION

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
#include "nvblox/tests/test_utils_cuda.h"

namespace nvblox {
namespace test_utils {

/* Fills a device vector with elements
 * Call with
 * - GridDim: 1 block, 1D
 * - BlockDim: 1D size > num_elements (1 thread per element)
 */
template <typename T>
__global__ void fillVectorWithConstant(int num_elements, T value, T* vec) {
  if ((blockIdx.x == 0) && (threadIdx.x < num_elements)) {
    vec[threadIdx.x] = value;
  }
}

/* Fills a device vector with elements
 * Call with
 * - GridDim: 1D
 * - BlockDim: 1D (1 thread per element)
 */
template <typename T>
__global__ void checkVectorAllConstant(int num_elements, T value, const T* vec,
                                       bool* flag) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx == 0) {
    *flag = true;
  }
  __syncthreads();
  if (thread_idx < num_elements) {
    if (vec[thread_idx] != value) {
      *flag = false;
    }
  }
}

/* Fills a device vector with elements
 * Call with
 * - GridDim: N blocks, 1D
 * - BlockDim: 1D (1 thread per element)
 */
__global__ void addOneToAll(int num_elements, int* vec) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] + 1;
  }
}

void fillVectorWithConstant(float value, unified_vector<float>* vec_ptr) {
  constexpr int kMaxThreadBlockSize = 512;
  CHECK_LT(vec_ptr->size(), kMaxThreadBlockSize);
  // kernel
  int num_thread_blocks = 1;
  int num_threads = vec_ptr->size();
  fillVectorWithConstant<<<num_thread_blocks, num_threads>>>(
      vec_ptr->size(), value, vec_ptr->data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

void fillVectorWithConstant(int value, unified_vector<int>* vec_ptr) {
  constexpr int kMaxThreadBlockSize = 512;
  CHECK_LT(vec_ptr->size(), kMaxThreadBlockSize);
  // kernel
  int num_thread_blocks = 1;
  int num_threads = vec_ptr->size();
  fillVectorWithConstant<<<num_thread_blocks, num_threads>>>(
      vec_ptr->size(), value, vec_ptr->data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

void fillWithConstant(float value, size_t num_elems, float* vec_ptr) {
  constexpr int kMaxThreadBlockSize = 512;
  CHECK_LT(num_elems, kMaxThreadBlockSize);
  // kernel
  int num_thread_blocks = 1;
  int num_threads = num_elems;
  fillVectorWithConstant<<<num_thread_blocks, num_threads>>>(num_elems, value,
                                                             vec_ptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

void fillWithConstant(int value, size_t num_elems, int* vec_ptr) {
  constexpr int kMaxThreadBlockSize = 512;
  CHECK_LT(num_elems, kMaxThreadBlockSize);
  // kernel
  int num_thread_blocks = 1;
  int num_threads = num_elems;
  fillVectorWithConstant<<<num_thread_blocks, num_threads>>>(num_elems, value,
                                                             vec_ptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename T>
bool checkVectorAllConstantTemplate(const T* vec, T value, size_t size) {
  // Allocate memory
  bool* flag_device_ptr;
  checkCudaErrors(cudaMalloc(&flag_device_ptr, sizeof(bool)));

  // Kernel
  constexpr int kNumThreads = 512;
  const int num_thread_blocks = size / kNumThreads + 1;
  checkVectorAllConstant<T>
      <<<num_thread_blocks, kNumThreads>>>(size, value, vec, flag_device_ptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // Copy the flag back
  bool flag;
  checkCudaErrors(
      cudaMemcpy(&flag, flag_device_ptr, sizeof(bool), cudaMemcpyDeviceToHost));

  // Free the flag
  checkCudaErrors(cudaFree(flag_device_ptr));

  return flag;
}

bool checkVectorAllConstant(const unified_vector<float>& vec, float value) {
  return checkVectorAllConstantTemplate<float>(vec.data(), value, vec.size());
}

bool checkVectorAllConstant(const unified_vector<int>& vec, int value) {
  return checkVectorAllConstantTemplate<int>(vec.data(), value, vec.size());
}

bool checkAllConstant(const float* vec_ptr, float value, size_t num_elems) {
  return checkVectorAllConstantTemplate<float>(vec_ptr, value, num_elems);
}

bool checkAllConstant(const int* vec_ptr, int value, size_t num_elems) {
  return checkVectorAllConstantTemplate<int>(vec_ptr, value, num_elems);
}

void addOneToAllGPU(unified_vector<int>* vec_ptr) {
  constexpr int kNumThreadsPerBlock = 512;
  const int kNumBlocks = vec_ptr->size() / kNumThreadsPerBlock + 1;
  addOneToAll<<<kNumBlocks, kNumThreadsPerBlock>>>(vec_ptr->size(),
                                                   vec_ptr->data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace test_utils
}  // namespace nvblox
