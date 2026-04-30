/*
Copyright 2023 NVIDIA CORPORATION

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

#include <cuda_runtime.h>

#include "nvblox/core/internal/error_check.h"

// Runs a dummy memcpy operation on the default stream. Can be used
// together with nsys to obtain the ID of the default stream.
//
// TODO(dtingdahl) use dedicated function for getting the stream ID
// once we've upgraded to cuda 12.X
int main() {
  constexpr size_t kSize = 1000;
  char *ptr1, *ptr2;
  checkCudaErrors(cudaMalloc(&ptr1, kSize));
  checkCudaErrors(cudaMalloc(&ptr2, kSize));
  checkCudaErrors(
      cudaMemcpyAsync(ptr1, ptr1, kSize, cudaMemcpyDeviceToDevice, 0));
  checkCudaErrors(cudaFree(ptr1));
  checkCudaErrors(cudaFree(ptr2));
}
