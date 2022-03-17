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
#include "nvblox/core/cuda/image_cuda.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "nvblox/core/cuda/error_check.cuh"

namespace nvblox {
namespace cuda {

float max(const int rows, const int cols, const float* image) {
  const thrust::device_ptr<const float> dev_ptr(image);
  const thrust::device_ptr<const float> max_elem =
      thrust::max_element(thrust::device, dev_ptr, dev_ptr + (rows * cols));
  return *max_elem;
}

float min(const int rows, const int cols, const float* image) {
  const thrust::device_ptr<const float> dev_ptr(image);
  const thrust::device_ptr<const float> min_elem =
      thrust::min_element(thrust::device, dev_ptr, dev_ptr + (rows * cols));
  return *min_elem;
}

std::pair<float, float> minmax(const int rows, const int cols,
                               const float* image) {
  // Wrap our memory and reduce using thrust
  const thrust::device_ptr<const float> dev_ptr(image);
  const auto minmax_elem =
      thrust::minmax_element(thrust::device, dev_ptr, dev_ptr + (rows * cols));
  return {*minmax_elem.first, *minmax_elem.second};
}

}  // namespace cuda
}  // namespace nvblox
