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
#include "nvblox/core/image.h"

#include <glog/logging.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

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

struct max_with_constant_functor {
  const float constant_;

  max_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return fmaxf(pixel_value, constant_);
  }
};

struct min_with_constant_functor {
  const float constant_;

  min_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return fminf(pixel_value, constant_);
  }
};

void elementWiseMinInPlace(const int rows, const int cols, const float constant,
                           float* image) {
  thrust::device_ptr<float> dev_ptr(image);
  thrust::transform(thrust::device, dev_ptr, dev_ptr + (rows * cols), dev_ptr,
                    min_with_constant_functor(constant));
}

void elementWiseMaxInPlace(const int rows, const int cols, const float constant,
                           float* image) {
  thrust::device_ptr<float> dev_ptr(image);
  thrust::transform(thrust::device, dev_ptr, dev_ptr + (rows * cols), dev_ptr,
                    max_with_constant_functor(constant));
}

__device__ Color diff(const Color& color_1, const Color& color_2) {
  return Color(static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.r) -
                                             static_cast<int16_t>(color_2.r))),
               static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.g) -
                                             static_cast<int16_t>(color_2.g))),
               static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.b) -
                                             static_cast<int16_t>(color_2.b))));
}

__device__ float diff(const float& depth_1, const float& depth_2) {
  return fabsf(depth_1 - depth_2);
}

template <typename ElementType>
__global__ void differenceImageKernel(ElementType* diff_image_ptr,
                                      const int rows, const int cols,
                                      const ElementType* image_1,
                                      const ElementType* image_2) {
  // NOTE(alexmillane): Memory access is fully coallesed because neighbouring
  // threads in the grid x dimension, access neighbouring memory elements
  // (row-major images).
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx < cols && row_idx < rows) {
    const ElementType color_1 = image::access(row_idx, col_idx, cols, image_1);
    const ElementType color_2 = image::access(row_idx, col_idx, cols, image_2);
    const ElementType abs_color_diff = diff(color_1, color_2);
    image::access(row_idx, col_idx, cols, diff_image_ptr) = abs_color_diff;
  }
}

template <typename ElementType>
void getDifferenceImageTemplate(const int rows, const int cols,
                                const ElementType* image_1,
                                const ElementType* image_2,
                                ElementType* diff_image_ptr) {
  CHECK_NOTNULL(diff_image_ptr);
  // Set the pixels to a constant value. One thread per pixel (lol)
  constexpr int kThreadsPerBlockInEachDimension = 8;
  dim3 blockShape(kThreadsPerBlockInEachDimension,
                  kThreadsPerBlockInEachDimension);
  dim3 gridShape((rows / kThreadsPerBlockInEachDimension) + 1,
                 (cols / kThreadsPerBlockInEachDimension) + 1);
  differenceImageKernel<<<gridShape, blockShape>>>(diff_image_ptr, rows, cols,
                                                   image_1, image_2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void diff(const int rows, const int cols, const Color* image_1,
          const Color* image_2, Color* diff_image_ptr) {
  getDifferenceImageTemplate(rows, cols, image_1, image_2, diff_image_ptr);
}

void diff(const int rows, const int cols, const float* image_1,
          const float* image_2, float* diff_image_ptr) {
  getDifferenceImageTemplate(rows, cols, image_1, image_2, diff_image_ptr);
}

}  // namespace cuda
}  // namespace nvblox
