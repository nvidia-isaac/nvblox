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
#include "nvblox/tests/gpu_image_routines.h"

#include "nvblox/core/cuda/error_check.cuh"
#include "nvblox/core/image.h"

namespace nvblox {
namespace test_utils {

template <typename ElementType>
__global__ void setImageConstantKernel(ElementType* image, const int rows,
                                       const int cols, ElementType value) {
  // NOTE(alexmillane): Memory access is fully coallesed because neighbouring
  // threads in the grid x dimension, access neighbouring memory elements
  // (row-major images).
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx < cols && row_idx < rows) {
    image::access(row_idx, col_idx, cols, image) = value;
  }
}

template <typename ElementType>
void setImageConstantOnGpuTemplate(const ElementType value,
                                   Image<ElementType>* image_ptr) {
  // Set the pixels to a constant value. One thread per pixel (lol)
  constexpr int kThreadsPerBlockInEachDimension = 8;
  dim3 blockShape(kThreadsPerBlockInEachDimension,
                  kThreadsPerBlockInEachDimension);
  dim3 gridShape((image_ptr->rows() / kThreadsPerBlockInEachDimension) + 1,
                 (image_ptr->cols() / kThreadsPerBlockInEachDimension) + 1);
  setImageConstantKernel<<<gridShape, blockShape>>>(
      image_ptr->dataPtr(), image_ptr->rows(), image_ptr->cols(), value);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void setImageConstantOnGpu(const float value, DepthImage* image_ptr) {
  setImageConstantOnGpuTemplate<float>(value, image_ptr);
}

void setImageConstantOnGpu(const Color value, ColorImage* image_ptr) {
  setImageConstantOnGpuTemplate<Color>(value, image_ptr);
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

template <typename ImageType>
void getDifferenceImageTemplate(const ImageType& image_1,
                                const ImageType& image_2,
                                ImageType* diff_image_ptr) {
  CHECK_EQ(image_1.rows(), image_2.rows());
  CHECK_EQ(image_1.cols(), image_2.cols());
  CHECK(image_1.memory_type() == image_2.memory_type());
  CHECK_NOTNULL(diff_image_ptr);
  // Create the image for output
  *diff_image_ptr =
      ImageType(image_1.rows(), image_1.cols(), image_1.memory_type());
  // Set the pixels to a constant value. One thread per pixel (lol)
  constexpr int kThreadsPerBlockInEachDimension = 8;
  dim3 blockShape(kThreadsPerBlockInEachDimension,
                  kThreadsPerBlockInEachDimension);
  dim3 gridShape((image_1.rows() / kThreadsPerBlockInEachDimension) + 1,
                 (image_1.cols() / kThreadsPerBlockInEachDimension) + 1);
  differenceImageKernel<<<gridShape, blockShape>>>(
      diff_image_ptr->dataPtr(), image_1.rows(), image_1.cols(),
      image_1.dataConstPtr(), image_2.dataConstPtr());
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void getDifferenceImageOnGPU(const ColorImage& image_1, const ColorImage& image_2,
                        ColorImage* diff_image_ptr) {
  getDifferenceImageTemplate<ColorImage>(image_1, image_2, diff_image_ptr);
}

void getDifferenceImageOnGPU(const DepthImage& image_1, const DepthImage& image_2,
                        DepthImage* diff_image_ptr) {
  getDifferenceImageTemplate<DepthImage>(image_1, image_2, diff_image_ptr);
}

}  // namespace test_utils
}  // namespace nvblox
