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

#include "nvblox/core/internal/error_check.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {
namespace test_utils {

template <typename ElementType>
__global__ void setImageConstantKernel(ImageView<ElementType> image_view,
                                       ElementType value) {
  // NOTE(alexmillane): Memory access is fully coallesed because neighbouring
  // threads in the grid x dimension, access neighbouring memory elements
  // (row-major images).
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx < image_view.cols() && row_idx < image_view.rows()) {
    for (int c = 0; c < image_view.num_elements_per_pixel(); ++c) {
      image_view(row_idx, col_idx, c) = value;
    }
  }
}

template <typename ElementType>
void setImageConstantOnGpuTemplate(const ElementType value,
                                   Image<ElementType>* image_ptr) {
  // Set the pixels to a constant value. One thread per pixel (lol)
  constexpr int kThreadsPerBlockInEachDimension = 8;
  dim3 blockShape(kThreadsPerBlockInEachDimension,
                  kThreadsPerBlockInEachDimension);
  dim3 gridShape(
      divideRoundUp(image_ptr->rows(), kThreadsPerBlockInEachDimension),
      divideRoundUp(image_ptr->cols(), kThreadsPerBlockInEachDimension));
  setImageConstantKernel<<<gridShape, blockShape>>>(
      ImageView<ElementType>(*image_ptr), value);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void setImageConstantOnGpu(const float value, DepthImage* image_ptr) {
  setImageConstantOnGpuTemplate<float>(value, image_ptr);
}

void setImageConstantOnGpu(const Color value, ColorImage* image_ptr) {
  setImageConstantOnGpuTemplate<Color>(value, image_ptr);
}

}  // namespace test_utils
}  // namespace nvblox
