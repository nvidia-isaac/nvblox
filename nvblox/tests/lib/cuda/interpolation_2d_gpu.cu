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
#include "nvblox/tests/interpolation_2d_gpu.h"

#include "nvblox/interpolation/interpolation_2d.h"

namespace nvblox {
namespace test_utils {

template <typename ElementType>
__global__ void interpolate(const ElementType* image_unified_ptr, int rows,
                            int cols, const Vector2f* u_px_vec,
                            ElementType* values_ptr, int* success_flags,
                            int num_points) {
  const int lin_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (lin_idx < num_points) {
    ElementType value;
    if (interpolation::interpolate2DLinear(image_unified_ptr, u_px_vec[lin_idx],
                                           rows, cols, &value)) {
      success_flags[lin_idx] = 1;
      values_ptr[lin_idx] = value;
    } else {
      success_flags[lin_idx] = 0;
    }
  }
}

template <typename ElementType>
void linearInterpolateImageGpuTemplate(const Image<ElementType>& image,
                                       const std::vector<Vector2f>& u_px_vec,
                                       std::vector<ElementType>* values_ptr,
                                       std::vector<int>* success_flags_ptr) {
  CHECK_NOTNULL(values_ptr);
  CHECK_NOTNULL(success_flags_ptr);
  CHECK_EQ(values_ptr->size(), u_px_vec.size());
  CHECK_EQ(success_flags_ptr->size(), u_px_vec.size());
  // Move the data to the GPU
  const int num_points = u_px_vec.size();
  Vector2f* u_px_vec_device_ptr;
  ElementType* values_device_ptr;
  int* success_flags_device_ptr;
  checkCudaErrors(
      cudaMalloc(&u_px_vec_device_ptr, num_points * sizeof(Eigen::Vector2f)));
  checkCudaErrors(
      cudaMalloc(&values_device_ptr, num_points * sizeof(ElementType)));
  checkCudaErrors(
      cudaMalloc(&success_flags_device_ptr, num_points * sizeof(int)));
  checkCudaErrors(cudaMemcpy(u_px_vec_device_ptr, u_px_vec.data(),
                             num_points * sizeof(Eigen::Vector2f),
                             cudaMemcpyHostToDevice));

  // Interpolate on the GPUUUU
  constexpr int kThreadsPerBlock = 32;
  const int num_blocks = static_cast<int>(num_points / kThreadsPerBlock) + 1;
  interpolate<<<num_blocks, kThreadsPerBlock>>>(
      image.dataConstPtr(), image.rows(), image.cols(), u_px_vec_device_ptr,
      values_device_ptr, success_flags_device_ptr, num_points);
  checkCudaErrors(cudaPeekAtLastError());
  // Get results back
  checkCudaErrors(cudaMemcpy(values_ptr->data(), values_device_ptr,
                             num_points * sizeof(ElementType),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(success_flags_ptr->data(),
                             success_flags_device_ptr, num_points * sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(values_device_ptr));
  checkCudaErrors(cudaFree(success_flags_device_ptr));
}

void linearInterpolateImageGpu(const DepthImage& image,
                               const std::vector<Vector2f>& u_px_vec,
                               std::vector<float>* values_ptr,
                               std::vector<int>* success_flags_ptr) {
  linearInterpolateImageGpuTemplate<float>(image, u_px_vec, values_ptr,
                                           success_flags_ptr);
}

void linearInterpolateImageGpu(const ColorImage& image,
                               const std::vector<Vector2f>& u_px_vec,
                               std::vector<Color>* values_ptr,
                               std::vector<int>* success_flags_ptr) {
  linearInterpolateImageGpuTemplate<Color>(image, u_px_vec, values_ptr,
                                           success_flags_ptr);
}

}  //  namespace test_utils
}  //  namespace nvblox
