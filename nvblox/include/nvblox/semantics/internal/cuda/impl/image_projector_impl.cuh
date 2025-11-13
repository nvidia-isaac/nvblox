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
#pragma once

#include "nvblox/semantics/image_projector.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {

template <typename SensorType>
__global__ void projectImageKernel(const SensorType sensor, const float* image,
                                   const int rows, const int cols,
                                   const float max_back_projection_distance_m,
                                   Vector3f* pointcloud, int* pointcloud_size) {
  // Each thread does a single pixel
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  float depth = image::access(row_idx, col_idx, cols, image);
  if (depth <= 0.0f || depth > max_back_projection_distance_m) {
    return;
  }

  Index2D u_C(col_idx, row_idx);

  // Unproject from the image.
  const Vector3f p_C = sensor.unprojectFromPixelIndices(u_C, depth);

  // Insert into the pointcloud.
  pointcloud[atomicAdd(pointcloud_size, 1)] = p_C;
}

template <typename SensorType>
void DepthImageBackProjector::backProjectOnGPU(
    const DepthImage& image, const SensorType& sensor,
    Pointcloud* pointcloud_C_ptr, const float max_back_projection_distance_m) {
  CHECK_NOTNULL(pointcloud_C_ptr);
  CHECK(pointcloud_C_ptr->memory_type() == MemoryType::kDevice ||
        pointcloud_C_ptr->memory_type() == MemoryType::kUnified);

  // Create the max number of output points.
  pointcloud_C_ptr->resizeAsync(image.numel(), *cuda_stream_);

  // Reset the counter.
  if (pointcloud_size_device_ == nullptr || pointcloud_size_host_ == nullptr) {
    pointcloud_size_device_ = make_unified<int>(MemoryType::kDevice);
    pointcloud_size_host_ = make_unified<int>(MemoryType::kHost);
  }
  pointcloud_size_device_.setZero();

  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(divideRoundUp(image.cols(), kThreadsPerThreadBlock.x),
                        divideRoundUp(image.rows(), kThreadsPerThreadBlock.y),
                        1);
  projectImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0, *cuda_stream_>>>(
      sensor, image.dataConstPtr(), image.rows(), image.cols(),
      max_back_projection_distance_m, pointcloud_C_ptr->dataPtr(),
      pointcloud_size_device_.get());
  checkCudaErrors(cudaPeekAtLastError());

  pointcloud_size_device_.copyToAsync(pointcloud_size_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  pointcloud_C_ptr->resize(*pointcloud_size_host_);
}

}  // namespace nvblox
