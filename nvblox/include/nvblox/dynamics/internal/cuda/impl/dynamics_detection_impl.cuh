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

#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/utils/cuda_kernel_utils.h"

#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/sensors/sensor.h"

namespace nvblox {

__device__ inline Color getOverlayColor(const bool is_dynamic,
                                        const float depth) {
  constexpr float max_display_depth_m = 10.f;
  constexpr float depth_scale_factor = 255.0f / max_display_depth_m;
  const uint8_t scaled_depth = fmin(depth_scale_factor * depth, 255u);

  // Dynamics shown in red and rest greyish scaled depending on depth
  return Color(is_dynamic * 255u, scaled_depth, scaled_depth);
}

template <typename SensorType>
__global__ void findDynamicPointsKernel(
    const float* depth_frame_C,
    const Index3DDeviceHashMapType<FreespaceBlock> block_hash, float block_size,
    const Transform T_L_C, const SensorType sensor, const int rows,
    const int cols, int* dynamic_points_counter, Vector3f* dynamic_points,
    uint8_t* dynamic_mask_image, Color* dynamic_overlay_image) {
  // Each thread does a single pixel on the depth image.
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // Set pixel default values on output images.
  image::access(row_idx, col_idx, cols, dynamic_mask_image) = 0;
  image::access(row_idx, col_idx, cols, dynamic_overlay_image) = Color::White();

  // Get depth value.
  const float depth = image::access(row_idx, col_idx, cols, depth_frame_C);
  if (depth <= 0.0f) {
    return;  // Depth pixel invalid
  }

  // Get 3D point in the freespace layer frame.
  const Vector3f point_C =
      sensor.unprojectFromPixelIndices(Index2D(col_idx, row_idx), depth);
  const Vector3f point_L = T_L_C * point_C;

  // Get the corresponding voxel.
  FreespaceVoxel* freespace_voxel;
  if (!getVoxelAtPosition<FreespaceVoxel>(block_hash, point_L, block_size,
                                          &freespace_voxel)) {
    return;  // Voxel not found.
  }

  // If a projected depth pixel falls into a high confidence freespace voxel we
  // assume it must be dynamic.
  const bool is_dynamic = freespace_voxel->is_high_confidence_freespace;

  // Store dynamic points.
  if (is_dynamic) {
    int current_idx = atomicAdd(dynamic_points_counter, 1);
    dynamic_points[current_idx] = point_L;
  }

  // Update mask and overlay image
  image::access(row_idx, col_idx, cols, dynamic_mask_image) =
      is_dynamic * image::kMaskedValue;
  image::access(row_idx, col_idx, cols, dynamic_overlay_image) =
      getOverlayColor(is_dynamic, depth);
}

template <typename SensorType>
void DynamicsDetection::computeDynamics(const DepthImage& depth_frame_C,
                                        const FreespaceLayer& freespace_layer_L,
                                        const SensorType& sensor,
                                        const Transform& T_L_C) {
  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

  const int rows = depth_frame_C.rows();
  const int cols = depth_frame_C.cols();
  prepareOutputs(depth_frame_C);

  // Kernel call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(divideRoundUp(cols, kThreadsPerThreadBlock.x),
                        divideRoundUp(rows, kThreadsPerThreadBlock.y), 1);
  findDynamicPointsKernel<SensorType>
      <<<num_blocks, kThreadsPerThreadBlock, 0,
         *cuda_stream_>>>(depth_frame_C.dataConstPtr(),  // NOLINT
                          freespace_layer_L.getGpuLayerView(*cuda_stream_)
                              .getHash()
                              .impl_,                            // NOLINT
                          freespace_layer_L.block_size(),        // NOLINT
                          T_L_C,                                 // NOLINT
                          sensor,                                // NOLINT
                          rows,                                  // NOLINT
                          cols,                                  // NOLINT
                          dynamic_points_counter_device_.get(),  // NOLINT
                          dynamic_points_device_.data(),         // NOLINT
                          dynamics_mask_.dataPtr(),              // NOLINT
                          dynamics_overlay_.dataPtr());          // NOLINT
  dynamic_points_counter_device_.copyToAsync(dynamic_points_counter_host_,
                                             *cuda_stream_);
  cuda_stream_->synchronize();
  dynamic_points_device_.resizeAsync(*dynamic_points_counter_host_,
                                     *cuda_stream_);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox
