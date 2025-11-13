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
#pragma once

#include "nvblox/interpolation/interpolation_2d.h"
namespace nvblox {

template <typename SensorType>
__device__ inline bool projectThreadVoxel(
    const Index3D& block_idx, const Index3D& voxel_idx,
    const SensorType& sensor, const Transform& T_C_L, const float block_size,
    const float max_depth, Eigen::Vector2f* u_px_ptr, float* u_depth_ptr,
    Vector3f* p_voxel_center_C_ptr) {
  // Voxel center point
  const Vector3f p_voxel_center_L =
      getCenterPositionFromBlockIndexAndVoxelIndex(block_size, block_idx,
                                                   voxel_idx);
  // To sensor frame
  *p_voxel_center_C_ptr = T_C_L * p_voxel_center_L;

  // Project to image plane
  if (!sensor.project(*p_voxel_center_C_ptr, u_px_ptr)) {
    return false;
  }

  // Depth
  *u_depth_ptr = sensor.getDepth(*p_voxel_center_C_ptr);

  // Test for max depth
  if ((max_depth > 0.0f) && (*u_depth_ptr > max_depth)) {
    return false;
  }

  return true;
}

__device__ inline void voxelAndBlockIndexFromCudaThreadIndex(
    const Index3D* block_indices_device_ptr, Index3D* const block_idx,
    Index3D* voxel_idx) {
  *block_idx = block_indices_device_ptr[blockIdx.x];
  *voxel_idx = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
}

template <typename SensorType>
__device__ inline bool doesVoxelHaveDepthMeasurement(
    const Index3D& block_idx, const Index3D& voxel_idx,
    const SensorType& sensor, const DepthImageConstView image,
    const Transform T_C_L, const float voxel_size, const float block_size,
    const float max_integration_distance, const float truncation_distance_m) {
  // Project the voxel into the depth image
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_idx, voxel_idx, sensor, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return false;
  }
  // Interpolate on the image plane
  // This is done differently depending on the sensor type.
  float surface_depth_measured;
  if (!sensor.interpolateDepthImage(image, u_px, p_voxel_center_C, voxel_size,
                                    &surface_depth_measured)) {
    return false;
  }

  // Handle invalid depth values
  if (!interpolation::checkers::PixelIsValidDepth::check(
          surface_depth_measured)) {
    // Voxel is not in view for invalid depth values.
    return false;
  }

  // Check the distance from the surface
  const float voxel_to_surface_distance =
      surface_depth_measured - voxel_depth_m;
  // Check that we're not occluded (we're occluded if we're more than the
  // truncation distance behind a surface).
  if (voxel_to_surface_distance < -truncation_distance_m) {
    return false;
  }
  return true;
}

}  // namespace nvblox
