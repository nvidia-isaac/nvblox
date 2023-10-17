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

namespace nvblox {

template <typename SensorType>
__device__ inline bool projectThreadVoxel(
    const Index3D* block_indices_device_ptr, const SensorType& sensor,
    const Transform& T_C_L, const float block_size, const float max_depth,
    Eigen::Vector2f* u_px_ptr, float* u_depth_ptr,
    Vector3f* p_voxel_center_C_ptr) {
  // The indices of the voxel this thread will work on
  // block_indices_device_ptr[blockIdx.x]:
  //                 - The index of the block we're working on (blockIdx.y/z
  //                   should be zero)
  // threadIdx.x/y/z - The indices of the voxel within the block (we
  //                   expect the threadBlockDims == voxelBlockDims)
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

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

}  // namespace nvblox
