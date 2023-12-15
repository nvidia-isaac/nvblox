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

#include <thrust/pair.h>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/voxels.h"

#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"

namespace nvblox {

template <typename VoxelType>
__device__ bool getVoxelPtr(
    const Index3DDeviceHashMapType<VoxelBlock<VoxelType>>& block_hash,
    const Index3D& block_index, const Index3D& voxel_index,
    VoxelType** voxel_ptr) {
  auto it = block_hash.find(block_index);
  if (it != block_hash.end()) {
    *voxel_ptr =
        &it->second->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
    return true;
  }
  return false;
}

template <typename VoxelType>
__device__ VoxelBlock<VoxelType>* getBlockPtr(
    const Index3DDeviceHashMapType<VoxelBlock<VoxelType>>& block_hash,
    const Index3D& block_index) {
  auto it = block_hash.find(block_index);
  if (it != block_hash.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

template <typename VoxelType>
__device__ inline bool getVoxelAtPosition(
    const Index3DDeviceHashMapType<VoxelBlock<VoxelType>>& block_hash,
    const Vector3f& p_L, float block_size, VoxelType** voxel_ptr) {
  Index3D block_idx;
  Index3D voxel_idx;
  getBlockAndVoxelIndexFromPositionInLayer(block_size, p_L, &block_idx,
                                           &voxel_idx);
  return getVoxelPtr(block_hash, block_idx, voxel_idx, voxel_ptr);
}

}  // namespace nvblox