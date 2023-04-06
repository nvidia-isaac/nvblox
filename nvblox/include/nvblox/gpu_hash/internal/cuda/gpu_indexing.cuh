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
__device__ inline bool getVoxelAtPosition(
    Index3DDeviceHashMapType<VoxelBlock<VoxelType>> block_hash,
    const Vector3f& p_L, float block_size, VoxelType** voxel_ptr) {
  Index3D block_idx;
  Index3D voxel_idx;
  getBlockAndVoxelIndexFromPositionInLayer(block_size, p_L, &block_idx,
                                           &voxel_idx);
  auto it = block_hash.find(block_idx);
  if (it != block_hash.end()) {
    *voxel_ptr =
        &it->second->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
    return true;
  }
  return false;
}

}  // namespace nvblox