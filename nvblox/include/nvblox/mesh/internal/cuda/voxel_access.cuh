/*
Copyright 2026 NVIDIA CORPORATION

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

#include <cuda_runtime.h>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/internal/appearance_getter.h"

namespace nvblox {

/// Read a TSDF voxel at a local coordinate that may cross block boundaries.
///
/// Given a voxel coordinate (vx, vy, vz) relative to @p block, looks up the
/// SDF value and world position. If the coordinate falls outside the block,
/// the neighbor block is fetched from @p tsdf_hash.
///
/// @return false if the neighbor block is missing or the voxel weight is below
///         @p min_weight.
__device__ inline bool getTsdfVoxelAtLocalCoord(
    const TsdfBlock* block, const Index3D& block_index, int vx, int vy, int vz,
    const Index3DDeviceHashMapType<TsdfBlock>& tsdf_hash, float min_weight,
    float* sdf_out, Vector3f* position_out, float block_size) {
  constexpr int kVPS = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;

  const bool need_neighbor =
      (vx < 0 || vx >= kVPS || vy < 0 || vy >= kVPS || vz < 0 || vz >= kVPS);

  const TsdfBlock* block_to_use = block;
  int local_x = vx, local_y = vy, local_z = vz;
  Index3D actual_block_index = block_index;

  if (need_neighbor) {
    Index3D offset(vx < 0 ? -1 : (vx >= kVPS ? 1 : 0),
                   vy < 0 ? -1 : (vy >= kVPS ? 1 : 0),
                   vz < 0 ? -1 : (vz >= kVPS ? 1 : 0));
    actual_block_index = block_index + offset;

    block_to_use = getBlockPtr<TsdfVoxel>(tsdf_hash, actual_block_index);
    if (block_to_use == nullptr) {
      return false;
    }

    local_x = (vx < 0) ? (kVPS + vx) : (vx >= kVPS ? (vx - kVPS) : vx);
    local_y = (vy < 0) ? (kVPS + vy) : (vy >= kVPS ? (vy - kVPS) : vy);
    local_z = (vz < 0) ? (kVPS + vz) : (vz >= kVPS ? (vz - kVPS) : vz);
  }

  const TsdfVoxel& voxel = block_to_use->voxels[local_x][local_y][local_z];

  if (voxel.weight < min_weight) {
    return false;
  }

  *sdf_out = voxel.distance;
  *position_out = getCenterPositionFromBlockIndexAndVoxelIndex(
      block_size, actual_block_index, Index3D(local_x, local_y, local_z));

  return true;
}

/// Sample appearance from a layer GPU hash at a given world position.
///
/// Converts @p position to block+voxel indices, looks up the block in the
/// hash, and returns the appearance value. Returns the default appearance if
/// the block is missing or the voxel weight is zero.
template <typename AppearanceVoxelType>
__device__ inline typename AppearanceVoxelType::ArrayType
getAppearanceAtPosition(const Index3DDeviceHashMapType<
                            VoxelBlock<AppearanceVoxelType>>& appearance_hash,
                        const Vector3f& position, float block_size) {
  Index3D block_idx, voxel_idx;
  getBlockAndVoxelIndexFromPositionInLayer(block_size, position, &block_idx,
                                           &voxel_idx);

  const auto* appearance_block =
      getBlockPtr<AppearanceVoxelType>(appearance_hash, block_idx);
  if (appearance_block == nullptr) {
    return AppearanceGetter<AppearanceVoxelType>::getDefaultAppearance();
  }

  const AppearanceVoxelType& appearance_voxel =
      appearance_block->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];

  if (static_cast<float>(appearance_voxel.weight) <= 0.0f) {
    return AppearanceGetter<AppearanceVoxelType>::getDefaultAppearance();
  }

  return AppearanceGetter<AppearanceVoxelType>::getAppearance(appearance_voxel);
}

}  // namespace nvblox
