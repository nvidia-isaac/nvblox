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

#include "nvblox/map/accessors.h"

namespace nvblox {

inline std::vector<Index3D> getBlockIndicesTouchedByBoundingBox(
    const float block_size, const AxisAlignedBoundingBox& aabb_L) {
  // Get bounds (in voxels)
  const Index3D aabb_min_vox_L =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.min());
  const Index3D aabb_max_vox_L =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.max());
  const Index3D num_blocks = aabb_max_vox_L - aabb_min_vox_L;
  // Fill up the vector
  std::vector<Index3D> block_indices;
  block_indices.reserve(num_blocks.x() * num_blocks.y() * num_blocks.z());
  Index3D current_idx = aabb_min_vox_L;
  for (int x_ind = 0; x_ind <= num_blocks.x(); x_ind++) {
    for (int y_ind = 0; y_ind <= num_blocks.y(); y_ind++) {
      for (int z_ind = 0; z_ind <= num_blocks.z(); z_ind++) {
        block_indices.push_back(current_idx);
        current_idx.z()++;
      }
      current_idx.z() -= (num_blocks.z() + 1);  // reset z
      current_idx.y()++;
    }
    current_idx.y() -= (num_blocks.y() + 1);  // reset y
    current_idx.x()++;
  }
  return block_indices;
}

inline AxisAlignedBoundingBox getAABBOfBlock(const float block_size,
                                             const Index3D& block_index) {
  return AxisAlignedBoundingBox(
      block_index.cast<float>() * block_size,
      (block_index.cast<float>() + Vector3f(1.0f, 1.0f, 1.0f)) * block_size);
}

template <typename BlockType>
AxisAlignedBoundingBox getAABBOfAllocatedBlocks(
    const BlockLayer<BlockType>& layer) {
  return getAABBOfBlocks(layer.block_size(), layer.getAllBlockIndices());
}

template <typename BlockType>
std::vector<Index3D> getAllocatedBlocksWithinAABB(
    const BlockLayer<BlockType>& layer, const AxisAlignedBoundingBox& aabb) {
  std::vector<Index3D> allocated_blocks;
  for (const Index3D& idx : layer.getAllBlockIndices()) {
    if (aabb.intersects(getAABBOfBlock(layer.block_size(), idx))) {
      allocated_blocks.push_back(idx);
    }
  }
  return allocated_blocks;
}

}  // namespace nvblox
