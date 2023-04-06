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

#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"

namespace nvblox {
/// Get all of the blocks that are touched by an AABB, no matter how little.
/// @param block_size Metric size of the block.
/// @param aabb Axis-Aligned Bounding Box that does the touching.
/// @return Vector of all the touched block indices.
inline std::vector<Index3D> getBlockIndicesTouchedByBoundingBox(
    const float block_size, const AxisAlignedBoundingBox& aabb);

/// Gets the Axis-Aligned Bounding Box of a block.
/// @param block_size Metric size of the block.
/// @param block_index The index of the block.
/// @return The AABB.
inline AxisAlignedBoundingBox getAABBOfBlock(const float block_size,
                                             const Index3D& block_index);

/// Get AABB that covers ALL blocks in the block index list.
AxisAlignedBoundingBox getAABBOfBlocks(const float block_size,
                                       const std::vector<Index3D>& blocks);

/// Get the outer AABB of all of the allocated blocks in the layer.
template <typename BlockType>
AxisAlignedBoundingBox getAABBOfAllocatedBlocks(
    const BlockLayer<BlockType>& layer);

/// Get all of the allocated blocks that are within a given AABB.
template <typename BlockType>
std::vector<Index3D> getAllocatedBlocksWithinAABB(
    const BlockLayer<BlockType>& layer, const AxisAlignedBoundingBox& aabb);

/// Get the outer AABB of all of the blocks that contain observed voxels in an
/// ESDF layer.
AxisAlignedBoundingBox getAABBOfObservedVoxels(const EsdfLayer& layer);
/// Get the outer AABB of all of the blocks that contain observed voxels in an
/// TSDF layer.
AxisAlignedBoundingBox getAABBOfObservedVoxels(const TsdfLayer& layer,
                                               const float min_weight = 1e-4);
/// Get the outer AABB of all of the blocks that contain observed voxels in an
/// Color layer.
AxisAlignedBoundingBox getAABBOfObservedVoxels(const ColorLayer& layer,
                                               const float min_weight = 1e-4);

}  // namespace nvblox

#include "nvblox/geometry/internal/impl/bounding_boxes_impl.h"
