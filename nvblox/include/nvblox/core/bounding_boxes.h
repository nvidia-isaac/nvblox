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

#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"

namespace nvblox {

inline std::vector<Index3D> getBlockIndicesTouchedByBoundingBox(
    const float block_size, const AxisAlignedBoundingBox& aabb);

inline AxisAlignedBoundingBox getAABBOfBlock(const float block_size,
                                             const Index3D& block_index);

template <typename BlockType>
AxisAlignedBoundingBox getAABBOfAllocatedBlocks(
    const BlockLayer<BlockType>& layer);

template <typename BlockType>
std::vector<Index3D> getAllocatedBlocksWithinAABB(
    const BlockLayer<BlockType>& layer, const AxisAlignedBoundingBox& aabb);

AxisAlignedBoundingBox getAABBOfObservedVoxels(const EsdfLayer& layer);
AxisAlignedBoundingBox getAABBOfObservedVoxels(const TsdfLayer& layer,
                                               const float min_weight = 1e-4);
AxisAlignedBoundingBox getAABBOfObservedVoxels(const ColorLayer& layer,
                                               const float min_weight = 1e-4);

}  // namespace nvblox

#include "nvblox/core/impl/bounding_boxes_impl.h"
