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

#include "nvblox/core/blox.h"
#include "nvblox/core/layer.h"

namespace nvblox {

// Convert a list of BlockIndices on host, to a list of device pointers.
template <typename BlockType>
__host__ std::vector<BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    BlockLayer<BlockType>* layer_ptr);

// Const version
template <typename BlockType>
__host__ std::vector<const BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    const BlockLayer<BlockType>& layer_ptr);

// Allocates blocks in the block_indices list which are not already allocated.
template <typename VoxelType>
void allocateBlocksWhereRequired(const std::vector<Index3D>& block_indices,
                                 BlockLayer<VoxelBlock<VoxelType>>* layer);

}  // namespace nvblox

#include "nvblox/integrators/impl/integrators_common_impl.h"
