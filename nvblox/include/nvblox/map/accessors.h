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

#include <functional>

#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

/// Get Block
template <typename BlockType>
bool getBlockAtPosition(const BlockLayer<BlockType>& layer,
                        const Eigen::Vector3f& position,
                        const BlockType** data);

/// Get Voxel
template <typename VoxelType>
bool getVoxelAtPosition(const BlockLayer<VoxelBlock<VoxelType>>& layer,
                        const Eigen::Vector3f& position,
                        const VoxelType** data);

/// Accessors for voxelized types
template <typename VoxelType>
const VoxelType* getVoxelAtBlockAndVoxelIndex(
    const BlockLayer<VoxelBlock<VoxelType>>& layer, const Index3D& block_index,
    const Index3D& voxel_index);

/// Accessors for calling a function on all voxels in a layer (const).
template <typename VoxelType>
using ConstVoxelCallbackFunction =
    std::function<void(const Index3D& block_index, const Index3D& voxel_index,
                       const VoxelType* voxel)>;

/// Accessors for calling a function on all voxels in a layer (non-const).
template <typename VoxelType>
using VoxelCallbackFunction = std::function<void(
    const Index3D& block_index, const Index3D& voxel_index, VoxelType* voxel)>;

/// Call function on all voxels in a layer (const).
template <typename VoxelType>
void callFunctionOnAllVoxels(const BlockLayer<VoxelBlock<VoxelType>>& layer,
                             ConstVoxelCallbackFunction<VoxelType> callback);

/// Call function on all voxels in a layer (non-const).
template <typename VoxelType>
void callFunctionOnAllVoxels(BlockLayer<VoxelBlock<VoxelType>>* layer,
                             VoxelCallbackFunction<VoxelType> callback);

/// Accessors for calling a function on all voxels in a block (const).
template <typename VoxelType>
using ConstBlockCallbackFunction =
    std::function<void(const Index3D& voxel_index, const VoxelType* voxel)>;

/// Accessors for calling a function on all voxels in a block (non-const).
template <typename VoxelType>
using BlockCallbackFunction =
    std::function<void(const Index3D& voxel_index, VoxelType* voxel)>;

template <typename VoxelType>
void callFunctionOnAllVoxels(const VoxelBlock<VoxelType>& block,
                             ConstBlockCallbackFunction<VoxelType> callback);

template <typename VoxelType>
void callFunctionOnAllVoxels(VoxelBlock<VoxelType>* block,
                             BlockCallbackFunction<VoxelType> callback);

/// Accessors for calling a function on all blocks in a layer
template <typename BlockType>
using ConstBlockCallbackFunction =
    std::function<void(const Index3D& block_index, const BlockType* block)>;

template <typename BlockType>
using BlockCallbackFunction =
    std::function<void(const Index3D& block_index, BlockType* block)>;

template <typename BlockType>
void callFunctionOnAllBlocks(const BlockLayer<BlockType>& layer,
                             ConstBlockCallbackFunction<BlockType> callback);

template <typename BlockType>
void callFunctionOnAllBlocks(BlockLayer<BlockType>* layer,
                             BlockCallbackFunction<BlockType> callback);

}  // namespace nvblox

#include "nvblox/map/internal/impl/accessors_impl.h"
