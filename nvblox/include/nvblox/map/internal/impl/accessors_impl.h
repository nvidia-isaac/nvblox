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

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

template <typename BlockType>
bool getBlockAtPosition(const BlockLayer<BlockType>& layer,
                        const Eigen::Vector3f& position,
                        const BlockType** data) {
  typename BlockType::ConstPtr block_ptr = layer.getBlockAtPosition(position);
  if (block_ptr != nullptr) {
    *data = block_ptr.get();
    return true;
  }
  return false;
}

template <typename VoxelType>
bool getVoxelAtPosition(const BlockLayer<VoxelBlock<VoxelType>>& layer,
                        const Eigen::Vector3f& position,
                        const VoxelType** data) {
  Index3D block_idx;
  Index3D voxel_idx;
  getBlockAndVoxelIndexFromPositionInLayer(layer.block_size(), position,
                                           &block_idx, &voxel_idx);
  const typename VoxelBlock<VoxelType>::ConstPtr block_ptr =
      layer.getBlockAtIndex(block_idx);
  if (block_ptr) {
    *data = &block_ptr->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
    return true;
  }
  return false;
}

template <typename VoxelType>
void callFunctionOnAllVoxels(const BlockLayer<VoxelBlock<VoxelType>>& layer,
                             ConstVoxelCallbackFunction<VoxelType> callback) {
  std::vector<Index3D> block_indices = layer.getAllBlockIndices();

  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;

  bool clone = (layer.memory_type() == MemoryType::kDevice);

  // Iterate over all the blocks:
  for (const Index3D& block_index : block_indices) {
    Index3D voxel_index;
    typename VoxelBlock<VoxelType>::ConstPtr block;
    if (clone) {
      block = layer.getBlockAtIndex(block_index).clone(MemoryType::kHost);
    } else {
      block = layer.getBlockAtIndex(block_index);
    }
    if (!block) {
      continue;
    }
    // Iterate over all the voxels:
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          // Get the voxel and call the callback on it:
          const VoxelType* voxel =
              &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          callback(block_index, voxel_index, voxel);
        }
      }
    }
  }
}

template <typename VoxelType>
void callFunctionOnAllVoxels(BlockLayer<VoxelBlock<VoxelType>>* layer,
                             VoxelCallbackFunction<VoxelType> callback) {
  std::vector<Index3D> block_indices = layer->getAllBlockIndices();

  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;

  // Iterate over all the blocks:
  for (const Index3D& block_index : block_indices) {
    Index3D voxel_index;
    typename VoxelBlock<VoxelType>::Ptr block =
        layer->getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    // Iterate over all the voxels:
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          // Get the voxel and call the callback on it:
          VoxelType* voxel =
              &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          callback(block_index, voxel_index, voxel);
        }
      }
    }
  }
}

template <typename VoxelType>
const VoxelType* getVoxelAtBlockAndVoxelIndex(
    const BlockLayer<VoxelBlock<VoxelType>>& layer, const Index3D& block_index,
    const Index3D& voxel_index) {
  typename VoxelBlock<VoxelType>::ConstPtr block =
      layer.getBlockAtIndex(block_index);
  if (!block) {
    return nullptr;
  }
  // TODO: add DCHECKS
  return &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
}

template <typename VoxelType>
void callFunctionOnAllVoxels(const VoxelBlock<VoxelType>& block,
                             ConstBlockCallbackFunction<VoxelType> callback) {
  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;
  // Iterate over all the voxels:
  Index3D voxel_index;
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        // Get the voxel and call the callback on it:
        const VoxelType* voxel =
            &block.voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
        callback(voxel_index, voxel);
      }
    }
  }
}

template <typename VoxelType>
void callFunctionOnAllVoxels(VoxelBlock<VoxelType>* block,
                             BlockCallbackFunction<VoxelType> callback) {
  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;
  // Iterate over all the voxels:
  Index3D voxel_index;
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        // Get the voxel and call the callback on it:
        VoxelType* voxel =
            &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
        callback(voxel_index, voxel);
      }
    }
  }
}

template <typename BlockType>
void callFunctionOnAllBlocks(const BlockLayer<BlockType>& layer,
                             ConstBlockCallbackFunction<BlockType> callback) {
  const std::vector<Index3D> block_indices = layer.getAllBlockIndices();
  for (const Index3D& block_index : block_indices) {
    typename BlockType::ConstPtr block = layer.getBlockAtIndex(block_index);
    CHECK(block);
    callback(block_index, block.get());
  }
}

template <typename BlockType>
void callFunctionOnAllBlocks(BlockLayer<BlockType>* layer,
                             BlockCallbackFunction<BlockType> callback) {
  std::vector<Index3D> block_indices = layer->getAllBlockIndices();
  for (const Index3D& block_index : block_indices) {
    typename BlockType::Ptr block = layer->getBlockAtIndex(block_index);
    CHECK(block);
    callback(block_index, block.get());
  }
}

}  // namespace nvblox
