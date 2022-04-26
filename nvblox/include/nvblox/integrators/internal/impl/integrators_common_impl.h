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

template <typename BlockType>
__host__ std::vector<BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    BlockLayer<BlockType>* layer_ptr) {
  std::vector<BlockType*> block_ptrs;
  block_ptrs.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    typename BlockType::Ptr block_ptr = layer_ptr->getBlockAtIndex(block_index);
    CHECK(block_ptr);
    block_ptrs.push_back(block_ptr.get());
  }
  return block_ptrs;
}

template <typename BlockType>
__host__ std::vector<const BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    const BlockLayer<BlockType>& layer) {
  std::vector<const BlockType*> block_ptrs;
  block_ptrs.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    const typename BlockType::ConstPtr block_ptr =
        layer.getBlockAtIndex(block_index);
    CHECK(block_ptr);
    block_ptrs.push_back(block_ptr.get());
  }
  return block_ptrs;
}

template <typename VoxelType>
void allocateBlocksWhereRequired(const std::vector<Index3D>& block_indices,
                                 BlockLayer<VoxelBlock<VoxelType>>* layer) {
  for (const Index3D& block_index : block_indices) {
    layer->allocateBlockAtIndex(block_index);
  }
}

}  // namespace nvblox