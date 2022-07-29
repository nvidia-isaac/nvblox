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

#include <glog/logging.h>

#include "nvblox/core/accessors.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"

namespace nvblox {

template <typename BlockType>
BlockLayer<BlockType>::BlockLayer(const BlockLayer& other)
    : BlockLayer(other, other.memory_type_) {}

template <typename BlockType>
BlockLayer<BlockType>::BlockLayer(const BlockLayer& other,
                                  MemoryType memory_type)
    : BlockLayer(other.block_size_, memory_type) {
  LOG(INFO) << "Deep copy of BlockLayer containing "
            << other.numAllocatedBlocks() << " blocks.";
  // Re-create all the blocks.
  std::vector<Index3D> all_block_indices = other.getAllBlockIndices();

  // Iterate over all blocks, clonin'.
  for (const Index3D& block_index : all_block_indices) {
    typename BlockType::ConstPtr block = other.getBlockAtIndex(block_index);
    if (block == nullptr) {
      continue;
    }
    blocks_.emplace(block_index, block.clone(memory_type_));
  }
}

template <typename BlockType>
BlockLayer<BlockType>& BlockLayer<BlockType>::operator=(
    const BlockLayer<BlockType>& other) {
  BlockLayer<BlockType> new_layer(other, memory_type_);
  *this = std::move(new_layer);
  return *this;
}

// Block accessors by index.
template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::getBlockAtIndex(
    const Index3D& index) {
  // Look up the block in the hash?
  // And return it.
  auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return it->second;
  } else {
    return typename BlockType::Ptr();
  }
}

template <typename BlockType>
typename BlockType::ConstPtr BlockLayer<BlockType>::getBlockAtIndex(
    const Index3D& index) const {
  const auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return (it->second);
  } else {
    return typename BlockType::ConstPtr();
  }
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtIndex(
    const Index3D& index) {
  auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return it->second;
  } else {
    // Invalidate the GPU hash
    gpu_layer_view_up_to_date_ = false;
    // Blocks define their own method for allocation.
    auto insert_status =
        blocks_.emplace(index, BlockType::allocate(memory_type_));
    return insert_status.first->second;
  }
}

// Block accessors by position.
template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::getBlockAtPosition(
    const Eigen::Vector3f& position) {
  return getBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
typename BlockType::ConstPtr BlockLayer<BlockType>::getBlockAtPosition(
    const Eigen::Vector3f& position) const {
  return getBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtPosition(
    const Eigen::Vector3f& position) {
  return allocateBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
std::vector<Index3D> BlockLayer<BlockType>::getAllBlockIndices() const {
  std::vector<Index3D> indices;
  indices.reserve(blocks_.size());

  for (const auto& kv : blocks_) {
    indices.push_back(kv.first);
  }

  return indices;
}

template <typename BlockType>
bool BlockLayer<BlockType>::isBlockAllocated(const Index3D& index) const {
  const auto it = blocks_.find(index);
  return (it != blocks_.end());
}

template <typename BlockType>
typename BlockLayer<BlockType>::GPULayerViewType
BlockLayer<BlockType>::getGpuLayerView() const {
  if (!gpu_layer_view_) {
    gpu_layer_view_ = std::make_unique<GPULayerViewType>();
  }
  if (!gpu_layer_view_up_to_date_) {
    (*gpu_layer_view_).reset(const_cast<BlockLayer<BlockType>*>(this));
    gpu_layer_view_up_to_date_ = true;
  }
  return *gpu_layer_view_;
}

// VoxelBlockLayer

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxels(
    const std::vector<Vector3f>& positions_L,
    std::vector<VoxelType>* voxels_ptr,
    std::vector<bool>* success_flags_ptr) const {
  CHECK_NOTNULL(voxels_ptr);
  CHECK_NOTNULL(success_flags_ptr);

  cudaStream_t transfer_stream;
  checkCudaErrors(cudaStreamCreate(&transfer_stream));

  voxels_ptr->resize(positions_L.size());
  success_flags_ptr->resize(positions_L.size());

  for (int i = 0; i < positions_L.size(); i++) {
    const Vector3f& p_L = positions_L[i];
    // Get the block address
    Index3D block_idx;
    Index3D voxel_idx;
    getBlockAndVoxelIndexFromPositionInLayer(this->block_size_, p_L, &block_idx,
                                             &voxel_idx);
    const typename VoxelBlock<VoxelType>::ConstPtr block_ptr =
        this->getBlockAtIndex(block_idx);
    if (!block_ptr) {
      (*success_flags_ptr)[i] = false;
      continue;
    }
    (*success_flags_ptr)[i] = true;
    // Get the voxel address
    const auto block_raw_ptr = block_ptr.get();
    const VoxelType* voxel_ptr =
        &block_raw_ptr->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
    // Copy the Voxel to the CPU (if on the GPU)
    if (this->memory_type_ == MemoryType::kDevice) {
      cudaMemcpyAsync(&(*voxels_ptr)[i], voxel_ptr, sizeof(VoxelType),
                      cudaMemcpyDefault, transfer_stream);
    }
    // Accessible by the CPU, just do a normal copy
    else {
      (*voxels_ptr)[i] = *voxel_ptr;
    }
  }
  cudaStreamSynchronize(transfer_stream);
  checkCudaErrors(cudaStreamDestroy(transfer_stream));
}

template <typename VoxelType>
std::pair<VoxelType, bool> VoxelBlockLayer<VoxelType>::getVoxel(
    const Vector3f& p_L) const {
  const std::vector<Vector3f> positions_L(1, p_L);
  std::vector<VoxelType> voxels;
  std::vector<bool> success_flags;
  getVoxels(positions_L, &voxels, &success_flags);
  return {voxels[0], success_flags[0]};
}

template <typename VoxelType>
VoxelBlockLayer<VoxelType>::VoxelBlockLayer(const VoxelBlockLayer& other)
    : VoxelBlockLayer(other, other.memory_type_) {}

template <typename VoxelType>
VoxelBlockLayer<VoxelType>::VoxelBlockLayer(const VoxelBlockLayer& other,
                                            MemoryType memory_type)
    : BlockLayer<VoxelBlock<VoxelType>>(other, memory_type),
      voxel_size_(other.voxel_size_) {
  // Copying done in the base class.
}

template <typename VoxelType>
VoxelBlockLayer<VoxelType>& VoxelBlockLayer<VoxelType>::operator=(
    const VoxelBlockLayer<VoxelType>& other) {
  VoxelBlockLayer<VoxelType> new_layer(other, this->memory_type_);
  *this = std::move(new_layer);
  return *this;
}

namespace internal {

template <typename is_voxel_layer>
inline float sizeArgumentFromVoxelSize(float voxel_size);

template <>
inline float sizeArgumentFromVoxelSize<std::true_type>(float voxel_size) {
  return voxel_size;
}

template <>
inline float sizeArgumentFromVoxelSize<std::false_type>(float voxel_size) {
  return voxel_size * VoxelBlock<bool>::kVoxelsPerSide;
}

}  // namespace internal

template <typename LayerType>
constexpr float sizeArgumentFromVoxelSize(float voxel_size) {
  return internal::sizeArgumentFromVoxelSize<
      typename traits::is_voxel_layer<LayerType>::type>(voxel_size);
}

}  // namespace nvblox
