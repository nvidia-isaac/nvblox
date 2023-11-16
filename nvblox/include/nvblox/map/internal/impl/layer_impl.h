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

#include "nvblox/utils/logging.h"

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/map/accessors.h"

namespace nvblox {

template <typename BlockType>
void BlockLayer<BlockType>::copyFrom(const BlockLayer& other) {
  block_size_ = other.block_size_;
  gpu_layer_view_up_to_date_ = false;

  copyFromAsync(other, CudaStreamOwning());
}

template <typename BlockType>
void BlockLayer<BlockType>::copyFromAsync(const BlockLayer& other,
                                          const CudaStream cuda_stream) {
  LOG(INFO) << "Deep copy of BlockLayer containing "
            << other.numAllocatedBlocks() << " blocks.";

  block_size_ = other.block_size_;
  gpu_layer_view_up_to_date_ = false;

  // Re-create all the blocks.
  std::vector<Index3D> all_block_indices = other.getAllBlockIndices();

  // Iterate over all blocks, clonin'.
  for (const Index3D& block_index : all_block_indices) {
    typename BlockType::ConstPtr block = other.getBlockAtIndex(block_index);
    if (block == nullptr) {
      continue;
    }
    blocks_.emplace(block_index, block.cloneAsync(memory_type_, cuda_stream));
  }
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
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtIndexAsync(
    const Index3D& index, const CudaStream& cuda_stream) {
  auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return it->second;
  } else {
    // Invalidate the GPU hash
    gpu_layer_view_up_to_date_ = false;
    // Blocks define their own method for allocation.
    auto insert_status = blocks_.emplace(
        index, BlockType::allocateAsync(memory_type_, cuda_stream));
    return insert_status.first->second;
  }
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtIndex(
    const Index3D& index) {
  return allocateBlockAtIndexAsync(index, CudaStreamOwning());
}

template <typename BlockType>
void BlockLayer<BlockType>::allocateBlocksAtIndices(
    const std::vector<Index3D>& indices, const CudaStream& cuda_stream) {
  for (const Index3D& idx : indices) {
    allocateBlockAtIndexAsync(idx, cuda_stream);
  }
  cuda_stream.synchronize();
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
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtPositionAsync(
    const Eigen::Vector3f& position, const CudaStream& cuda_stream) {
  return allocateBlockAtIndexAsync(
      getBlockIndexFromPositionInLayer(block_size_, position), cuda_stream);
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtPosition(
    const Eigen::Vector3f& position) {
  return allocateBlockAtPositionAsync(position, CudaStreamOwning());
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
std::vector<BlockType*> BlockLayer<BlockType>::getAllBlockPointers() {
  std::vector<BlockType*> block_ptrs;
  block_ptrs.reserve(blocks_.size());

  for (auto& kv : blocks_) {
    block_ptrs.push_back(kv.second.get());
  }
  return block_ptrs;
}

template <typename BlockType>
std::vector<Index3D> BlockLayer<BlockType>::getBlockIndicesIf(
    std::function<bool(const Index3D&)> predicate) {
  std::vector<Index3D> all_indices = getAllBlockIndices();
  std::vector<Index3D> indices_out;
  std::copy_if(all_indices.begin(), all_indices.end(),
               std::back_inserter(indices_out), predicate);
  return indices_out;
}

template <typename BlockType>
bool BlockLayer<BlockType>::isBlockAllocated(const Index3D& index) const {
  const auto it = blocks_.find(index);
  return (it != blocks_.end());
}

template <typename BlockType>
bool BlockLayer<BlockType>::clearBlock(const Index3D& index) {
  if (blocks_.erase(index)) {
    gpu_layer_view_up_to_date_ = false;
    return true;
  } else {
    return false;
  }
}

template <typename BlockType>
void BlockLayer<BlockType>::clearBlocks(const std::vector<Index3D>& indices) {
  for (const auto& idx : indices) {
    if (blocks_.erase(idx)) {
      gpu_layer_view_up_to_date_ = false;
    }
  }
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
  // Call the underlying streamed method on a newly created stream.
  CudaStreamOwning cuda_stream;
  getVoxels(positions_L, voxels_ptr, success_flags_ptr, &cuda_stream);
}

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxels(
    const std::vector<Vector3f>& positions_L,
    std::vector<VoxelType>* voxels_ptr, std::vector<bool>* success_flags_ptr,
    CudaStream* cuda_stream_ptr) const {
  CHECK_NOTNULL(voxels_ptr);
  CHECK_NOTNULL(success_flags_ptr);
  CHECK_NOTNULL(cuda_stream_ptr);

  voxels_ptr->resize(positions_L.size());
  success_flags_ptr->resize(positions_L.size());

  for (size_t i = 0; i < positions_L.size(); i++) {
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
      checkCudaErrors(cudaMemcpyAsync(&(*voxels_ptr)[i], voxel_ptr,
                                      sizeof(VoxelType), cudaMemcpyDefault,
                                      *cuda_stream_ptr));
    }
    // Accessible by the CPU, just do a normal copy
    else {
      (*voxels_ptr)[i] = *voxel_ptr;
    }
  }
  cuda_stream_ptr->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
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
