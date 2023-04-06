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

template <typename VoxelType>
typename VoxelBlock<VoxelType>::Ptr VoxelBlock<VoxelType>::allocate(
    MemoryType memory_type) {
  Ptr voxel_block_ptr = make_unified<VoxelBlock>(memory_type);
  if (memory_type == MemoryType::kDevice) {
    initOnGPU(voxel_block_ptr.get());
  }
  return voxel_block_ptr;
}

template <typename VoxelType>
void VoxelBlock<VoxelType>::initOnGPU(VoxelBlock<VoxelType>* block_ptr) {
  setBlockBytesZeroOnGPU(block_ptr);
}

// Initialization specialization for ColorVoxel which is initialized to gray
// with zero weight
template <>
inline void VoxelBlock<ColorVoxel>::initOnGPU(
    VoxelBlock<ColorVoxel>* block_ptr) {
  setColorBlockGrayOnGPU(block_ptr);
}

template <typename BlockType>
void setBlockBytesZeroOnGPU(BlockType* block_device_ptr) {
  // TODO(alexmillane): This is a cuda call in a public header... Is this
  // causing issues?
  cudaMemset(block_device_ptr, 0, sizeof(BlockType));
}

}  // namespace nvblox
