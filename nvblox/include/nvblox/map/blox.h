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

#include <memory>

#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

/// A block that contains 8x8x8 voxels of a given type.
template <typename _VoxelType>
struct VoxelBlock {
  typedef unified_ptr<VoxelBlock> Ptr;
  typedef unified_ptr<const VoxelBlock> ConstPtr;

  /// Allow introspection of the voxel type through BlockType::VoxelType
  typedef _VoxelType VoxelType;

  static constexpr int kVoxelsPerSide = 8;
  static constexpr int kNumVoxels =
      kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;
  VoxelType voxels[kVoxelsPerSide][kVoxelsPerSide][kVoxelsPerSide];

  /// Allocate a voxel block of a given memory type.
  static Ptr allocateAsync(MemoryType memory_type,
                           const CudaStream& cuda_stream);
  static Ptr allocate(MemoryType memory_type);
  /// Initializes all the memory of the voxels to 0 by default, can be
  /// specialized by voxel type.
  static void initOnGPUAsync(VoxelBlock* block_ptr,
                             const CudaStream& cuda_stream);
};

// Initialization Utility Functions
/// Set all the memory of the block to 0 on the GPU.
template <typename BlockType>
void setBlockBytesZeroOnGPUAsync(BlockType* block_device_ptr);
/// Set all of the default colors to gray on a GPU.
void setColorBlockGrayOnGPUAsync(VoxelBlock<ColorVoxel>* block_device_ptr,
                                 const CudaStream& cuda_stream);

}  // namespace nvblox

#include "nvblox/map/internal/impl/blox_impl.h"
