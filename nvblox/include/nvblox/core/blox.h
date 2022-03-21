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
#include "nvblox/core/voxels.h"

namespace nvblox {

template <typename VoxelType>
struct VoxelBlock {
  typedef unified_ptr<VoxelBlock> Ptr;
  typedef unified_ptr<const VoxelBlock> ConstPtr;

  static constexpr size_t kVoxelsPerSide = 8;
  VoxelType voxels[kVoxelsPerSide][kVoxelsPerSide][kVoxelsPerSide];

  static Ptr allocate(MemoryType memory_type);
  static void initOnGPU(VoxelBlock* block_ptr);
};

struct FreespaceBlock {
  typedef unified_ptr<FreespaceBlock> Ptr;
  typedef unified_ptr<const FreespaceBlock> ConstPtr;

  bool free = true;
};

// Multires Block

// Reference Block

// Initialization Utility Functions
template <typename BlockType>
void setBlockBytesZeroOnGPU(BlockType* block_device_ptr);
void setColorBlockGrayOnGPU(VoxelBlock<ColorVoxel>* block_device_ptr);

}  // namespace nvblox

#include "nvblox/core/impl/blox_impl.h"
