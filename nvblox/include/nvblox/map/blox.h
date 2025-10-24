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
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/voxel_iterator.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

/// A block that contains 8x8x8 voxels of a given type.
template <typename _VoxelType>
struct VoxelBlock {
  using Ptr = unified_ptr<VoxelBlock>;
  using ConstPtr = unified_ptr<const VoxelBlock>;

  /// Allow introspection of the voxel type through BlockType::VoxelType
  using VoxelType = _VoxelType;

  static constexpr int kVoxelsPerSide = 8;
  static constexpr int kNumVoxels =
      kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;

  /// Voxel iterator types
  using iterator = VoxelIterator<VoxelType, kVoxelsPerSide, false>;
  using const_iterator = VoxelIterator<const VoxelType, kVoxelsPerSide, true>;

  VoxelType voxels[kVoxelsPerSide][kVoxelsPerSide][kVoxelsPerSide];

  /// Allocate a voxel block of a given memory type.
  static Ptr allocateAsync(MemoryType memory_type,
                           const CudaStream& cuda_stream);
  static Ptr allocate(MemoryType memory_type);
  /// Initializes all the memory of the voxels to 0 by default, can be
  /// specialized by voxel type.
  static void initAsync(VoxelBlock* block_ptr, const MemoryType memory_type,
                        const CudaStream& cuda_stream);

  /// Indexing via 3D index vector
  const VoxelType& operator()(const Index3D& idx) const;
  /// Indexing via 3D index vector
  VoxelType& operator()(const Index3D& idx);

  /// Get an iterator to the first voxel
  iterator begin();
  const_iterator cbegin() const;

  /// Get an iterator to the past-the-end voxel
  iterator end();
  const_iterator cend() const;
};

/// Return the size in bytes of a voxel block. Note that  function needs to be
/// called from host and can therefore not be a member of VoxelBlock (which is
/// typically allocated as a GPU pointers)
template <typename VoxelType>
constexpr size_t sizeInBytes(const VoxelBlock<VoxelType>*);

// Initialization Utility Functions
/// Set all the memory of the block to 0 on the GPU.
template <typename BlockType>
void setBlockBytesZeroOnGPUAsync(BlockType* block_device_ptr);
/// Set all of the default colors to gray on a GPU.
void setColorBlockGrayOnGPUAsync(VoxelBlock<ColorVoxel>* block_device_ptr,
                                 const CudaStream& cuda_stream);
/// Batch initialization of device blocks.
///
/// Voxels are initialized according to their respective constructor
///
/// @param blocks  Vector of device block pointers
/// @param cuda_stream  Cuda stream
/// @param memory_type  Memory type (only used for meshblock specialization)
template <typename BlockType>
void initializeBlocksAsync(host_vector<BlockType*>& blocks,
                           const CudaStream& cuda_stream,
                           const MemoryType memory_type = MemoryType::kHost);

/// Return the max number of cuda threads needed to process a VoxelBlock
template <typename VoxelType>
constexpr int kMaxNumThreadsPerBlock() {
  return VoxelBlock<VoxelType>::kNumVoxels;
}

/// Get the voxel size for a block
/// @param block_size_m The side-length of a block.
template <typename BlockType>
__host__ __device__ float voxelSize(const float block_size_m) {
  return block_size_m / BlockType::kVoxelsPerSide;
}

}  // namespace nvblox

#include "nvblox/map/internal/impl/blox_impl.h"
