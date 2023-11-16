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

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/layer.h"
#include "nvblox/tests/voxels.h"

namespace nvblox {

// Dummy block type that just stores its own index.
struct IndexBlock {
  typedef unified_ptr<IndexBlock> Ptr;
  typedef unified_ptr<const IndexBlock> ConstPtr;

  Index3D data;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<IndexBlock>(memory_type);
  }

  static Ptr allocateAsync(MemoryType memory_type, const CudaStream&) {
    return make_unified<IndexBlock>(memory_type);
  }
};

// Dummy block that stores a single bool
struct BooleanBlock {
  typedef unified_ptr<BooleanBlock> Ptr;
  typedef unified_ptr<const BooleanBlock> ConstPtr;

  bool data = false;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<BooleanBlock>(memory_type);
  }

  static Ptr allocateAsync(MemoryType memory_type, const CudaStream&) {
    return make_unified<BooleanBlock>(memory_type);
  }
};

struct FloatBlock {
  typedef unified_ptr<FloatBlock> Ptr;
  typedef unified_ptr<const FloatBlock> ConstPtr;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<FloatBlock>(memory_type);
  }

  static Ptr allocateAsync(MemoryType memory_type, const CudaStream&) {
    return make_unified<FloatBlock>(memory_type);
  };

  float block_data = 0.0f;
};

template <typename BlockType>
void setBlockBytesConstantOnGPU(int value, BlockType* block_device_ptr) {
  setBlockBytesConstantOnGPU(value, block_device_ptr, CudaStreamOwning());
}

template <typename BlockType>
void setBlockBytesConstantOnGPUAsync(int value, BlockType* block_device_ptr,
                                     const CudaStream& cuda_stream) {
  checkCudaErrors(
      cudaMemsetAsync(block_device_ptr, value, sizeof(BlockType), cuda_stream));
}

// We use this voxel to test GPU initialization
struct InitializationTestVoxel {
  static constexpr uint8_t kCPUInitializationValue = 0;
  static constexpr uint8_t kGPUInitializationValue = 1;

  uint8_t data = kCPUInitializationValue;
};

template <>
inline void VoxelBlock<InitializationTestVoxel>::initOnGPUAsync(
    VoxelBlock<InitializationTestVoxel>* voxel_block_ptr,
    const CudaStream& cuda_stream) {
  setBlockBytesConstantOnGPUAsync(
      InitializationTestVoxel::kGPUInitializationValue, voxel_block_ptr,
      cuda_stream);
}

// A custom (non-voxel) block which forgets to define allocate(), and therefore
// will fail the type_trait has nvblox::traits::has_allocate<T>()
struct TestBlockNoAllocation {
  typedef unified_ptr<TestBlockNoAllocation> Ptr;
  typedef unified_ptr<const TestBlockNoAllocation> ConstPtr;

  static constexpr uint8_t kCPUInitializationValue = 0;

  uint8_t data = kCPUInitializationValue;
};

using IndexBlockLayer = BlockLayer<IndexBlock>;
using BooleanBlockLayer = BlockLayer<BooleanBlock>;
using FloatBlockLayer = BlockLayer<FloatBlock>;
using FloatVoxelBlock = VoxelBlock<FloatVoxel>;
using FloatVoxelLayer = VoxelBlockLayer<FloatVoxel>;
using InitializationTestVoxelBlock = VoxelBlock<InitializationTestVoxel>;
using InitializationTestVoxelLayer = VoxelBlockLayer<InitializationTestVoxel>;

}  // namespace nvblox
