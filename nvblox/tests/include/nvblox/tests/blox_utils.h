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

#include "nvblox/core/blox.h"
#include "nvblox/core/common_names.h"

namespace nvblox {

template <typename BlockType>
void setBlockBytesConstantOnGPU(int value, BlockType* block_device_ptr) {
  cudaMemset(block_device_ptr, value, sizeof(BlockType));
}

struct TestVoxel {
  static constexpr uint8_t kCPUInitializationValue = 0;
  static constexpr uint8_t kGPUInitializationValue = 1;

  uint8_t data = kCPUInitializationValue;
};

template <>
inline void VoxelBlock<TestVoxel>::initOnGPU(
    VoxelBlock<TestVoxel>* voxel_block_ptr) {
  setBlockBytesConstantOnGPU(TestVoxel::kGPUInitializationValue,
                             voxel_block_ptr);
}

// A custom (non-voxel) block which forgets to define allocate(), and therefore
// will fail the type_trait has nvblox::traits::has_allocate<T>()
struct TestBlockNoAllocation {
  typedef unified_ptr<TestBlockNoAllocation> Ptr;
  typedef unified_ptr<const TestBlockNoAllocation> ConstPtr;

  static constexpr uint8_t kCPUInitializationValue = 0;

  uint8_t data = kCPUInitializationValue;
};

using TestBlock = VoxelBlock<TestVoxel>;

namespace test_utils {

// Fills a TsdfBlock such that the voxels distance and weight values are their
// linear index (as a float)
void setTsdfBlockVoxelsInSequence(TsdfBlock::Ptr block);

bool checkBlockAllConstant(const TsdfBlock::Ptr block, TsdfVoxel voxel_cpu);
bool checkBlockAllConstant(const TestBlock::Ptr block, TestVoxel voxel_cpu);
bool checkBlockAllConstant(const ColorBlock::Ptr block, ColorVoxel voxel_cpu);

}  // namespace test_utils
}  // namespace nvblox
