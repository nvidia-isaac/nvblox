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
#include <gtest/gtest.h>

#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"

#include "nvblox/tests/blox.h"
#include "nvblox/tests/blox_utils.h"
#include "nvblox/tests/test_utils_cuda.h"

using namespace nvblox;

TEST(BloxTest, InitializeDistanceBloxOnGPU) {
  TsdfBlock::Ptr block_ptr = TsdfBlock::allocate(MemoryType::kDevice);
  TsdfVoxel zero_voxel;
  zero_voxel.distance = 0.0f;
  zero_voxel.weight = 0.0f;
  EXPECT_TRUE(test_utils::checkBlockAllConstant(block_ptr, zero_voxel));
  TsdfVoxel one_voxel;
  one_voxel.distance = 1.0f;
  one_voxel.weight = 1.0f;
  EXPECT_FALSE(test_utils::checkBlockAllConstant(block_ptr, one_voxel));
}

TEST(BloxTestDeathTest, NoAllocationDefined) {
  // This block type has not defined an allocation function
  EXPECT_FALSE(traits::has_allocate<TestBlockNoAllocation>::value);
}

TEST(BloxTest, CustomGPUInitialization) {
  constexpr float block_size_m = 0.1;
  BlockLayer<VoxelBlock<InitializationTestVoxel>> layer(block_size_m,
                                                        MemoryType::kDevice);
  auto block_ptr = layer.allocateBlockAtIndex(Index3D(0, 0, 0));
  InitializationTestVoxel one_voxel;
  one_voxel.data = 1;
  EXPECT_TRUE(test_utils::checkBlockAllConstant(block_ptr, one_voxel));
  InitializationTestVoxel zero_voxel;
  zero_voxel.data = 0;
  EXPECT_FALSE(test_utils::checkBlockAllConstant(block_ptr, zero_voxel));
}

TEST(BloxTest, ColorInitialization) {
  // The color block has non-trivial initialization. Therefore we test this
  // block specifically as a representative of the class of non-trivially
  // constructed blocks.

  // kUnified
  constexpr float block_size_m = 0.1;
  BlockLayer<ColorBlock> layer_unified(block_size_m, MemoryType::kUnified);
  auto block_ptr = layer_unified.allocateBlockAtIndex(Index3D(0, 0, 0));
  auto check_color_voxels = [](const Index3D&, const ColorVoxel* voxel_ptr) {
    EXPECT_EQ(voxel_ptr->color, Color::Gray());
    EXPECT_EQ(voxel_ptr->weight, 0.0f);
  };
  callFunctionOnAllVoxels<ColorVoxel>(*block_ptr, check_color_voxels);

  // kDevice
  BlockLayer<ColorBlock> layer_device(block_size_m, MemoryType::kDevice);
  auto block_device_ptr = layer_device.allocateBlockAtIndex(Index3D(0, 0, 0));
  ColorVoxel gray_voxel;
  gray_voxel.color.r = 127;
  gray_voxel.color.g = 127;
  gray_voxel.color.b = 127;
  gray_voxel.weight = 0.0f;
  EXPECT_TRUE(test_utils::checkBlockAllConstant(block_device_ptr, gray_voxel));
  ColorVoxel zero_voxel;
  zero_voxel.color.r = 0;
  zero_voxel.color.g = 0;
  zero_voxel.color.b = 0;
  zero_voxel.weight = 0.0f;
  EXPECT_FALSE(test_utils::checkBlockAllConstant(block_device_ptr, zero_voxel));
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
