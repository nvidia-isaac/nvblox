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

#include <memory>
#include <type_traits>
#include <typeindex>

#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"

#include "nvblox/tests/blox.h"

using namespace nvblox;

TEST(LayerCakeTest, addAndRetrieve) {
  // Create
  const float voxel_size = 0.1f;
  LayerCake cake(voxel_size);

  // Add a TsdfLayer with a block
  TsdfLayer* tsdf_layer_ptr_1 = cake.add<TsdfLayer>(MemoryType::kUnified);
  auto block_ptr_1 = tsdf_layer_ptr_1->allocateBlockAtIndex(Index3D(0, 0, 0));
  block_ptr_1->voxels[0][0][0].distance = 1.0f;

  // Retrieve
  ASSERT_TRUE(cake.exists<TsdfLayer>());
  const TsdfLayer* tsdf_layer_ptr_2 = cake.getConstPtr<TsdfLayer>();
  EXPECT_TRUE(tsdf_layer_ptr_2->isBlockAllocated(Index3D(0, 0, 0)));
  auto block_ptr_2 = tsdf_layer_ptr_2->getBlockAtIndex(Index3D(0, 0, 0));
  EXPECT_EQ(block_ptr_2->voxels[0][0][0].distance, 1.0f);
}

TEST(LayerCakeTest, create) {
  // Bring in a custom test layer
  using BooleanLayer = BlockLayer<BooleanBlock>;

  // Create
  const float voxel_size = 0.1f;
  auto cake = LayerCake::create<TsdfLayer, ColorLayer, EsdfLayer, BooleanLayer>(
      voxel_size, MemoryType::kUnified);

  // Checks
  EXPECT_TRUE(cake.exists<TsdfLayer>());
  EXPECT_TRUE(cake.exists<ColorLayer>());
  EXPECT_TRUE(cake.exists<EsdfLayer>());
  EXPECT_TRUE(cake.exists<BooleanLayer>());

  EXPECT_FALSE(cake.exists<MeshLayer>());
}

TEST(LayerCakeTest, moveOperations) {
  const float voxel_size = 0.1f;
  LayerCake cake_1 =
      LayerCake::create<TsdfLayer, ColorLayer>(voxel_size, MemoryType::kDevice);

  LayerCake cake_2 =
      LayerCake::create<MeshLayer, EsdfLayer>(voxel_size, MemoryType::kDevice);

  cake_1 = std::move(cake_2);

  EXPECT_FALSE(cake_1.exists<TsdfLayer>());
  EXPECT_FALSE(cake_1.exists<ColorLayer>());
  EXPECT_TRUE(cake_1.exists<MeshLayer>());
  EXPECT_TRUE(cake_1.exists<EsdfLayer>());

  EXPECT_FALSE(cake_2.exists<TsdfLayer>());
  EXPECT_FALSE(cake_2.exists<ColorLayer>());
  EXPECT_FALSE(cake_2.exists<MeshLayer>());
  EXPECT_FALSE(cake_2.exists<EsdfLayer>());
}

TEST(LayerCakeTest, voxelAndBlockSize) {
  const float voxel_size = 0.1f;
  LayerCake cake =
      LayerCake::create<TsdfLayer, MeshLayer>(voxel_size, MemoryType::kDevice);

  const float expected_block_size = voxel_size * TsdfBlock::kVoxelsPerSide;

  EXPECT_EQ(cake.getPtr<TsdfLayer>()->voxel_size(), 0.1f);
  EXPECT_EQ(cake.getPtr<MeshLayer>()->block_size(), expected_block_size);
}

TEST(LayerCakeTest, differentMemoryTypes) {
  const float voxel_size = 0.1f;

  LayerCake cake =
      LayerCake::create<TsdfLayer, ColorLayer, EsdfLayer, MeshLayer>(
          voxel_size, MemoryType::kDevice, MemoryType::kUnified,
          MemoryType::kHost, MemoryType::kDevice);

  EXPECT_EQ(cake.get<TsdfLayer>().memory_type(), MemoryType::kDevice);
  EXPECT_EQ(cake.get<ColorLayer>().memory_type(), MemoryType::kUnified);
  EXPECT_EQ(cake.get<EsdfLayer>().memory_type(), MemoryType::kHost);
  EXPECT_EQ(cake.get<MeshLayer>().memory_type(), MemoryType::kDevice);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
