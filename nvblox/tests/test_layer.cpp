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

#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

#include "nvblox/tests/blox.h"
#include "nvblox/tests/blox_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

TEST(LayerTest, InsertionAndRetrieval) {
  // Make sure this is deterministic.
  std::srand(0);

  // Empty layer
  constexpr float KTestBlockSize = 0.1;
  BlockLayer<IndexBlock> layer(KTestBlockSize, MemoryType::kUnified);

  // Get some distict block locations
  constexpr int kNumTestItems = 100;
  constexpr int kMinimumIndexValue = -100;
  constexpr int kMaximumIndexValue = 100;

  Index3DSet unique_block_indices;
  for (int i = 0; i < kNumTestItems; i++) {
    while (!unique_block_indices
                .insert(test_utils::getRandomIndex3dInRange(kMinimumIndexValue,
                                                            kMaximumIndexValue))
                .second) {
      continue;
    }
  }

  // Allocate blocks at these indices (and store index in block for checking
  // later)
  for (auto it = unique_block_indices.begin(); it != unique_block_indices.end();
       it++) {
    EXPECT_FALSE(layer.getBlockAtIndex(*it));
    IndexBlock::Ptr block_ptr = layer.allocateBlockAtIndex(*it);
    block_ptr->data = *it;
  }

  // Allocate blocks at these positions
  for (auto it = unique_block_indices.begin(); it != unique_block_indices.end();
       it++) {
    IndexBlock::ConstPtr block_ptr = layer.getBlockAtIndex(*it);
    // Check it's there
    EXPECT_NE(block_ptr, nullptr);
    // Check it has the data we put in.
    EXPECT_EQ(block_ptr->data, *it);
  }
}

TEST(LayerTest, EmptyLayer) {
  constexpr float KTestBlockSize = 0.1;
  BlockLayer<IndexBlock> layer(KTestBlockSize, MemoryType::kUnified);

  IndexBlock::Ptr index_block_ptr = layer.getBlockAtIndex(Index3D(4, 50, -10));

  EXPECT_FALSE(index_block_ptr);
  EXPECT_EQ(index_block_ptr.get(), nullptr);
}

TEST(LayerTest, MinCornerBasedIndexing) {
  // Check that indexing is performed with block origin at minimum corner and
  // spanning block_size. E.g. from [0,0,0], [1,1,1], exclusive on the top side.

  constexpr float KTestBlockSize = 0.1;
  BlockLayer<BooleanBlock> layer(KTestBlockSize, MemoryType::kUnified);

  const Vector3f kPosition3DEpsilson(0.001f, 0.001f, 0.001f);
  Vector3f position_low(0, 0, 0);
  Vector3f position_high =
      KTestBlockSize * Vector3f::Ones() - kPosition3DEpsilson;

  // Put something in on the low side of the block's range
  BooleanBlock::Ptr block_low_ptr = layer.allocateBlockAtPosition(position_low);
  block_low_ptr->data = true;

  // Check we get the same block back on the high side.
  BooleanBlock::ConstPtr block_high_ptr =
      layer.allocateBlockAtPosition(position_high);
  EXPECT_NE(block_high_ptr, nullptr);
  EXPECT_EQ(block_low_ptr, block_high_ptr);
  EXPECT_TRUE(block_high_ptr->data);
}

TsdfLayer generateTsdfLayer() {
  constexpr float voxel_size_m = 1.0;
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kDevice);
  auto block_ptr = tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 0));
  test_utils::setTsdfBlockVoxelsInSequence(block_ptr);
  return tsdf_layer;
}

std::vector<Vector3f> generateQueryPositions() {
  // Generating the voxel center positions
  std::vector<Vector3f> positions_L;
  for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
    for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
      for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
        positions_L.push_back(Index3D(x, y, z).cast<float>() +
                              0.5 * Vector3f::Ones());
      }
    }
  }
  return positions_L;
}

void checkSequentialTsdfTest(const std::vector<TsdfVoxel>& voxels,
                             const std::vector<bool>& flags) {
  for (size_t i = 0; i < voxels.size(); i++) {
    EXPECT_TRUE(flags[i]);
    EXPECT_EQ(voxels[i].distance, static_cast<float>(i));
    EXPECT_EQ(voxels[i].weight, static_cast<float>(i));
  }
}

TEST(VoxelLayerTest, CopyVoxelsToHost) {
  TsdfLayer tsdf_layer = generateTsdfLayer();

  // Check voxel center positions
  std::vector<Vector3f> positions_L = generateQueryPositions();
  std::vector<TsdfVoxel> voxels;
  std::vector<bool> flags;
  tsdf_layer.getVoxels(positions_L, &voxels, &flags);
  checkSequentialTsdfTest(voxels, flags);

  // Now try some edge cases

  // Just inside the block from {0.0f, 0.0f, 0.0f}
  constexpr float kEps = 1e-5;
  const Vector3f kVecEps = kEps * Vector3f::Ones();
  tsdf_layer.getVoxels({kVecEps}, &voxels, &flags);
  EXPECT_EQ(flags.size(), 1);
  EXPECT_EQ(voxels.size(), 1);
  EXPECT_TRUE(flags[0]);
  EXPECT_EQ(voxels[0].distance, 0.0f);
  EXPECT_EQ(voxels[0].weight, 0.0f);

  // Just inside the block from it's far boundary {8.0f, 8.0f, 8.0f}
  tsdf_layer.getVoxels({Vector3f(8.0f, 8.0f, 8.0f) - kVecEps}, &voxels, &flags);
  EXPECT_TRUE(flags[0]);
  EXPECT_EQ(voxels[0].distance, 511.0f);
  EXPECT_EQ(voxels[0].weight, 511.0f);

  // Just outside the block from {0.0f, 0.0f, 0.0f}
  tsdf_layer.getVoxels({-kVecEps}, &voxels, &flags);
  EXPECT_FALSE(flags[0]);

  // Just outside the block from it's far boundary {8.0f, 8.0f, 8.0f}
  tsdf_layer.getVoxels({Vector3f(8.0f, 8.0f, 8.0f) + kVecEps}, &voxels, &flags);
  EXPECT_FALSE(flags[0]);
}

TEST(VoxelLayerTest, GetTsdfVoxelsOnDevice) {
  TsdfLayer tsdf_layer = generateTsdfLayer();

  // Check voxel center positions
  device_vector<Vector3f> query_device;
  query_device.copyFrom(generateQueryPositions());

  device_vector<TsdfVoxel> voxels;
  device_vector<bool> flags;
  tsdf_layer.getVoxelsGPU(query_device, &voxels, &flags);
  unified_vector<TsdfVoxel> voxels_host;
  voxels_host.copyFrom(voxels);
  unified_vector<bool> flags_host;
  flags_host.copyFrom(flags);
  checkSequentialTsdfTest(voxels_host.toVector(), flags_host.toVector());

  // Now try some edge cases

  // Just inside the block from {0.0f, 0.0f, 0.0f}
  Vector3f kVecEps = 1e-5 * Vector3f::Ones();
  std::vector<Vector3f> query = {kVecEps};
  query_device.copyFrom(query);
  tsdf_layer.getVoxelsGPU(query_device, &voxels, &flags);
  voxels_host.copyFrom(voxels);
  flags_host.copyFrom(flags);
  EXPECT_EQ(flags_host.size(), 1);
  EXPECT_EQ(voxels_host.size(), 1);
  EXPECT_TRUE(flags_host[0]);
  EXPECT_EQ(voxels_host[0].distance, 0.0f);
  EXPECT_EQ(voxels_host[0].weight, 0.0f);

  // Just inside the block from it's far boundary {8.0f, 8.0f, 8.0f}
  query = {Vector3f(8.0f, 8.0f, 8.0f) - kVecEps};
  query_device.copyFrom(query);
  tsdf_layer.getVoxelsGPU(query_device, &voxels, &flags);
  voxels_host.copyFrom(voxels);
  flags_host.copyFrom(flags);
  EXPECT_TRUE(flags_host[0]);
  EXPECT_EQ(voxels_host[0].distance, 511.0f);
  EXPECT_EQ(voxels_host[0].weight, 511.0f);

  // Just outside the block from {0.0f, 0.0f, 0.0f}
  query = {-kVecEps};
  query_device.copyFrom(query);
  tsdf_layer.getVoxelsGPU(query_device, &voxels, &flags);
  flags_host.copyFrom(flags);
  EXPECT_FALSE(flags_host[0]);

  // Just outside the block from it's far boundary {8.0f, 8.0f, 8.0f}
  query = {Vector3f(8.0f, 8.0f, 8.0f) + kVecEps};
  query_device.copyFrom(query);
  tsdf_layer.getVoxelsGPU(query_device, &voxels, &flags);
  flags_host.copyFrom(flags);
  EXPECT_FALSE(flags_host[0]);
}

TEST(VoxelLayerTest, GetCustomVoxelsOnDevice) {
  // Generate FloatingVoxelLayer
  constexpr float voxel_size_m = 1.0;
  FloatVoxelLayer voxel_layer(voxel_size_m, MemoryType::kDevice);
  auto block_ptr = voxel_layer.allocateBlockAtIndex(Index3D(0, 0, 0));
  test_utils::setFloatingBlockVoxelsInSequence(block_ptr);

  // Check voxel center positions
  device_vector<Vector3f> query_device;
  query_device.copyFrom(generateQueryPositions());
  device_vector<FloatVoxel> voxels;
  device_vector<bool> flags;
  voxel_layer.getVoxelsGPU(query_device, &voxels, &flags);
  unified_vector<FloatVoxel> voxels_host;
  voxels_host.copyFrom(voxels);
  unified_vector<bool> flags_host;
  flags_host.copyFrom(flags);
  for (size_t i = 0; i < voxels.size(); i++) {
    EXPECT_TRUE(flags_host[i]);
    EXPECT_EQ(voxels_host[i].voxel_data, static_cast<float>(i));
  }
}

TEST(LayerTest, MoveOperations) {
  constexpr float voxel_size_m = 1.0;

  // BlockLayer (MeshLayer being the representative)
  MeshLayer mesh_layer_1(voxel_size_m, MemoryType::kDevice);
  mesh_layer_1.allocateBlockAtIndex(Index3D(0, 0, 0));

  EXPECT_TRUE(mesh_layer_1.isBlockAllocated(Index3D(0, 0, 0)));

  MeshLayer mesh_layer_2 = std::move(mesh_layer_1);

  EXPECT_FALSE(mesh_layer_1.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_TRUE(mesh_layer_2.isBlockAllocated(Index3D(0, 0, 0)));

  MeshLayer mesh_layer_3(std::move(mesh_layer_2));

  EXPECT_FALSE(mesh_layer_2.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_TRUE(mesh_layer_3.isBlockAllocated(Index3D(0, 0, 0)));

  // VoxelBlockLayer (TsdfLayer being the representative)
  TsdfLayer tsdf_layer_1(voxel_size_m, MemoryType::kDevice);
  tsdf_layer_1.allocateBlockAtIndex(Index3D(0, 0, 0));

  TsdfLayer tsdf_layer_2 = std::move(tsdf_layer_1);

  EXPECT_FALSE(tsdf_layer_1.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_TRUE(tsdf_layer_2.isBlockAllocated(Index3D(0, 0, 0)));

  TsdfLayer tsdf_layer_3(std::move(tsdf_layer_2));

  EXPECT_FALSE(tsdf_layer_2.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_TRUE(tsdf_layer_3.isBlockAllocated(Index3D(0, 0, 0)));

  TsdfLayer tsdf_layer_4(voxel_size_m, MemoryType::kHost);
  tsdf_layer_4.allocateBlockAtIndex(Index3D(1, 1, 1));

  TsdfLayer tsdf_layer_5 = std::move(tsdf_layer_4);
  EXPECT_EQ(tsdf_layer_5.memory_type(), MemoryType::kHost);
  EXPECT_TRUE(tsdf_layer_5.isBlockAllocated(Index3D(1, 1, 1)));

  tsdf_layer_5 = std::move(tsdf_layer_1);
  EXPECT_EQ(tsdf_layer_5.memory_type(), MemoryType::kDevice);
  EXPECT_FALSE(tsdf_layer_5.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_FALSE(tsdf_layer_5.isBlockAllocated(Index3D(1, 1, 1)));
  tsdf_layer_5 = std::move(tsdf_layer_3);
  EXPECT_EQ(tsdf_layer_5.memory_type(), MemoryType::kDevice);
  EXPECT_TRUE(tsdf_layer_5.isBlockAllocated(Index3D(0, 0, 0)));
  EXPECT_FALSE(tsdf_layer_5.isBlockAllocated(Index3D(1, 1, 1)));
}

TEST(VoxelLayerTest, CopyVoxelsToHostFromUnified) {
  constexpr float voxel_size_m = 1.0;
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kUnified);
  auto block_ptr = tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 0));

  test_utils::setTsdfBlockVoxelsConstant(1.0f, block_ptr);

  const auto res = tsdf_layer.getVoxel(Vector3f(0.0f, 0.0f, 0.0f));

  EXPECT_TRUE(res.second);
  EXPECT_EQ(res.first.distance, 1.0f);
}

TEST(VoxelLayerTest, CopyLayerTest) {
  constexpr float voxel_size_m = 0.1f;

  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kDevice);

  // Create some blocks.
  std::vector<Index3D> all_blocks;
  all_blocks.push_back(Index3D(0, 0, 0));
  all_blocks.push_back(Index3D(1, 1, 1));
  all_blocks.push_back(Index3D(1, 2, 3));

  for (const Index3D& block_index : all_blocks) {
    auto block_ptr = tsdf_layer.allocateBlockAtIndex(block_index);
    test_utils::setTsdfBlockVoxelsConstant(block_index.norm(), block_ptr);
  }

  // Now copy over the layer with a deep copy operator.
  LOG(INFO) << "About to do first copy of layer";
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  tsdf_layer_host.copyFrom(tsdf_layer);
  EXPECT_EQ(tsdf_layer_host.memory_type(), MemoryType::kHost);

  std::vector<Index3D> all_blocks_host = tsdf_layer_host.getAllBlockIndices();
  EXPECT_EQ(all_blocks_host.size(), all_blocks.size());

  for (const Index3D& block_index : all_blocks) {
    auto block_ptr = tsdf_layer_host.getBlockAtIndex(block_index);
    ASSERT_TRUE(block_ptr);

    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
          EXPECT_NEAR(block_ptr->voxels[i][j][k].distance,
                      static_cast<float>(block_index.norm()), 1e-4f);
        }
      }
    }
  }

  // Now try the assignment. This triggers a second copy.
  TsdfLayer tsdf_layer_assignment(voxel_size_m, MemoryType::kHost);
  LOG(INFO) << "About to do second copy of layer";
  tsdf_layer_assignment.copyFrom(tsdf_layer);
  EXPECT_EQ(tsdf_layer_assignment.memory_type(), MemoryType::kHost);

  std::vector<Index3D> block_indices_assignment =
      tsdf_layer_assignment.getAllBlockIndices();
  EXPECT_EQ(block_indices_assignment.size(), all_blocks.size());

  for (const Index3D& block_index : all_blocks) {
    auto block_ptr = tsdf_layer_assignment.getBlockAtIndex(block_index);
    ASSERT_TRUE(block_ptr);

    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
          EXPECT_NEAR(block_ptr->voxels[i][j][k].distance,
                      static_cast<float>(block_index.norm()), 1e-4f);
        }
      }
    }
  }
}

TEST(VoxelLayerTest, ClearBlocks) {
  constexpr float voxel_size_m = 0.1f;

  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kDevice);

  // Create some blocks.
  tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 0));
  tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 1));
  tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 2));
  tsdf_layer.allocateBlockAtIndex(Index3D(0, 0, 3));
  EXPECT_EQ(tsdf_layer.numAllocatedBlocks(), 4);

  // Fail to clear non-allocated block
  tsdf_layer.clearBlocks({Index3D(0, 0, 4)});
  EXPECT_EQ(tsdf_layer.numAllocatedBlocks(), 4);

  // Clear 1 block
  tsdf_layer.clearBlocks({Index3D(0, 0, 0)});
  EXPECT_EQ(tsdf_layer.numAllocatedBlocks(), 3);

  // Clear 1 more block + 1 non-existant block
  tsdf_layer.clearBlocks({Index3D(0, 0, 1), Index3D(0, 0, 4)});
  EXPECT_EQ(tsdf_layer.numAllocatedBlocks(), 2);

  // Clear the rest
  tsdf_layer.clearBlocks(
      {Index3D(0, 0, 2), Index3D(0, 0, 3), Index3D(0, 0, 4)});
  EXPECT_EQ(tsdf_layer.numAllocatedBlocks(), 0);
}

TEST(VoxelLayerTest, AllocateMultipleBlocks) {
  constexpr float voxel_size_m = 0.1f;
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kDevice);

  const Index3D idx_1(0, 0, 0);
  const Index3D idx_2(0, 0, 1);
  const Index3D idx_3(0, 0, 2);
  const std::vector<Index3D> indices{idx_1, idx_2, idx_3};

  tsdf_layer.allocateBlocksAtIndices(indices, CudaStreamOwning());

  EXPECT_TRUE(tsdf_layer.isBlockAllocated(idx_1));
  EXPECT_TRUE(tsdf_layer.isBlockAllocated(idx_2));
  EXPECT_TRUE(tsdf_layer.isBlockAllocated(idx_3));
}

TEST(LayerTest, IsLayerTrait) {
  // NOTE(alexmillane): For some reason be have to assign to an intermediate
  // value for EXPECT_TRUE
  bool test_true = traits::are_layers<TsdfLayer, EsdfLayer, MeshLayer>::value;
  EXPECT_TRUE(test_true);
  bool test_false =
      traits::are_layers<TsdfLayer, EsdfLayer, MeshLayer, std::string>::value;
  EXPECT_FALSE(test_false);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
