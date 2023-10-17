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
#include <stdio.h>

#include "nvblox/io/layer_cake_io.h"
#include "nvblox/map/layer.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/serialization/internal/layer_serialization.h"
#include "nvblox/serialization/internal/layer_type_register.h"
#include "nvblox/serialization/internal/serializer.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

class SerializationTest : public ::testing::Test {
 protected:
  SerializationTest() {
    constexpr static float scene_sphere_radius = 2.0f;
    const Vector3f scene_sphere_center = Vector3f(0.0f, 0.0f, 2.0f);

    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-6.0f, -6.0f, -1.0f),
                                           Vector3f(6.0f, 6.0f, 6.0f));
    scene_.addGroundLevel(0.0f);
    scene_.addCeiling(5.0f);
    scene_.addPrimitive(std::make_unique<primitives::Sphere>(
        scene_sphere_center, scene_sphere_radius));
    scene_.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
  }

  // Layers
  constexpr static float voxel_size_m_ = 0.05f;
  constexpr static float truncation_distance = 0.15f;

  LayerCake cake_;

  primitives::Scene scene_;
};

TEST_F(SerializationTest, OpenInvalidFile) {
  Serializer serializer;

  ASSERT_FALSE(serializer.open("./this_file_is_not_real.nvblox", std::ios::in));
  EXPECT_FALSE(serializer.valid());
}

TEST_F(SerializationTest, SerializeAndDeserializeTsdfLayer) {
  // This creates an in-memory database that can be shared within the process.
  const std::string test_filename = "file:test1?mode=memory&cache=shared";
  // Create an empty Serializer object.
  Serializer serializer;

  // Create a TSDF layer from the scene.
  cake_ = LayerCake::create<TsdfLayer>(voxel_size_m_, MemoryType::kHost);
  scene_.generateLayerFromScene(truncation_distance, cake_.getPtr<TsdfLayer>());

  // Make sure we can open a file.
  // REMOVE the file.
  ASSERT_TRUE(serializer.open(test_filename, std::ios::out));
  EXPECT_TRUE(serializer.valid());

  EXPECT_TRUE(serializer.writeLayerCake(cake_, CudaStreamOwning()));

  LayerCake cake2 =
      serializer.loadLayerCake(MemoryType::kHost, CudaStreamOwning());

  EXPECT_EQ(cake2.voxel_size(), cake_.voxel_size());

  EXPECT_FALSE(cake2.empty());
  EXPECT_TRUE(cake2.exists<TsdfLayer>());
  EXPECT_EQ(cake_.getPtr<TsdfLayer>()->numAllocatedBlocks(),
            cake2.getPtr<TsdfLayer>()->numAllocatedBlocks());

  // Compare the contents of all blocks in the TSDF layer.
  std::vector<Index3D> all_blocks =
      cake_.getPtr<TsdfLayer>()->getAllBlockIndices();

  for (const Index3D& index : all_blocks) {
    auto block1 = cake_.getPtr<TsdfLayer>()->getBlockAtIndex(index);
    auto block2 = cake2.getPtr<TsdfLayer>()->getBlockAtIndex(index);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
      for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
        for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
          EXPECT_EQ(block1->voxels[x][y][z].distance,
                    block2->voxels[x][y][z].distance);
          EXPECT_EQ(block1->voxels[x][y][z].weight,
                    block2->voxels[x][y][z].weight);
        }
      }
    }
  }

  // And close it again.
  ASSERT_TRUE(serializer.close());
  EXPECT_FALSE(serializer.valid());
}

TEST_F(SerializationTest, SerializeLayerParameters) {
  // This creates an in-memory database that can be shared within the process.
  const std::string test_filename = "file:test2?mode=memory&cache=shared";

  const std::string layer_name = "awesome_layer";
  const std::string string_param_name = "stringy_parameter";
  const std::string float_param_name = "floaty_parameter";
  const std::string int_param_name = "inty_parameter";

  const std::string string_param_value = "aaaaa";
  const float float_param_value = -0.25f;
  const int int_param_value = -5;

  // Create an empty Serializer object.
  Serializer serializer;

  // Create a parameter value struct.
  LayerParameterStruct param_struct;
  param_struct.string_params.emplace(string_param_name, string_param_value);
  param_struct.int_params.emplace(int_param_name, int_param_value);
  param_struct.float_params.emplace(float_param_name, float_param_value);

  // Make sure we can open a file.
  ASSERT_TRUE(serializer.open(test_filename, std::ios::out));
  EXPECT_TRUE(serializer.valid());

  EXPECT_TRUE(serializer.createLayerTables(layer_name));

  EXPECT_TRUE(serializer.setLayerParameters(layer_name, param_struct));

  LayerParameterStruct result_param_struct;

  EXPECT_TRUE(serializer.getLayerParameters(layer_name, &result_param_struct));

  // Check that all the params are there.
  EXPECT_EQ(result_param_struct.string_params[string_param_name],
            string_param_value);
  EXPECT_EQ(result_param_struct.int_params[int_param_name], int_param_value);
  EXPECT_EQ(result_param_struct.float_params[float_param_name],
            float_param_value);

  // Make sure there's no extra spare parameters.
  EXPECT_EQ(result_param_struct.string_params.size(), 1);
  EXPECT_EQ(result_param_struct.int_params.size(), 1);
  EXPECT_EQ(result_param_struct.float_params.size(), 1);

  // And close it again.
  EXPECT_TRUE(serializer.close());
  EXPECT_FALSE(serializer.valid());

  remove(test_filename.c_str());
}

TEST_F(SerializationTest, PopulateLayerParameterStruct) {
  // Create a TSDF layer from the scene.
  cake_ = LayerCake::create<TsdfLayer>(voxel_size_m_, MemoryType::kHost);
  scene_.generateLayerFromScene(truncation_distance, cake_.getPtr<TsdfLayer>());

  LayerParameterStruct params =
      serializeLayerParameters(*cake_.getConstPtr<TsdfLayer>());

  EXPECT_EQ(params.float_params["block_size"],
            cake_.getConstPtr<TsdfLayer>()->block_size());
}

TEST_F(SerializationTest, LayerBlockSerialization) {
  // Create a TSDF layer from the scene.
  cake_ = LayerCake::create<TsdfLayer>(voxel_size_m_, MemoryType::kHost);
  scene_.generateLayerFromScene(truncation_distance, cake_.getPtr<TsdfLayer>());

  const TsdfLayer& tsdf_layer = *cake_.getConstPtr<TsdfLayer>();

  std::vector<Index3D> indices = getLayerDataIndices(tsdf_layer);

  EXPECT_EQ(indices.size(), tsdf_layer.numAllocatedBlocks());

  std::vector<Byte> block_byte_string =
      serializeLayerDataAtIndex(tsdf_layer, indices[0], CudaStreamOwning());

  EXPECT_NE(block_byte_string.size(), 0);
}

TEST_F(SerializationTest, SerializeDeviceBlock) {
  cake_ = LayerCake::create<TsdfLayer>(voxel_size_m_, MemoryType::kHost);
  scene_.generateLayerFromScene(truncation_distance, cake_.getPtr<TsdfLayer>());
  const TsdfLayer& tsdf_layer = *cake_.getConstPtr<TsdfLayer>();

  TsdfLayer::BlockType::ConstPtr block =
      tsdf_layer.getBlockAtIndex(Index3D::Zero());

  std::vector<Byte> block_byte_string =
      serializeBlock(block, CudaStreamOwning());
  EXPECT_NE(block_byte_string.size(), 0);

  // Create a device block.
  TsdfLayer::BlockType::ConstPtr block_device =
      block.clone(MemoryType::kDevice);

  std::vector<Byte> device_block_byte_string =
      serializeBlock(block_device, CudaStreamOwning());

  EXPECT_NE(device_block_byte_string.size(), 0);
  ASSERT_EQ(block_byte_string.size(), device_block_byte_string.size());

  for (size_t i = 0; i < block_byte_string.size(); i++) {
    EXPECT_EQ(block_byte_string[i], device_block_byte_string[i]);
  }

  unified_ptr<TsdfBlock> deserialized_block =
      make_unified<TsdfBlock>(MemoryType::kDevice);

  deserializeBlock(device_block_byte_string, deserialized_block,
                   CudaStreamOwning());

  auto deserialized_block_host = deserialized_block.clone(MemoryType::kHost);

  for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
    for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
      for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
        EXPECT_EQ(block->voxels[x][y][z].distance,
                  deserialized_block_host->voxels[x][y][z].distance);
        EXPECT_EQ(block->voxels[x][y][z].weight,
                  deserialized_block_host->voxels[x][y][z].weight);
      }
    }
  }
}

TEST_F(SerializationTest, OverwriteTest) {
  // Idea here is to save twice. If overwriting is working, only the blocks from
  // the second save will show up in the reloaded map.

  cake_ = LayerCake::create<TsdfLayer>(voxel_size_m_, MemoryType::kHost);
  TsdfLayer& tsdf_layer = *cake_.getPtr<TsdfLayer>();
  const std::string filename = "overwrite_test_layer.nvblx";

  // First save
  tsdf_layer.allocateBlockAtIndex(Index3D::Zero());
  EXPECT_TRUE(io::writeLayerCakeToFile(filename, cake_));

  // Clear layer back to zero.
  tsdf_layer.clear();

  // Second save
  tsdf_layer.allocateBlockAtIndex(Index3D::Ones());
  EXPECT_EQ(cake_.get<TsdfLayer>().numAllocatedBlocks(), 1);
  EXPECT_TRUE(io::writeLayerCakeToFile(filename, cake_));

  // Reload
  LayerCake cake_reloaded =
      io::loadLayerCakeFromFile(filename, MemoryType::kDevice);

  // Expect that there is only a single block in the saved map.
  EXPECT_EQ(cake_reloaded.get<TsdfLayer>().numAllocatedBlocks(), 1);

  // Expect that the allocated block is the one allocated in the SECOND
  // allocation.
  const auto block_idx_vector =
      cake_reloaded.get<TsdfLayer>().getAllBlockIndices();
  EXPECT_TRUE((block_idx_vector[0].array() == Index3D::Ones().array()).all());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
