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
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <filesystem>

#include "nvblox/utils/logging.h"

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/mapper/multi_mapper.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-6;

// MultiMapper child that gives the tests access to the internal
// functions.
class TestMultiMapper : public MultiMapper {
 public:
  TestMultiMapper(float voxel_size_m, MemoryType memory_type)
      : MultiMapper(voxel_size_m, MappingType::kHumanWithStaticTsdf,
                    EsdfMode::k3D, memory_type) {}
  FRIEND_TEST(MultiMapperTest, MaskOnAndOff);
};

TEST(MultiMapperTest, MaskOnAndOff) {
  // Load some 3DMatch data
  constexpr int kSeqID = 1;
  constexpr bool kMultithreadedLoading = false;
  auto data_loader = datasets::threedmatch::DataLoader::create(
      "./data/3dmatch", kSeqID, kMultithreadedLoading);
  EXPECT_TRUE(data_loader) << "Cant find the test input data.";

  DepthImage depth_frame(MemoryType::kDevice);
  ColorImage color_frame(MemoryType::kDevice);
  Transform T_L_C;
  Camera camera;
  Transform T_CM_CD = Transform::Identity();  // depth to mask camera transform
  data_loader->loadNext(&depth_frame, &T_L_C, &camera, &color_frame);

  // Two mappers - one with mask, one without
  constexpr float voxel_size_m = 0.05f;
  Mapper mapper(voxel_size_m, MemoryType::kUnified);
  // Multi mapper using T_CM_CD = identity to mimic the standard mapper
  TestMultiMapper multi_mapper(voxel_size_m, MemoryType::kUnified);
  // Do not consider occlusion
  multi_mapper.image_masker_.occlusion_threshold_m(
      std::numeric_limits<float>::max());

  // Make a mask where everything is masked out
  MonoImage mask_one(depth_frame.rows(), depth_frame.cols(),
                     MemoryType::kUnified);
  for (int row_idx = 0; row_idx < mask_one.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask_one.cols(); col_idx++) {
      mask_one(row_idx, col_idx) = 1;
    }
  }
  // Make a mask where nothing is masked out
  MonoImage mask_zero(MemoryType::kDevice);
  mask_zero.copyFrom(mask_one);
  mask_zero.setZero();

  // Depth masked out - expect nothing integrated
  multi_mapper.integrateDepth(depth_frame, mask_one, T_L_C, T_CM_CD, camera,
                              camera);
  EXPECT_EQ(multi_mapper.unmasked_mapper()->tsdf_layer().numAllocatedBlocks(),
            0);

  // Depth NOT masked out - expect same results as normal mapper
  mapper.integrateDepth(depth_frame, T_L_C, camera);
  multi_mapper.integrateDepth(depth_frame, mask_zero, T_L_C, T_CM_CD, camera,
                              camera);
  EXPECT_GT(mapper.tsdf_layer().numAllocatedBlocks(), 0);
  EXPECT_EQ(mapper.tsdf_layer().numAllocatedBlocks(),
            multi_mapper.unmasked_mapper()->tsdf_layer().numAllocatedBlocks());

  // Color masked out - expect blocks allocated but zero weight
  multi_mapper.integrateColor(color_frame, mask_one, T_L_C, camera);
  int num_non_zero_weight_voxels = 0;
  callFunctionOnAllVoxels<ColorVoxel>(
      multi_mapper.unmasked_mapper()->color_layer(),
      [&](const Index3D&, const Index3D&,
          const ColorVoxel* voxel) -> void {
        EXPECT_NEAR(voxel->weight, 0.0f, kFloatEpsilon);
        if (voxel->weight) {
          ++num_non_zero_weight_voxels;
        }
      });
  EXPECT_EQ(num_non_zero_weight_voxels, 0);

  // Color NOT masked out - expect same results as normal mapper
  mapper.integrateColor(color_frame, T_L_C, camera);
  multi_mapper.integrateColor(color_frame, mask_zero, T_L_C, camera);
  EXPECT_EQ(multi_mapper.unmasked_mapper()->color_layer().numAllocatedBlocks(),
            mapper.color_layer().numAllocatedBlocks());
  for (const Index3D& block_idx : mapper.color_layer().getAllBlockIndices()) {
    const auto block = mapper.color_layer().getBlockAtIndex(block_idx);
    const auto unmasked_block =
        multi_mapper.unmasked_mapper()->color_layer().getBlockAtIndex(
            block_idx);
    CHECK(block);
    CHECK(unmasked_block);
    for (int x_idx = 0; x_idx < ColorBlock::kVoxelsPerSide; x_idx++) {
      for (int y_idx = 0; y_idx < ColorBlock::kVoxelsPerSide; y_idx++) {
        for (int z_idx = 0; z_idx < ColorBlock::kVoxelsPerSide; z_idx++) {
          ColorVoxel voxel = block->voxels[x_idx][y_idx][z_idx];
          ColorVoxel unmasked_voxel =
              unmasked_block->voxels[x_idx][y_idx][z_idx];
          EXPECT_TRUE(voxel.color == unmasked_voxel.color);
          EXPECT_NEAR(voxel.weight, unmasked_voxel.weight, kFloatEpsilon);
          if (unmasked_voxel.weight > 0.0f) {
            ++num_non_zero_weight_voxels;
          }
        }
      }
    }
  }
  EXPECT_GT(num_non_zero_weight_voxels, 0);
  LOG(INFO) << "num_non_zero_weight_voxels: ";
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
