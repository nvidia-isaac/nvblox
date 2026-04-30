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

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/mapper/multi_mapper.h"
#include "nvblox/sensors/lidar.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-6;

// MultiMapper child that gives the tests access to the internal
// functions.
class TestMultiMapper : public MultiMapper {
 public:
  TestMultiMapper(float voxel_size_m, MemoryType memory_type)
      : MultiMapper(voxel_size_m, MappingType::kHumanWithStaticTsdf,
                    EsdfMode::k3D, memory_type) {
    // NOTE(alexmillane): In the test we have situations where we expect
    // different results from the same viewpoint so we turn off caching here.
    foreground_mapper_->tsdf_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    foreground_mapper_->color_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    foreground_mapper_->occupancy_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    background_mapper_->tsdf_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    background_mapper_->color_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    background_mapper_->occupancy_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
  }
  FRIEND_TEST(MultiMapperTest, MaskOnAndOff);
};

class MultiMapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);

    // NOTE(alexmillane): In the test we have situations where we expect
    // different results from exactly the same viewpoint so we turn off caching
    // here.
    mapper_.tsdf_integrator().view_calculator().cache_last_viewpoint(false);
    mapper_.color_integrator().view_calculator().cache_last_viewpoint(false);
    mapper_.occupancy_integrator().view_calculator().cache_last_viewpoint(
        false);

    multi_mapper_.foreground_mapper()
        ->tsdf_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    multi_mapper_.foreground_mapper()
        ->color_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    multi_mapper_.foreground_mapper()
        ->occupancy_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    multi_mapper_.background_mapper()
        ->tsdf_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    multi_mapper_.background_mapper()
        ->color_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
    multi_mapper_.background_mapper()
        ->occupancy_integrator()
        .view_calculator()
        .cache_last_viewpoint(false);
  }

  float voxel_size_m = 0.05f;

  Mapper mapper_ = Mapper(voxel_size_m, MemoryType::kUnified);
  TestMultiMapper multi_mapper_ =
      TestMultiMapper(voxel_size_m, MemoryType::kUnified);
};

TEST_F(MultiMapperTest, MaskOnAndOff) {
  // Load some 3DMatch data
  constexpr int kSeqID = 1;
  constexpr bool kMultithreadedLoading = false;
  auto data_loader = datasets::threedmatch::DataLoader::create(
      test_utils::getTestDataPath("data/3dmatch"), kSeqID,
      kMultithreadedLoading);
  EXPECT_TRUE(data_loader) << "Cant find the test input data.";

  DepthImage depth_frame(MemoryType::kDevice);
  ColorImage color_frame(MemoryType::kDevice);
  Transform T_L_C;
  Camera camera;
  Transform T_CM_CD = Transform::Identity();  // depth to mask camera transform
  data_loader->loadNext(&depth_frame, &T_L_C, &camera, &color_frame);

  // Do not consider occlusion
  multi_mapper_.image_masker_.occlusion_threshold_m(
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
  mask_zero.setZeroAsync(CudaStreamOwning());

  // Depth masked out - expect nothing integrated
  multi_mapper_.integrateDepth(depth_frame, mask_one, T_L_C, T_CM_CD, camera,
                               camera);
  EXPECT_EQ(multi_mapper_.background_mapper()->tsdf_layer().numBlocks(), 0);

  // Depth NOT masked out - expect same results as normal mapper
  mapper_.integrateDepth(depth_frame, T_L_C, camera);
  multi_mapper_.integrateDepth(depth_frame, mask_zero, T_L_C, T_CM_CD, camera,
                               camera);
  EXPECT_GT(mapper_.tsdf_layer().numBlocks(), 0);
  EXPECT_EQ(mapper_.tsdf_layer().numBlocks(),
            multi_mapper_.background_mapper()->tsdf_layer().numBlocks());

  // Color masked out - expect blocks allocated but zero weight
  multi_mapper_.integrateColor(color_frame, mask_one, T_L_C, camera);
  int num_non_zero_weight_voxels = 0;
  callFunctionOnAllVoxels<ColorVoxel>(
      multi_mapper_.background_mapper()->color_layer(),
      [&](const Index3D&, const Index3D&, const ColorVoxel* voxel) -> void {
        EXPECT_NEAR(voxel->weight, 0.0f, kFloatEpsilon);
        if (voxel->weight) {
          ++num_non_zero_weight_voxels;
        }
      });
  EXPECT_EQ(num_non_zero_weight_voxels, 0);

  // Color NOT masked out - expect same results as normal mapper
  mapper_.integrateColor(color_frame, T_L_C, camera);
  multi_mapper_.integrateColor(color_frame, mask_zero, T_L_C, camera);
  EXPECT_EQ(multi_mapper_.background_mapper()->color_layer().numBlocks(),
            mapper_.color_layer().numBlocks());
  for (const Index3D& block_idx : mapper_.color_layer().getAllBlockIndices()) {
    const auto block = mapper_.color_layer().getBlockAtIndex(block_idx);
    const auto background_block =
        multi_mapper_.background_mapper()->color_layer().getBlockAtIndex(
            block_idx);
    CHECK(block);
    CHECK(background_block);
    for (int x_idx = 0; x_idx < ColorBlock::kVoxelsPerSide; x_idx++) {
      for (int y_idx = 0; y_idx < ColorBlock::kVoxelsPerSide; y_idx++) {
        for (int z_idx = 0; z_idx < ColorBlock::kVoxelsPerSide; z_idx++) {
          ColorVoxel voxel = block->voxels[x_idx][y_idx][z_idx];
          ColorVoxel background_voxel =
              background_block->voxels[x_idx][y_idx][z_idx];
          EXPECT_TRUE(voxel.color == background_voxel.color);
          EXPECT_NEAR(voxel.weight, background_voxel.weight, kFloatEpsilon);
          if (background_voxel.weight > 0.0f) {
            ++num_non_zero_weight_voxels;
          }
        }
      }
    }
  }
  EXPECT_GT(num_non_zero_weight_voxels, 0);
  LOG(INFO) << "num_non_zero_weight_voxels: ";
}

TEST_F(MultiMapperTest, IntegratePointcloud) {
  // Create a simple pointcloud
  Pointcloud pointcloud(MemoryType::kUnified);
  std::vector<Vector3f> points_host = {
      Vector3f(1.0f, 0.0f, 0.0f),  Vector3f(2.0f, 0.0f, 0.0f),
      Vector3f(3.0f, 0.0f, 0.0f),  Vector3f(1.5f, 0.5f, 0.0f),
      Vector3f(2.5f, -0.5f, 0.0f), Vector3f(2.0f, 1.0f, 0.0f),
      Vector3f(2.0f, -1.0f, 0.0f), Vector3f(3.0f, 0.5f, 0.5f)};
  pointcloud.copyPointsFromAsync(points_host, CudaStreamOwning());

  // Create a Lidar sensor with valid dimensions
  const int num_azimuth_divisions = 1024;
  const int num_elevation_divisions = 16;
  const float min_valid_range_m = 0.1f;
  const float vertical_fov_rad = 30.0f * M_PI / 180.0f;
  Lidar lidar(num_azimuth_divisions, num_elevation_divisions, min_valid_range_m,
              vertical_fov_rad);

  // Integrate pointcloud using MultiMapper without motion compensation
  Transform T_L_S = Transform::Identity();
  bool use_lidar_motion_compensation = false;
  multi_mapper_.integrateDepth(pointcloud, T_L_S, lidar,
                               use_lidar_motion_compensation);

  // Verify integration worked - should have allocated some blocks
  // MultiMapper integrates into the background mapper by default
  const int num_blocks_after_first =
      multi_mapper_.background_mapper()->tsdf_layer().numBlocks();
  EXPECT_GT(num_blocks_after_first, 0);

  // Integrate pointcloud again with motion compensation from a different
  // location Motion compensation requires timestamps for each point
  const int64_t scan_duration_ms_value = 100;
  Time scan_duration_ms(scan_duration_ms_value);
  std::vector<Time> timestamps_ms(points_host.size());
  for (size_t i = 0; i < points_host.size(); i++) {
    // Spread timestamps evenly across scan duration
    timestamps_ms[i] = Time(i * scan_duration_ms_value / points_host.size());
  }
  pointcloud.copyTimestampsFromAsync(timestamps_ms, CudaStreamOwning());

  // Start from a different position to cover new space
  Transform T_L_S_scanStart2 = Transform::Identity();
  T_L_S_scanStart2.translate(Vector3f(0.0f, 3.0f, 0.0f));  // Start 3m away in y
  Transform T_L_S_scanEnd2 = Transform::Identity();
  T_L_S_scanEnd2.translate(Vector3f(0.0f, 4.0f, 0.0f));  // End 4m away in y
  use_lidar_motion_compensation = true;
  multi_mapper_.integrateDepth(pointcloud, T_L_S_scanStart2, lidar,
                               use_lidar_motion_compensation, T_L_S_scanEnd2,
                               scan_duration_ms);

  // Verify more blocks were allocated after second integration
  const int num_blocks_after_second =
      multi_mapper_.background_mapper()->tsdf_layer().numBlocks();
  EXPECT_GT(num_blocks_after_second, num_blocks_after_first);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
