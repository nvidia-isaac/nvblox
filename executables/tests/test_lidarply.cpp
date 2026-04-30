/*
Copyright 2025 NVIDIA CORPORATION

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

#include <filesystem>
#include <iostream>

#include "nvblox/datasets/lidarply_loader.h"
#include "nvblox/datasets/lidarply_writer.h"
#include "nvblox/fuser/fuser.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kTolerance = 1e-4;

class DatasetLidarPLYTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    // Clean up the test directories
    for (const auto& directory : created_directories_) {
      std::filesystem::remove_all(directory);
    }
  }

  void createTempDirectory(const std::string& path) {
    if (!std::filesystem::exists(path)) {
      std::filesystem::create_directories(path);
      created_directories_.push_back(path);
    }
  }

  std::string base_path_ = test_utils::getTestDataPath("data/lidarply/");
  std::vector<std::string> created_directories_;
};

TEST_F(DatasetLidarPLYTest, ParseTransform) {
  const std::string transform_filename =
      datasets::lidarply::internal::getPathForFramePose(base_path_, 1, 0);

  Transform T_L_C_test;

  ASSERT_TRUE(datasets::lidarply::internal::parsePoseFromFile(
      transform_filename, &T_L_C_test));
  // Expected values from frame-000000.pose.txt
  Eigen::Matrix4f T_L_C_mat;
  T_L_C_mat << 0.91145, -0.396816, 0.108608, -2.00566, 0.398171, 0.917258,
      0.00985408, -5.75096, -0.103532, 0.0342632, 0.994036, 0.72337, 0, 0, 0, 1;

  Transform T_L_C_true(T_L_C_mat);

  EXPECT_TRUE(T_L_C_test.isApprox(T_L_C_true, kTolerance));
}

TEST_F(DatasetLidarPLYTest, ParseLidarIntrinsics) {
  const std::string intrinsics_filename =
      datasets::lidarply::internal::getPathForLidarIntrinsics(base_path_);

  Lidar lidar_test;

  ASSERT_TRUE(datasets::lidarply::internal::parseLidarFromFile(
      intrinsics_filename, &lidar_test));

  // Expected values from lidar-intrinsics.txt
  // num_azimuth_divisions num_elevation_divisions min_valid_range_m
  // 2048 64 0.01
  EXPECT_EQ(lidar_test.num_azimuth_divisions(), 2048);
  EXPECT_EQ(lidar_test.num_elevation_divisions(), 64);
  EXPECT_NEAR(lidar_test.min_valid_range_m(), 0.01f, kTolerance);

  // Check that the lidar object has reasonable computed values
  EXPECT_GT(lidar_test.vertical_fov_rad(), 0.0f);
  EXPECT_LT(lidar_test.vertical_fov_rad(), 3.15f);  // Less than ~180 degrees
}

TEST_F(DatasetLidarPLYTest, GetPathFunctions) {
  // Test that path generation functions work correctly
  const std::string intrinsics_path =
      datasets::lidarply::internal::getPathForLidarIntrinsics(base_path_);
  EXPECT_EQ(intrinsics_path, base_path_ + "/lidar-intrinsics.txt");

  const std::string pose_path =
      datasets::lidarply::internal::getPathForFramePose(base_path_, 1, 0);
  EXPECT_EQ(pose_path, base_path_ + "/seq-01/frame-000000.pose.txt");

  const std::string pointcloud_path =
      datasets::lidarply::internal::getPathForPointcloud(base_path_, 1, 0);
  EXPECT_EQ(pointcloud_path,
            base_path_ + "/seq-01/frame-000000.pointcloud.ply");

  const std::string timestamp_path =
      datasets::lidarply::internal::getPathToFrameTimestampFile(base_path_, 1,
                                                                0);
  EXPECT_EQ(timestamp_path, base_path_ + "/seq-01/frame-000000.timestamp.txt");
}

TEST_F(DatasetLidarPLYTest, DataLoaderCreation) {
  // Test that the dataset loader can be created
  auto loader = datasets::lidarply::DataLoader::create(base_path_, 1);
  EXPECT_NE(loader, nullptr);
  EXPECT_TRUE(loader->setup_success());
}

TEST_F(DatasetLidarPLYTest, DataLoaderCreationInvalidPath) {
  // Test that the dataset loader fails gracefully with invalid path
  auto loader = datasets::lidarply::DataLoader::create("/nonexistent/path", 1);
  EXPECT_EQ(loader, nullptr);
}

TEST_F(DatasetLidarPLYTest, RunLidarPLYFuser) {
  auto fuser = datasets::lidarply::createFuser(base_path_, 1);
  EXPECT_NE(fuser, nullptr);
  const int result = fuser->run();
  EXPECT_EQ(result, 0);
}

TEST_F(DatasetLidarPLYTest, RoundTrip) {
  // Create a temporary directory for the roundtrip test
  const std::string temp_directory_name = "./temp_lidarply_files/";
  createTempDirectory(temp_directory_name);

  // Create test pointcloud
  Pointcloud original_pointcloud(MemoryType::kUnified);
  std::vector<Vector3f> test_points = {
      Vector3f(1.0f, 2.0f, 3.0f), Vector3f(4.0f, 5.0f, 6.0f),
      Vector3f(7.0f, 8.0f, 9.0f), Vector3f(-1.0f, -2.0f, -3.0f)};
  std::vector<Time> test_point_timestamps = {Time(10), Time(20), Time(30),
                                             Time(40)};
  CudaStreamOwning cuda_stream;
  original_pointcloud.copyPointsFromAsync(test_points, cuda_stream);
  original_pointcloud.copyTimestampsFromAsync(test_point_timestamps,
                                              cuda_stream);
  cuda_stream.synchronize();

  // Create test pose
  Eigen::Matrix4f T_L_C_mat;
  T_L_C_mat << 0.727611f, 0.682612f, 0.0679934f, -5.43125f, -0.674846f,
      0.73006f, -0.107682f, 0.371024f, -0.123144f, 0.0324654f, 0.991858f,
      0.715804f, 0.0f, 0.0f, 0.0f, 1.0f;
  Transform original_pose(T_L_C_mat);

  // Create test lidar intrinsics
  Lidar original_lidar(2048, 64, 0.01f, 0.28037f, 0.30263f);

  // Create test frame timestamp
  Time original_frame_timestamp_ms(1763555027421);

  // Write test data to disk
  LidarPlyWriter writer(temp_directory_name);
  ASSERT_TRUE(writer.writeNext(original_pointcloud, original_pose,
                               original_lidar, cuda_stream,
                               original_frame_timestamp_ms));
  // Write a second frame to provide next frame data needed to
  // compute lidar scan data (end pose and duration).
  Transform original_next_pose = original_pose;
  original_next_pose.translation() += Vector3f(0.1f, 0.0f, 0.0f);
  Time original_next_frame_timestamp_ms =
      original_frame_timestamp_ms + Time(100);
  ASSERT_TRUE(writer.writeNext(original_pointcloud, original_next_pose,
                               original_lidar, cuda_stream,
                               original_next_frame_timestamp_ms));

  // Load data back using DataLoader
  auto loader = datasets::lidarply::DataLoader::create(temp_directory_name, 1);
  ASSERT_NE(loader, nullptr);
  ASSERT_TRUE(loader->setup_success());

  Pointcloud loaded_pointcloud(MemoryType::kUnified);
  Transform loaded_pose, loaded_next_pose;
  Lidar loaded_lidar;
  Time loaded_frame_timestamp_ms, loaded_scan_duration_ms;

  auto result =
      loader->loadNext(&loaded_pointcloud, &loaded_pose, &loaded_lidar, nullptr,
                       nullptr, nullptr, &loaded_frame_timestamp_ms,
                       &loaded_next_pose, &loaded_scan_duration_ms);
  ASSERT_EQ(result, datasets::DataLoadResult::kSuccess);

  // Verify the data matches
  // 1. Check pointcloud size
  EXPECT_EQ(loaded_pointcloud.size(), original_pointcloud.size());

  // 2. Check pointcloud points (with tolerance for floating point errors)
  for (int i = 0; i < original_pointcloud.size(); i++) {
    EXPECT_NEAR(loaded_pointcloud.point(i).x(),
                original_pointcloud.point(i).x(), kTolerance);
    EXPECT_NEAR(loaded_pointcloud.point(i).y(),
                original_pointcloud.point(i).y(), kTolerance);
    EXPECT_NEAR(loaded_pointcloud.point(i).z(),
                original_pointcloud.point(i).z(), kTolerance);
  }

  // 3. Check point timestamps
  EXPECT_TRUE(original_pointcloud.timestamps_ms().has_value());
  EXPECT_TRUE(loaded_pointcloud.timestamps_ms().has_value());
  for (int i = 0; i < original_pointcloud.size(); i++) {
    EXPECT_EQ(loaded_pointcloud.timestamps_ms().value()[i],
              original_pointcloud.timestamps_ms().value()[i]);
  }

  // 4. Check pose
  EXPECT_TRUE(loaded_pose.isApprox(original_pose, kTolerance));

  // 5. Check lidar intrinsics
  EXPECT_TRUE(loaded_lidar == original_lidar);

  // 6. Check frame timestamp
  EXPECT_EQ(loaded_frame_timestamp_ms, original_frame_timestamp_ms);

  // 7. Check loaded next pose
  EXPECT_TRUE(loaded_next_pose.isApprox(original_next_pose, kTolerance));

  // 8. Check loaded scan duration
  EXPECT_EQ(loaded_scan_duration_ms, Time(100));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
