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

#include "nvblox/sensors/pointcloud.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

TEST(PointcloudTest, SetAndAccess) {
  constexpr int kNumOfPoints = 10;
  Pointcloud pointcloud(kNumOfPoints, MemoryType::kUnified);
  EXPECT_EQ(pointcloud.memory_type(), MemoryType::kUnified);

  CHECK_EQ(pointcloud.size(), kNumOfPoints);

  // Write points
  for (int i = 0; i < kNumOfPoints; i++) {
    pointcloud(i) = Vector3f(static_cast<float>(i), 0.0f, 0.0f);
  }

  // Read points
  for (int i = 0; i < kNumOfPoints; i++) {
    EXPECT_EQ(i, static_cast<int>(pointcloud(i).x()));
  }
}

TEST(PointcloudTest, Copy) {
  // Dummy data.
  constexpr int kNumOfPoints = 10;
  Pointcloud pointcloud(kNumOfPoints, MemoryType::kUnified);
  for (int i = 0; i < kNumOfPoints; i++) {
    pointcloud(i) = Vector3f(static_cast<float>(i), 0.0f, 0.0f);
  }
  EXPECT_EQ(pointcloud.memory_type(), MemoryType::kUnified);

  // Copy
  Pointcloud pointcloud_copy_1(MemoryType::kUnified);
  pointcloud_copy_1.copyFrom(pointcloud);
  Pointcloud pointcloud_copy_2(MemoryType::kUnified);
  pointcloud_copy_2.copyFrom(pointcloud);
  EXPECT_EQ(pointcloud_copy_1.memory_type(), MemoryType::kUnified);
  EXPECT_EQ(pointcloud_copy_2.memory_type(), MemoryType::kUnified);

  // Set the original to zero
  pointcloud.setZero();

  // Read points
  for (int i = 0; i < kNumOfPoints; i++) {
    EXPECT_EQ(i, static_cast<int>(pointcloud_copy_1(i).x()));
  }
  for (int i = 0; i < kNumOfPoints; i++) {
    EXPECT_EQ(i, static_cast<int>(pointcloud_copy_2(i).x()));
  }
}

TEST(PointcloudTest, DeviceCopy) {
  constexpr int kNumOfPoints = 10;
  Pointcloud pointcloud(kNumOfPoints, MemoryType::kHost);
  EXPECT_EQ(pointcloud.memory_type(), MemoryType::kHost);

  // Test point
  Vector3f test_point(1.0, 2.0, 3.0);

  // Push it into host pointcloud
  for (int i = 0; i < kNumOfPoints; i++) {
    pointcloud(i) = test_point;
  }

  // Pointcloud to the device
  Pointcloud pointcloud_gpu(MemoryType::kDevice);
  pointcloud_gpu.copyFrom(pointcloud);
  EXPECT_EQ(pointcloud_gpu.memory_type(), MemoryType::kDevice);

  // Check that the data made it.
  std::vector<Vector3f> points_on_host(kNumOfPoints);
  checkCudaErrors(
      cudaMemcpy(points_on_host.data(), pointcloud_gpu.dataConstPtr(),
                 sizeof(Vector3f) * kNumOfPoints, cudaMemcpyDeviceToHost));

  for (int i = 0; i < kNumOfPoints; i++) {
    EXPECT_TRUE(points_on_host[i] == test_point);
  }
}

TEST(PointcloudTest, ConstructFromVector) {
  // Dummy points
  constexpr int kNumPoints = 10;
  std::vector<Vector3f> points_vec;
  for (int i = 0; i < kNumPoints; i++) {
    points_vec.push_back(Vector3f(i, 0, 0));
  }

  // To kUnified Pointcloud
  Pointcloud pointcloud(MemoryType::kUnified);
  pointcloud.copyFrom(points_vec);
  EXPECT_EQ(pointcloud.memory_type(), MemoryType::kUnified);

  // Check
  CHECK_EQ(pointcloud.size(), kNumPoints);
  for (int i = 0; i < kNumPoints; i++) {
    EXPECT_TRUE(pointcloud(i) == Vector3f(i, 0, 0));
  }
}

TEST(PointcloudTest, TransformPointcloudOnGpu) {
  // NOTE(alexmillane): A line of points at x = 5, and z = [-5,..., 5]

  // Points
  Pointcloud pointcloud_A(MemoryType::kUnified);
  for (int z = -5; z <= 5; z++) {
    pointcloud_A.points().push_back(Vector3f(5, 0, z));
  }
  EXPECT_EQ(pointcloud_A.memory_type(), MemoryType::kUnified);

  // Rotate 90 degrees counter clock-wise around z
  Transform T_B_A = Transform::Identity();
  T_B_A.prerotate(
      Eigen::AngleAxisf(M_PI / 2.0f, Vector3f::UnitZ()).toRotationMatrix());

  Pointcloud pointcloud_B(MemoryType::kUnified);
  transformPointcloudOnGPU(T_B_A, pointcloud_A, &pointcloud_B);
  EXPECT_EQ(pointcloud_B.memory_type(), MemoryType::kUnified);

  constexpr float kEpsilon = 1e-4;
  for (int i = 0; i < pointcloud_B.size(); i++) {
    const Vector3f& p_B = pointcloud_B(i);
    EXPECT_NEAR(p_B.x(), 0.0f, kEpsilon);
    EXPECT_NEAR(p_B.y(), 5.0f, kEpsilon);
    EXPECT_NEAR(p_B.z(), i - 5.0f, kEpsilon);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
