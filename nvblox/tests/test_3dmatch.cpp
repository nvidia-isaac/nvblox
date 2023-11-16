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

#include <iostream>

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/sensors/image.h"

// DEBUG
#include <chrono>
#include <thread>

using namespace nvblox;

constexpr float kTolerance = 1e-4;

class Dataset3DMatchTest : public ::testing::Test {
 protected:
  void SetUp() override { base_path_ = "./data/3dmatch/"; }

  std::string base_path_;
};

TEST_F(Dataset3DMatchTest, ParseTransform) {
  const std::string transform_filename =
      datasets::threedmatch::internal::getPathForFramePose(base_path_, 1, 0);

  Transform T_L_C_test;

  ASSERT_TRUE(datasets::threedmatch::internal::parsePoseFromFile(
      transform_filename, &T_L_C_test));

  Eigen::Matrix4f T_L_C_mat;
  T_L_C_mat << 3.13181000e-01, 3.09473000e-01, -8.97856000e-01, 1.97304600e+00,
      -8.73910000e-02, -9.32015000e-01, -3.51729000e-01, 1.12573400e+00,
      -9.45665000e-01, 1.88620000e-01, -2.64844000e-01, 3.09820000e-01,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;

  Transform T_L_C_true(T_L_C_mat);

  EXPECT_TRUE(T_L_C_test.isApprox(T_L_C_true));
}

TEST_F(Dataset3DMatchTest, ParseCameraIntrinsics) {
  const std::string intrinsics_filename =
      datasets::threedmatch::internal::getPathForCameraIntrinsics(base_path_);

  Eigen::Matrix3f intrinsics_test;

  ASSERT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      intrinsics_filename, &intrinsics_test));

  Eigen::Matrix3f intrinsics_true;
  intrinsics_true << 5.70342205e+02, 0.00000000e+00, 3.20000000e+02,
      0.00000000e+00, 5.70342205e+02, 2.40000000e+02, 0.00000000e+00,
      0.00000000e+00, 1.00000000e+00;

  EXPECT_TRUE(intrinsics_test.isApprox(intrinsics_true));
}

TEST_F(Dataset3DMatchTest, LoadImage) {
  const std::string image_filename =
      datasets::threedmatch::internal::getPathForDepthImage(base_path_, 1, 0);

  DepthImage image(MemoryType::kDevice);
  ASSERT_TRUE(datasets::load16BitDepthImage(image_filename, &image));

  EXPECT_EQ(image.rows(), 480);
  EXPECT_EQ(image.cols(), 640);

  EXPECT_NEAR(image::minGPU(image), 0.0, kTolerance);
  EXPECT_NEAR(image::maxGPU(image), 7.835, kTolerance);
}

enum class LoaderType { kSingleThreaded, kMultiThreaded };

class LoaderParameterizedTest
    : public Dataset3DMatchTest,
      public ::testing::WithParamInterface<LoaderType> {
  //
};

TEST_P(LoaderParameterizedTest, ImageLoaderObject) {
  const LoaderType loader_type = GetParam();
  constexpr int seq_id = 1;
  std::unique_ptr<datasets::ImageLoader<DepthImage>> depth_loader_ptr;
  if (loader_type == LoaderType::kSingleThreaded) {
    depth_loader_ptr = datasets::threedmatch::internal::createDepthImageLoader(
        base_path_, seq_id, false);
  } else {
    depth_loader_ptr = datasets::threedmatch::internal::createDepthImageLoader(
        base_path_, seq_id, true);
  }

  DepthImage depth_image_1(MemoryType::kDevice);
  DepthImage depth_image_2(MemoryType::kDevice);
  DepthImage depth_image_3(MemoryType::kDevice);
  DepthImage depth_image_4(MemoryType::kDevice);
  EXPECT_TRUE(depth_loader_ptr->getNextImage(&depth_image_1));
  EXPECT_TRUE(depth_loader_ptr->getNextImage(&depth_image_2));
  EXPECT_TRUE(depth_loader_ptr->getNextImage(&depth_image_3));
  EXPECT_FALSE(depth_loader_ptr->getNextImage(&depth_image_4));

  EXPECT_EQ(depth_image_1.rows(), 480);
  EXPECT_EQ(depth_image_1.cols(), 640);
  EXPECT_NEAR(image::minGPU(depth_image_1), 0.0, kTolerance);
  EXPECT_NEAR(image::maxGPU(depth_image_1), 7.835, kTolerance);
}

INSTANTIATE_TEST_CASE_P(LoaderTests, LoaderParameterizedTest,
                        ::testing::Values(LoaderType::kSingleThreaded,
                                          LoaderType::kMultiThreaded));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
