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

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/sensors/image.h"

#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/interpolation_2d_gpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

TEST(ColorImageTest, SetConstant) {
  // Set constant on the GPU
  ColorImage image(480, 640, MemoryType::kUnified);
  const Color color(255, 0, 0);
  test_utils::setImageConstantOnGpu(color, &image);
  // Check on the CPU
  for (int row_idx = 0; row_idx < image.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < image.cols(); col_idx++) {
      EXPECT_EQ(image(row_idx, col_idx), color);
    }
  }
}

TEST(ColorImageTest, NearbyImagesSimilar) {
  // This tests that two consequtive images in the 3DMatch dataset have small
  // pixelwise difference.

  // Load 3dmatch image
  const std::string base_path = "./data/3dmatch";
  constexpr int seq_id = 1;
  ColorImage image_1(MemoryType::kUnified);
  EXPECT_TRUE(datasets::load8BitColorImage(
      datasets::threedmatch::internal::getPathForColorImage(base_path, seq_id,
                                                            0),
      &image_1));
  ColorImage image_2(MemoryType::kUnified);
  EXPECT_TRUE(datasets::load8BitColorImage(
      datasets::threedmatch::internal::getPathForColorImage(base_path, seq_id,
                                                            1),
      &image_2));

  // Compute the diff image on the GPU
  ColorImage diff_image(MemoryType::kDevice);
  image::getDifferenceImageGPU(image_1, image_2, &diff_image);

  // Write diff image
  io::writeToPng("./color_image_difference.png", diff_image);

  // Check that there's not much difference between the images using the CPU
  // - 50 pixel values means constitutes a "big" difference for this test
  // - We allow up to 10% of pixels to have a "big" difference.
  constexpr float kBigDifferenceThreshold = 50.0;
  constexpr float kAllowableBigDifferencePixelsRatio = 0.10;
  CHECK_EQ(image_1.rows(), image_2.rows());
  CHECK_EQ(image_1.cols(), image_2.cols());
  int num_big_diff_pixels = 0;
  for (int i = 0; i < image_2.numel(); i++) {
    const float r_diff =
        static_cast<float>(image_1(i).r) - static_cast<float>(image_2(i).r);
    const float g_diff =
        static_cast<float>(image_1(i).g) - static_cast<float>(image_2(i).g);
    const float b_diff =
        static_cast<float>(image_1(i).b) - static_cast<float>(image_2(i).b);
    const float average_diff =
        (std::fabs(r_diff) + std::fabs(g_diff) + std::fabs(b_diff)) / 3.0f;
    if (average_diff > kBigDifferenceThreshold) {
      ++num_big_diff_pixels;
    }
  }
  const float big_diff_pixel_ratio = static_cast<float>(num_big_diff_pixels) /
                                     static_cast<float>(image_1.numel());
  std::cout << "num_big_diff_pixels: " << num_big_diff_pixels << std::endl;
  std::cout << "big_diff_pixel_ratio: " << big_diff_pixel_ratio << std::endl;
  EXPECT_LT(big_diff_pixel_ratio, kAllowableBigDifferencePixelsRatio);
}

TEST(ColorImageTest, InterpolationNearestNeighbour) {
  // Tiny image with col index in g channel, row index in 1.
  ColorImage image(2, 2, MemoryType::kUnified);
  image(0, 0) = Color(0, 0, 0);
  image(0, 1) = Color(0, 1, 0);
  image(1, 0) = Color(1, 0, 0);
  image(1, 1) = Color(1, 1, 0);

  // Note the switching in index order below. This is because images are set:
  //   image(row,col) = val
  // but interpolated:
  //   val = image(u_px(x,y)) = image(y,x)
  Color color;
  EXPECT_TRUE(interpolation::interpolate2D(
      image, Vector2f(0.4, 0.4), &color, InterpolationType::kNearestNeighbor));
  EXPECT_EQ(color.r, 0);
  EXPECT_EQ(color.g, 0);
  EXPECT_TRUE(interpolation::interpolate2D(
      image, Vector2f(0.4, 0.6), &color, InterpolationType::kNearestNeighbor));
  EXPECT_EQ(color.r, 1);
  EXPECT_EQ(color.g, 0);
  EXPECT_TRUE(interpolation::interpolate2D(
      image, Vector2f(0.6, 0.4), &color, InterpolationType::kNearestNeighbor));
  EXPECT_EQ(color.r, 0);
  EXPECT_EQ(color.g, 1);
  EXPECT_TRUE(interpolation::interpolate2D(
      image, Vector2f(0.6, 0.6), &color, InterpolationType::kNearestNeighbor));
  EXPECT_EQ(color.r, 1);
  EXPECT_EQ(color.g, 1);

  // Out of bounds
  EXPECT_FALSE(interpolation::interpolate2D(
      image, Vector2f(-0.6, 0.0), &color, InterpolationType::kNearestNeighbor));
  EXPECT_FALSE(interpolation::interpolate2D(
      image, Vector2f(0.0, -0.6), &color, InterpolationType::kNearestNeighbor));
  EXPECT_FALSE(
      interpolation::interpolate2D(image, Vector2f(-0.6, -0.6), &color,
                                   InterpolationType::kNearestNeighbor));
  EXPECT_FALSE(interpolation::interpolate2D(
      image, Vector2f(1.6, 0.0), &color, InterpolationType::kNearestNeighbor));
  EXPECT_FALSE(interpolation::interpolate2D(
      image, Vector2f(0.0, 1.6), &color, InterpolationType::kNearestNeighbor));
  EXPECT_FALSE(interpolation::interpolate2D(
      image, Vector2f(1.6, 1.6), &color, InterpolationType::kNearestNeighbor));
}

TEST(ColorImageTest, InterpolationLinear) {
  // Tiny image with col index in g channel, row index in 1. x10
  ColorImage image(2, 2, MemoryType::kUnified);
  image(0, 0) = Color(5, 5, 0);
  image(0, 1) = Color(5, 15, 0);
  image(1, 0) = Color(15, 5, 0);
  image(1, 1) = Color(15, 15, 0);

  Color color;
  EXPECT_TRUE(interpolation::interpolate2D(image, Vector2f(0.5, 0.5), &color,
                                           InterpolationType::kLinear));
  EXPECT_EQ(color.r, 5);
  EXPECT_EQ(color.g, 5);
  EXPECT_TRUE(interpolation::interpolate2D(image, Vector2f(0.5, 0.6), &color,
                                           InterpolationType::kLinear));
  EXPECT_EQ(color.r, 6);
  EXPECT_EQ(color.g, 5);
  EXPECT_TRUE(interpolation::interpolate2D(image, Vector2f(0.6, 0.5), &color,
                                           InterpolationType::kLinear));
  EXPECT_EQ(color.r, 5);
  EXPECT_EQ(color.g, 6);
  EXPECT_TRUE(interpolation::interpolate2D(image, Vector2f(0.6, 0.6), &color,
                                           InterpolationType::kLinear));
  EXPECT_EQ(color.r, 6);
  EXPECT_EQ(color.g, 6);

  // Out of bounds
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(0.4, 0.4), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(0.4, 0.6), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(0.6, 0.4), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(-0.6, 0.0), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(0.0, -0.6), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(-0.6, -0.6), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(1.6, 0.0), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(0.0, 1.6), &color,
                                            InterpolationType::kLinear));
  EXPECT_FALSE(interpolation::interpolate2D(image, Vector2f(1.6, 1.6), &color,
                                            InterpolationType::kLinear));
}

TEST(ColorImageTest, InterpolationGPU) {
  // Tiny image with col coord in g channel, row coord in r. x10
  ColorImage image(2, 2, MemoryType::kUnified);
  image(0, 0) = Color(5, 5, 0);
  image(0, 1) = Color(5, 15, 0);
  image(1, 0) = Color(15, 5, 0);
  image(1, 1) = Color(15, 15, 0);

  constexpr int kNumTests = 1000;
  std::vector<Eigen::Vector2f> u_px_vec;
  u_px_vec.reserve(kNumTests);
  for (int i = 0; i < kNumTests; i++) {
    u_px_vec.push_back(
        Eigen::Vector2f(test_utils::randomFloatInRange(0.5f, 1.5f),
                        test_utils::randomFloatInRange(0.5f, 1.5f)));
  }

  std::vector<Color> values(kNumTests, Color(0, 0, 0));
  std::vector<int> success_flags(kNumTests, 0);
  test_utils::linearInterpolateImageGpu(image, u_px_vec, &values,
                                        &success_flags);

  for (int i = 0; i < kNumTests; i++) {
    EXPECT_TRUE(success_flags[i] == 1);
    const uint8_t x =
        static_cast<uint8_t>(std::round((u_px_vec[i].x() * 10.0f)));
    const uint8_t y =
        static_cast<uint8_t>(std::round((u_px_vec[i].y() * 10.0f)));
    EXPECT_NEAR(values[i].g, x, kFloatEpsilon);
    EXPECT_NEAR(values[i].r, y, kFloatEpsilon);
  }
}

TEST(ColorImageTest, LoadedImageAlpha) {
  // Load 3dmatch image
  const std::string base_path = "./data/3dmatch";
  constexpr int seq_id = 1;
  ColorImage image(MemoryType::kUnified);
  EXPECT_TRUE(datasets::load8BitColorImage(
      datasets::threedmatch::internal::getPathForColorImage(base_path, seq_id,
                                                            0),
      &image));

  for (int row_idx = 0; row_idx < image.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < image.rows(); col_idx++) {
      CHECK_EQ(image(row_idx, col_idx).a, 255);
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
