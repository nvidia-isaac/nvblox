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

#include "nvblox/core/image.h"
#include "nvblox/core/interpolation_2d.h"
#include "nvblox/core/types.h"

#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/interpolation_2d_gpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

class DepthImageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Uninitialized depth frame
    depth_frame_ = DepthImage(rows_, cols_, MemoryType::kUnified);
  }

  Index2D getRandomPixel() {
    return Index2D(test_utils::randomIntInRange(0., cols_ - 1),
                   test_utils::randomIntInRange(0, rows_ - 1));
  }

  int rows_ = 480;
  int cols_ = 640;
  DepthImage depth_frame_;
};

void setImageConstantOnCpu(const float value, DepthImage* depth_frame_ptr) {
  // Set everything to 1.0 through one access method and check through the other
  for (int row_idx = 0; row_idx < depth_frame_ptr->rows(); row_idx++) {
    for (int col_idx = 0; col_idx < depth_frame_ptr->cols(); col_idx++) {
      (*depth_frame_ptr)(row_idx, col_idx) = 1.0f;
    }
  }
}

TEST_F(DepthImageTest, Host) {
  // Set constant on CPU
  setImageConstantOnCpu(1.0, &depth_frame_);

  // Check on the CPU
  for (int lin_idx = 0; lin_idx < depth_frame_.numel(); lin_idx++) {
    EXPECT_EQ(depth_frame_(lin_idx), 1.0f);
  }
}

TEST_F(DepthImageTest, Device) {
  // Set constant on GPU
  constexpr float kPixelValue = 1.0f;
  test_utils::setImageConstantOnGpu(kPixelValue, &depth_frame_);

  // Check on the CPU
  for (int lin_idx = 0; lin_idx < depth_frame_.numel(); lin_idx++) {
    EXPECT_EQ(depth_frame_(lin_idx), 1.0f);
  }
}

TEST_F(DepthImageTest, DeviceReduction) {
  // Make sure this is deterministic.
  std::srand(0);

  // Set constant on CPU
  constexpr float kPixelValue = 1.0f;
  setImageConstantOnCpu(kPixelValue, &depth_frame_);

  // Change a single value on the CPU
  constexpr float kMaxValue = 100.0f;
  constexpr float kMinValue = -100.0f;
  const Index2D u_max = getRandomPixel();
  Index2D u_min = u_max;
  while ((u_min.array() == u_max.array()).all()) {
    u_min = getRandomPixel();
  }
  depth_frame_(u_max.y(), u_max.x()) = kMaxValue;
  depth_frame_(u_min.y(), u_min.x()) = kMinValue;

  // Reduction on the GPU
  const float max = image::max(depth_frame_);
  const float min = image::min(depth_frame_);
  const auto minmax = image::minmax(depth_frame_);

  // Check on the CPU
  EXPECT_EQ(max, kMaxValue);
  EXPECT_EQ(min, kMinValue);
  EXPECT_EQ(minmax.first, kMinValue);
  EXPECT_EQ(minmax.second, kMaxValue);
}

TEST_F(DepthImageTest, GpuOperation) {
  // Set constant on CPU
  constexpr float kPixelValue = 1.0f;
  setImageConstantOnCpu(kPixelValue, &depth_frame_);

  // Element wise min
  image::elementWiseMinInPlace(0.5f, &depth_frame_);

  // Reduction on the GPU
  const float max = image::max(depth_frame_);
  EXPECT_EQ(max, 0.5f);

  // Element wise max
  image::elementWiseMaxInPlace(1.5f, &depth_frame_);

  const float min = image::min(depth_frame_);
  EXPECT_EQ(min, 1.5f);
}

TEST_F(DepthImageTest, LinearInterpolation) {
  // The images {depth_frame_col_coords, depth_frame_row_coords} are set up such
  // that if you interpolate, you should get the interpolated position back.
  DepthImage depth_frame_col_coords(rows_, cols_, MemoryType::kUnified);
  DepthImage depth_frame_row_coords(rows_, cols_, MemoryType::kUnified);
  for (int col_idx = 0; col_idx < cols_; col_idx++) {
    for (int row_idx = 0; row_idx < rows_; row_idx++) {
      depth_frame_col_coords(row_idx, col_idx) =
          static_cast<float>(col_idx) + 0.5f;
      depth_frame_row_coords(row_idx, col_idx) =
          static_cast<float>(row_idx) + 0.5f;
    }
  }
  constexpr int kNumTests = 1000;
  // int num_failures = 0;
  for (int i = 0; i < kNumTests; i++) {
    // Random pixel location on image plane
    const Vector2f u_px(test_utils::randomFloatInRange(
                            0.5f, static_cast<float>(cols_ - 1) + 0.5f),
                        test_utils::randomFloatInRange(
                            0.5f, static_cast<float>(rows_ - 1) + 0.5f));
    // Interpolate x and y grids
    float interpolated_value_col;
    EXPECT_TRUE(interpolation::interpolate2DLinear(depth_frame_col_coords, u_px,
                                                   &interpolated_value_col));
    float interpolated_value_row;
    EXPECT_TRUE(interpolation::interpolate2DLinear(depth_frame_row_coords, u_px,
                                                   &interpolated_value_row));
    // Check result
    EXPECT_NEAR(interpolated_value_col, u_px.x(), kFloatEpsilon);
    EXPECT_NEAR(interpolated_value_row, u_px.y(), kFloatEpsilon);
  }
}

TEST_F(DepthImageTest, DeepCopy) {
  // Set constant on CPU
  constexpr float kPixelValue = 1.0f;
  setImageConstantOnCpu(kPixelValue, &depth_frame_);

  // Copy
  DepthImage copy(depth_frame_);

  // Check the copy is actually a copy
  for (int lin_idx = 0; lin_idx < copy.numel(); lin_idx++) {
    EXPECT_EQ(copy(lin_idx), kPixelValue);
  }
}

TEST_F(DepthImageTest, InterpolationGPU) {
  // Tiny images
  DepthImage image_x(2, 2, MemoryType::kUnified);
  image_x(0, 0) = 0.5f;
  image_x(0, 1) = 1.5f;
  image_x(1, 0) = 0.5f;
  image_x(1, 1) = 1.5f;
  DepthImage image_y(2, 2, MemoryType::kUnified);
  image_y(0, 0) = 0.5f;
  image_y(0, 1) = 0.5f;
  image_y(1, 0) = 1.5f;
  image_y(1, 1) = 1.5f;

  constexpr int kNumTests = 1000;
  std::vector<Eigen::Vector2f> u_px_vec;
  u_px_vec.reserve(kNumTests);
  for (int i = 0; i < kNumTests; i++) {
    u_px_vec.push_back(
        Eigen::Vector2f(test_utils::randomFloatInRange(0.5f, 1.5f),
                        test_utils::randomFloatInRange(0.5f, 1.5f)));
  }

  std::vector<float> values_x(kNumTests, 1.0f);
  std::vector<int> success_flags_x(kNumTests, 0);
  test_utils::linearInterpolateImageGpu(image_x, u_px_vec, &values_x,
                                        &success_flags_x);
  std::vector<float> values_y(kNumTests, 1.0f);
  std::vector<int> success_flags_y(kNumTests, 0);
  test_utils::linearInterpolateImageGpu(image_y, u_px_vec, &values_y,
                                        &success_flags_y);

  for (int i = 0; i < kNumTests; i++) {
    EXPECT_TRUE(success_flags_x[i] == 1);
    EXPECT_NEAR(values_x[i], u_px_vec[i].x(), kFloatEpsilon);
    EXPECT_NEAR(values_y[i], u_px_vec[i].y(), kFloatEpsilon);
  }
}

TEST_F(DepthImageTest, ValidityCheckers) {
  // Tiny images
  DepthImage image(2, 2, MemoryType::kUnified);
  image(0, 0) = -1.0f;
  image(0, 1) = -1.0f;
  image(1, 0) = -1.0f;
  image(1, 1) = -1.0f;

  // Linear
  const Vector2f u_px(1.0, 1.0);
  float interpolated_value;
  EXPECT_TRUE(
      interpolation::interpolate2DLinear(image, u_px, &interpolated_value));
  EXPECT_EQ(interpolated_value, -1.0);
  bool res = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, &interpolated_value);
  EXPECT_FALSE(res);

  // Closest
  EXPECT_TRUE(
      interpolation::interpolate2DClosest(image, u_px, &interpolated_value));
  EXPECT_EQ(interpolated_value, -1.0);
  res = interpolation::interpolate2DClosest<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, &interpolated_value);
  EXPECT_FALSE(res);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
