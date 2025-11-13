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
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/sensors/image.h"
#include "nvblox/tests/interpolation_2d_gpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;
const __half kHalfEpsilon = 1e-1;

constexpr int kRows = 480;
constexpr int kCols = 640;

TEST(InterpolationTest, LinearInterpolationDepthImage) {
  // The images {depth_frame_col_coords, depth_frame_row_coords} are set up such
  // that if you interpolate, you should get the interpolated position back.
  DepthImage depth_frame_col_coords(kRows, kCols, MemoryType::kUnified);
  DepthImage depth_frame_row_coords(kRows, kCols, MemoryType::kUnified);
  for (int col_idx = 0; col_idx < kCols; col_idx++) {
    for (int row_idx = 0; row_idx < kRows; row_idx++) {
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
                            0.5f, static_cast<float>(kCols - 1) + 0.5f),
                        test_utils::randomFloatInRange(
                            0.5f, static_cast<float>(kRows - 1) + 0.5f));
    // Interpolate x and y grids
    float interpolated_value_col;
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        DepthImageConstView(depth_frame_col_coords), u_px,
        &interpolated_value_col));
    float interpolated_value_row;
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        DepthImageConstView(depth_frame_row_coords), u_px,
        &interpolated_value_row));
    // Check result
    EXPECT_NEAR(interpolated_value_col, u_px.x(), kFloatEpsilon);
    EXPECT_NEAR(interpolated_value_row, u_px.y(), kFloatEpsilon);
  }
}

template <typename FloatType>
void testLinearInterpolationFloatArray(const int num_rows, const int num_cols,
                                       const FloatType epsilon) {
  // The images  are set up such
  // that if you interpolate, you should get the interpolated position back.
  constexpr size_t kArraySize = 10;

  using ArrayImage = Image<Array<FloatType, kArraySize>>;
  using ArrayImageConstView = ImageView<const Array<FloatType, kArraySize>>;

  ArrayImage frame_col_coords(num_rows, num_cols, MemoryType::kUnified);
  ArrayImage frame_row_coords(num_rows, num_cols, MemoryType::kUnified);

  for (int col_idx = 0; col_idx < num_cols; col_idx++) {
    for (int row_idx = 0; row_idx < num_rows; row_idx++) {
      // Note(dtingdahl) casting to "float" rather than FloatType since
      // cuda 11.8 doesn't support implicit conversion from int to __half
      const FloatType col_idx_float = static_cast<float>(col_idx);
      const FloatType row_idx_float = static_cast<float>(row_idx);

      std::fill(frame_col_coords(row_idx, col_idx).begin(),
                frame_col_coords(row_idx, col_idx).end(),
                col_idx_float + static_cast<FloatType>(0.5));

      std::fill(frame_row_coords(row_idx, col_idx).begin(),
                frame_row_coords(row_idx, col_idx).end(),
                row_idx_float + static_cast<FloatType>(0.5));
    }
  }
  constexpr int kNumTests = 1000;
  // int num_failures = 0;
  for (int i = 0; i < kNumTests; i++) {
    // Random pixel location on image plane
    const Vector2f u_px(test_utils::randomFloatInRange(
                            0.5f, static_cast<float>(num_cols - 1) + 0.5f),
                        test_utils::randomFloatInRange(
                            0.5f, static_cast<float>(num_rows - 1) + 0.5f));
    // Interpolate x and y grids
    Array<FloatType, kArraySize> interpolated_value_col;
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        ArrayImageConstView(frame_col_coords), u_px, &interpolated_value_col));

    Array<FloatType, kArraySize> interpolated_value_row;
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        ArrayImageConstView(frame_row_coords), u_px, &interpolated_value_row));
    // Check result

    for (size_t i_array = 0; i_array < kArraySize; ++i_array) {
      EXPECT_NEAR(interpolated_value_col[i_array], u_px.x(), epsilon);
      EXPECT_NEAR(interpolated_value_row[i_array], u_px.y(), epsilon);
    }
  }
}

TEST(InterpolationTest, LinearInterpolationFloatArray) {
  testLinearInterpolationFloatArray<float>(kRows, kCols, kFloatEpsilon);
}

TEST(InterpolationTest, LinearInterpolationHalfArray) {
  // Use smaller image size due to reduced precision of __half
  testLinearInterpolationFloatArray<__half>(64, 64, kHalfEpsilon);
}

TEST(InterpolationTest, InterpolationGPU) {
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

TEST(InterpolationTest, ValidityCheckers) {
  // Tiny images
  DepthImage image(2, 2, MemoryType::kUnified);
  image(0, 0) = -1.0f;
  image(0, 1) = -1.0f;
  image(1, 0) = -1.0f;
  image(1, 1) = -1.0f;

  DepthImageConstView view(image);
  // Linear
  const Vector2f u_px(1.0, 1.0);
  float interpolated_value;
  EXPECT_TRUE(
      interpolation::interpolate2DLinear(view, u_px, &interpolated_value));
  EXPECT_EQ(interpolated_value, -1.0);
  bool res = interpolation::interpolate2DLinear<
      float, interpolation::checkers::PixelIsValidDepth>(view, u_px,
                                                         &interpolated_value);
  EXPECT_FALSE(res);

  // Closest
  EXPECT_TRUE(
      interpolation::interpolate2DClosest(view, u_px, &interpolated_value));
  EXPECT_EQ(interpolated_value, -1.0);
  res = interpolation::interpolate2DClosest<
      float, interpolation::checkers::PixelIsValidDepth>(view, u_px,
                                                         &interpolated_value);
  EXPECT_FALSE(res);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
