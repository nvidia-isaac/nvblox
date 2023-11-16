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

#include "nvblox/io/image_io.h"
#include "nvblox/semantics/image_masker.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-6;

Camera getTestCamera(const int cols, const int rows) {
  float fu = 300;
  float fv = 300;
  float cu = static_cast<float>(cols) / 2.0f;
  float cv = static_cast<float>(rows) / 2.0f;
  return Camera(fu, fv, cu, cv, cols, rows);
}

class ParameterizedImageMaskerTest : public ::testing::TestWithParam<int> {
 protected:
};

TEST_P(ParameterizedImageMaskerTest, RandomMask) {
  // Test if the splitImageOnGPU function works as it should
  std::srand(0);
  const int rows = 480;
  const int cols = 640;
  const int size_addon = GetParam();
  DepthImage depth(rows, cols, MemoryType::kUnified);
  ColorImage color(rows + size_addon, cols + size_addon, MemoryType::kUnified);
  MonoImage mask(rows + size_addon, cols + size_addon, MemoryType::kUnified);
  const Camera depth_camera = getTestCamera(cols, rows);
  const Camera mask_camera =
      getTestCamera(cols + size_addon, rows + size_addon);

  const Color valid_pixel_color(255, 255, 255, 255);
  const Color invalid_pixel_color(0, 0, 0, 0);

  // Generate random mask
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      mask(row_idx, col_idx) = std::rand() % 2;
      color(row_idx, col_idx) = valid_pixel_color;
    }
  }
  for (int row_idx = 0; row_idx < depth.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < depth.cols(); col_idx++) {
      depth(row_idx, col_idx) = 1.0;
    }
  }

  // Split
  ImageMasker image_masker;
  DepthImage unmasked_depth_output(MemoryType::kDevice);
  DepthImage masked_depth_output(MemoryType::kDevice);
  ColorImage unmasked_color_output(MemoryType::kDevice);
  ColorImage masked_color_output(MemoryType::kDevice);

  Transform T_CM_CD = Transform::Identity();
  image_masker.splitImageOnGPU(depth, mask, T_CM_CD, depth_camera, mask_camera,
                               &unmasked_depth_output, &masked_depth_output);
  image_masker.splitImageOnGPU(color, mask, &unmasked_color_output,
                               &masked_color_output);

  // Check that output images exist and dont exist in the right places
  int num_valid_pixels_color = 0;
  int num_invalid_pixels_color = 0;
  int num_valid_pixels_depth = 0;
  int num_invalid_pixels_depth = 0;
  for (int row_idx = 0; row_idx < color.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < color.cols(); col_idx++) {
      if (mask(row_idx, col_idx)) {
        EXPECT_EQ(unmasked_color_output(row_idx, col_idx), invalid_pixel_color);
        EXPECT_EQ(masked_color_output(row_idx, col_idx), valid_pixel_color);
        ++num_valid_pixels_color;
      } else {
        EXPECT_EQ(unmasked_color_output(row_idx, col_idx), valid_pixel_color);
        EXPECT_EQ(masked_color_output(row_idx, col_idx), invalid_pixel_color);
        ++num_invalid_pixels_color;
      }
    }
  }
  for (int depth_row_idx = 0; depth_row_idx < depth.rows(); depth_row_idx++) {
    for (int depth_col_idx = 0; depth_col_idx < depth.cols(); depth_col_idx++) {
      int mask_row = depth_row_idx + size_addon / 2;
      int mask_col = depth_col_idx + size_addon / 2;
      if (mask_row >= 0 && mask_col >= 0 && mask_row < mask.rows() &&
          mask_col < mask.cols() && mask(mask_row, mask_col)) {
        EXPECT_NEAR(unmasked_depth_output(depth_row_idx, depth_col_idx), -1.0f,
                    kFloatEpsilon);
        EXPECT_NEAR(masked_depth_output(depth_row_idx, depth_col_idx), 1.0f, kFloatEpsilon);
        ++num_valid_pixels_depth;
      } else {
        EXPECT_NEAR(unmasked_depth_output(depth_row_idx, depth_col_idx), 1.0f,
                    kFloatEpsilon);
        EXPECT_NEAR(masked_depth_output(depth_row_idx, depth_col_idx), -1.0f,
                    kFloatEpsilon);
        ++num_invalid_pixels_depth;
      }
    }
  }

  std::cout << "num_valid_pixels_color: " << num_valid_pixels_color << std::endl;
  std::cout << "num_invalid_pixels_color: " << num_invalid_pixels_color << std::endl;
  std::cout << "num_valid_pixels_depth: " << num_valid_pixels_depth << std::endl;
  std::cout << "num_invalid_pixels_depth: " << num_invalid_pixels_depth << std::endl;

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("unmasked_output_image.png", unmasked_depth_output);
    io::writeToPng("masked_output_image.png", masked_depth_output);
  }
}

// We test the cases where the mask resolution is bigger, smaller and equal to the depth 
// resolution 
INSTANTIATE_TEST_CASE_P(ParameterizedImageMaskerTests,
                        ParameterizedImageMaskerTest, ::testing::Values(0, 256, -256));

TEST(ImageMaskerTest, PerpendicularTransformMask) {
  const int rows = 481;  // must be odd to have a center pixel
  const int cols = 641;  // must be odd to have a center pixel
  DepthImage depth(rows, cols, MemoryType::kUnified);
  MonoImage mask(rows, cols, MemoryType::kUnified);
  const Camera depth_camera = getTestCamera(cols, rows);
  const Camera mask_camera = getTestCamera(cols, rows);

  // Mask the center pixel
  mask.setZero();
  mask(rows / 2, cols / 2) = 1;

  // Set the whole depth image to one
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      depth(row_idx, col_idx) = 1.0;
    }
  }

  ImageMasker image_masker;
  DepthImage unmasked_output(MemoryType::kDevice);
  DepthImage masked_output(MemoryType::kDevice);

  // Generate depth to mask camera transform
  Transform T_CM_CD = Transform::Identity();
  // Cameras have a relative rotation of 90 degree
  T_CM_CD.prerotate(Eigen::AngleAxisf(-0.5 * M_PI, Vector3f::UnitY()));
  // The mask camera has the the 3D depth points centered horizontally
  // and 5m distance to prevent depth points lying behind it
  T_CM_CD.pretranslate(Vector3f(1, 0, 5));

  // Do not mask points occluded on the mask image.
  image_masker.occlusion_threshold_m(0);
  image_masker.splitImageOnGPU(depth, mask, T_CM_CD, depth_camera, mask_camera,
                               &unmasked_output, &masked_output);

  // Only the leftmost vertically centered pixels on the depth image should
  // be masked with this configuration. Pixels on the right are not masked
  // because they are occluded.
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      if (col_idx == 0 && row_idx == rows / 2) {
        EXPECT_NEAR(unmasked_output(row_idx, col_idx), -1.0f, kFloatEpsilon);
        EXPECT_NEAR(masked_output(row_idx, col_idx), 1.0f, kFloatEpsilon);
      } else if (col_idx != 0 || abs(row_idx - rows / 2) > 1) {
        EXPECT_NEAR(unmasked_output(row_idx, col_idx), 1.0f, kFloatEpsilon);
        EXPECT_NEAR(masked_output(row_idx, col_idx), -1.0f, kFloatEpsilon);
      }
    }
  }

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("input_mask.png", mask);
    io::writeToPng("unmasked_output_occlusion.png", unmasked_output);
    io::writeToPng("masked_output_occlusion.png", masked_output);
  }

  // Mask all depth points that project onto the masked pixel on the mask image
  // (not considering occlusion)
  image_masker.occlusion_threshold_m(std::numeric_limits<float>::max());
  image_masker.splitImageOnGPU(depth, mask, T_CM_CD, depth_camera, mask_camera,
                               &unmasked_output, &masked_output);

  // All vertically centered pixels on the depth image should
  // be masked with this configuration.
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      if (row_idx == rows / 2) {
        EXPECT_NEAR(unmasked_output(row_idx, col_idx), -1.0f, kFloatEpsilon);
        EXPECT_NEAR(masked_output(row_idx, col_idx), 1.0f, kFloatEpsilon);
      } else if (abs(row_idx - rows / 2) > 3) {
        EXPECT_NEAR(unmasked_output(row_idx, col_idx), 1.0f, kFloatEpsilon);
        EXPECT_NEAR(masked_output(row_idx, col_idx), -1.0f, kFloatEpsilon);
      }
    }
  }

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("unmasked_output_no_occlusion.png", unmasked_output);
    io::writeToPng("masked_output_no_occlusion.png", masked_output);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
