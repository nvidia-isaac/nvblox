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

TEST(ImageMaskerTest, RandomMask) {
  std::srand(0);
  const int rows = 480;
  const int cols = 640;
  DepthImage depth(rows, cols, MemoryType::kUnified);
  ColorImage color(rows, cols, MemoryType::kUnified);
  MonoImage mask(rows, cols, MemoryType::kUnified);
  const Camera depth_camera = getTestCamera(cols, rows);
  const Camera mask_camera = getTestCamera(cols, rows);

  const Color valid_pixel_color(255, 255, 255, 255);
  const Color invalid_pixel_color(0, 0, 0, 0);

  // Generate random mask
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      mask(row_idx, col_idx) = std::rand() % 2;
      depth(row_idx, col_idx) = 1.0;
      color(row_idx, col_idx) = valid_pixel_color;
    }
  }

  // Split
  ImageMasker image_masker;
  DepthImage unmasked_depth_output;
  DepthImage masked_depth_output;
  ColorImage unmasked_color_output;
  ColorImage masked_color_output;

  Transform T_CM_CD = Transform::Identity();
  image_masker.splitImageOnGPU(depth, mask, T_CM_CD, depth_camera, mask_camera,
                               &unmasked_depth_output, &masked_depth_output);
  image_masker.splitImageOnGPU(color, mask, &unmasked_color_output,
                               &masked_color_output);

  // Check that output images exist and dont exist in the right places
  int num_valid_pixels = 0;
  int num_invalid_pixels = 0;
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      if (mask(row_idx, col_idx)) {
        EXPECT_NEAR(unmasked_depth_output(row_idx, col_idx), -1.0f,
                    kFloatEpsilon);
        EXPECT_NEAR(masked_depth_output(row_idx, col_idx), 1.0f, kFloatEpsilon);
        EXPECT_EQ(unmasked_color_output(row_idx, col_idx), invalid_pixel_color);
        EXPECT_EQ(masked_color_output(row_idx, col_idx), valid_pixel_color);
        ++num_valid_pixels;
      } else {
        EXPECT_NEAR(unmasked_depth_output(row_idx, col_idx), 1.0f,
                    kFloatEpsilon);
        EXPECT_NEAR(masked_depth_output(row_idx, col_idx), -1.0f,
                    kFloatEpsilon);
        EXPECT_EQ(unmasked_color_output(row_idx, col_idx), valid_pixel_color);
        EXPECT_EQ(masked_color_output(row_idx, col_idx), invalid_pixel_color);
        ++num_invalid_pixels;
      }
    }
  }

  std::cout << "num_valid_pixels: " << num_valid_pixels << std::endl;
  std::cout << "num_invalid_pixels: " << num_invalid_pixels << std::endl;

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("unmasked_output_image.csv", unmasked_depth_output);
    io::writeToPng("masked_output_image.csv", masked_depth_output);
  }
}

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
  DepthImage unmasked_output;
  DepthImage masked_output;

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
    io::writeToPng("input_mask.csv", mask);
    io::writeToPng("unmasked_output_occlusion.csv", unmasked_output);
    io::writeToPng("masked_output_occlusion.csv", masked_output);
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
    io::writeToPng("unmasked_output_no_occlusion.csv", unmasked_output);
    io::writeToPng("masked_output_no_occlusion.csv", masked_output);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
