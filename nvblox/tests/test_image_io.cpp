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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <numeric>

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/io/image_io.h"
#include "nvblox/sensors/image.h"

using namespace nvblox;

TEST(ImageIO, SaveToPng) {
  MonoImage mono_image(2, 2, MemoryType::kUnified);
  mono_image(0, 0) = 0;
  mono_image(0, 1) = 1;
  mono_image(1, 0) = 2;
  mono_image(1, 1) = 3;

  // Save
  std::string filepath = "image_io_test_mono_image.png";
  io::writeToPng(filepath, mono_image);

  // Read back
  MonoImage mono_image_readback(MemoryType::kDevice);
  EXPECT_TRUE(io::readFromPng(filepath, &mono_image_readback));

  MonoImage diff_image(MemoryType::kUnified);
  image::getDifferenceImageGPU(mono_image, mono_image_readback, &diff_image);

  EXPECT_EQ(diff_image(0, 0), 0);
  EXPECT_EQ(diff_image(0, 1), 0);
  EXPECT_EQ(diff_image(1, 0), 0);
  EXPECT_EQ(diff_image(1, 1), 0);
}

TEST(ImageIO, 3DMatchDepthToMonoAndSave) {
  const std::string base_path = "./data/3dmatch";
  constexpr int seq_id = 1;
  constexpr int frame_id = 0;
  DepthImage depth_image(MemoryType::kUnified);
  const std::string depth_image_path =
      datasets::threedmatch::internal::getPathForDepthImage(base_path, seq_id,
                                                            frame_id);
  EXPECT_TRUE(io::readFromPng(depth_image_path, &depth_image));

  std::string filepath = "image_io_test_3dmatch_depth.png";
  io::writeToPng(filepath, depth_image);
  // Check the depth image visually

  // Check that the average value is sensible.
  const float average_depth =
      std::accumulate(depth_image.dataConstPtr(),
                      depth_image.dataConstPtr() + depth_image.numel(), 0.0f) /
      static_cast<float>(depth_image.numel());
  LOG(INFO) << "Average depth of loaded 3DMatch frame: " << average_depth;
  EXPECT_GT(average_depth, 1.0f);
  EXPECT_LT(average_depth, 10.0f);
}

TEST(ImageIO, 3DMatchColorImageLoadAndSave) {
  const std::string base_path = "./data/3dmatch";
  constexpr int seq_id = 1;
  ColorImage color_image(MemoryType::kDevice);
  EXPECT_TRUE(datasets::load8BitColorImage(
      datasets::threedmatch::internal::getPathForColorImage(base_path, seq_id,
                                                            0),
      &color_image));

  std::string filepath = "image_io_test_3dmatch_color.png";
  io::writeToPng(filepath, color_image);

  // Readback
  ColorImage image_readback(MemoryType::kDevice);
  EXPECT_TRUE(io::readFromPng(filepath, &image_readback));

  // Check difference
  ColorImage diff_image(color_image.rows(), color_image.cols(),
                        MemoryType::kUnified);
  image::getDifferenceImageGPU(color_image, image_readback, &diff_image);
  for (int row_idx = 0; row_idx < color_image.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < color_image.cols(); col_idx++) {
      const Color el = diff_image(row_idx, col_idx);
      EXPECT_EQ(el.r, 0);
      EXPECT_EQ(el.g, 0);
      EXPECT_EQ(el.b, 0);
      EXPECT_EQ(el.a, 0);
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
