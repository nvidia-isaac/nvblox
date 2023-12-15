/*
Copyright 2023 NVIDIA CORPORATION

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
#include "nvblox/sensors/connected_components.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;
using namespace test_utils;

// known sizes of the two squares used in some tests
constexpr size_t kSquare1Size = 400;
constexpr size_t kSquare2Size = 900;
constexpr size_t kMaskImageResolution = 640 * 480;
constexpr size_t kFromDiskNumMaskedPixels = 11603;
class ConnectedComponentsFixture : public ::testing::Test {
 protected:
  void run(const MaskImageType type, int32_t threshold,
           int32_t num_expected_pixels) {
    createMaskImage(&mask_, type);
    mask_out_ = MonoImage(mask_.rows(), mask_.cols(), MemoryType::kHost);

    image::removeSmallConnectedComponents(mask_, threshold, &mask_out_,
                                          CudaStreamOwning());

    EXPECT_EQ(getNumMaskedPixels(mask_out_), num_expected_pixels);
  }

  int getNumMaskedPixels(const MonoImage& mask) {
    int count = 0;
    for (int i = 0; i < mask.numel(); ++i) {
      count += mask(i) > 0;
    }
    return count;
  }

  MonoImage mask_{MemoryType::kHost};
  MonoImage mask_out_{MemoryType::kHost};
};
TEST_F(ConnectedComponentsFixture, RealMask) {
  run(MaskImageType::kFromDisk, 10000, kFromDiskNumMaskedPixels);
}

TEST_F(ConnectedComponentsFixture, EmptyMask) {
  run(MaskImageType::kEverythingZero, 10000, 0);
}

TEST_F(ConnectedComponentsFixture, FullMask) {
  run(MaskImageType::kEverythingFilled, 10000, kMaskImageResolution);
}

TEST_F(ConnectedComponentsFixture, TwoSquares_keepBoth) {
  run(MaskImageType::kTwoSquares, kSquare1Size, kSquare1Size + kSquare2Size);
}

TEST_F(ConnectedComponentsFixture, TwoSquares_keepOne) {
  run(MaskImageType::kTwoSquares, kSquare1Size + 1, kSquare2Size);
}

TEST_F(ConnectedComponentsFixture, TwoSquares_keepNone) {
  run(MaskImageType::kTwoSquares, kSquare2Size + 1, 0);
}

TEST_F(ConnectedComponentsFixture, GridPattern) {
  run(MaskImageType::kGrid, 1, kMaskImageResolution / 4);
}

TEST(ConnectedComponents, BlobInTopLeftCorner) {
  MonoImage mask(10, 10, MemoryType::kHost);
  MonoImage mask_out(10, 10, MemoryType::kHost);

  std::vector<std::array<int, 2>> pixels{{0, 0}, {0, 1}, {1, 0}};
  for (auto p : pixels) mask(p[0], p[1]) = 255;
  image::removeSmallConnectedComponents(mask, 3, &mask_out, CudaStreamOwning());
  for (auto p : pixels) EXPECT_GT(mask_out(p[0], p[1]), 0);
}

TEST(ConnectedComponents, BlobInTopRightCorner) {
  MonoImage mask(10, 10, MemoryType::kHost);
  MonoImage mask_out(10, 10, MemoryType::kHost);
  std::vector<std::array<int, 2>> pixels{{0, 9}, {1, 9}, {0, 8}};
  for (auto p : pixels) mask(p[0], p[1]) = 255;
  image::removeSmallConnectedComponents(mask, 3, &mask_out, CudaStreamOwning());
  for (auto p : pixels) EXPECT_GT(mask_out(p[0], p[1]), 0);
}

TEST(ConnectedComponents, BlobInBottomLeftCorner) {
  MonoImage mask(10, 10, MemoryType::kHost);
  MonoImage mask_out(10, 10, MemoryType::kHost);
  std::vector<std::array<int, 2>> pixels{{9, 0}, {9, 1}, {8, 0}};
  for (auto p : pixels) mask(p[0], p[1]) = 255;
  image::removeSmallConnectedComponents(mask, 3, &mask_out, CudaStreamOwning());
  for (auto p : pixels) EXPECT_GT(mask_out(p[0], p[1]), 0);
}

TEST(ConnectedComponents, BlobInBottomRightCorner) {
  MonoImage mask(10, 10, MemoryType::kHost);
  MonoImage mask_out(10, 10, MemoryType::kHost);
  std::vector<std::array<int, 2>> pixels{{9, 9}, {9, 8}, {8, 9}};
  for (auto p : pixels) mask(p[0], p[1]) = 255;
  image::removeSmallConnectedComponents(mask, 3, &mask_out, CudaStreamOwning());
  for (auto p : pixels) EXPECT_GT(mask_out(p[0], p[1]), 0);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
