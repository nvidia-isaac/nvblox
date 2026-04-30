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

#include "nvblox/sensors/image.h"

using namespace nvblox;

// Helper: fill a unified DepthImage with known values and return it.
DepthImage makeTestDepthImage(int rows, int cols) {
  DepthImage image(rows, cols, MemoryType::kUnified);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      image(r, c) = static_cast<float>(r * cols + c);
    }
  }
  return image;
}

// Helper: verify that a unified DepthImage contains the expected sequential
// values written by makeTestDepthImage.
void verifyTestDepthImage(const DepthImage& image, int rows, int cols) {
  ASSERT_EQ(image.rows(), rows);
  ASSERT_EQ(image.cols(), cols);
  ASSERT_EQ(image.stride_num_elements(), cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ASSERT_EQ(image(r, c), static_cast<float>(r * cols + c))
          << "Mismatch at (" << r << ", " << c << ")";
    }
  }
}

// Verify dimensions, stride and num_elements_per_pixel of an image.
template <typename ImageType>
void verifyImageAttributes(const ImageType& image, int rows, int cols,
                           int expected_stride_num_elements,
                           int expected_num_elements_per_pixel = 1) {
  EXPECT_EQ(image.rows(), rows);
  EXPECT_EQ(image.cols(), cols);
  EXPECT_EQ(image.stride_num_elements(), expected_stride_num_elements);
  EXPECT_EQ(image.num_elements_per_pixel(), expected_num_elements_per_pixel);
}

// ---------- Image (memory-owning) move operations ----------

// Use dimensions where cols is divisible by sizeof(float), so the
// stride/bytes conversion succeeds silently and produces a wrong stride value.
constexpr int kRows = 3;
constexpr int kCols = 480;

TEST(ImageMoveTest, MoveConstruct) {
  DepthImage original = makeTestDepthImage(kRows, kCols);
  const float* original_data = original.dataConstPtr();

  DepthImage moved(std::move(original));

  // The moved-to image must have the original attributes.
  verifyImageAttributes(moved, kRows, kCols, kCols);
  // Data pointer should be the same (ownership transferred).
  EXPECT_EQ(moved.dataConstPtr(), original_data);
  // Pixel values must survive the move.
  verifyTestDepthImage(moved, kRows, kCols);

  // The moved-from image must be empty.
  EXPECT_EQ(original.rows(), 0);
  EXPECT_EQ(original.cols(), 0);
  EXPECT_EQ(original.dataConstPtr(), nullptr);
}

TEST(ImageMoveTest, MoveAssign) {
  DepthImage original = makeTestDepthImage(kRows, kCols);
  const float* original_data = original.dataConstPtr();

  DepthImage moved(MemoryType::kUnified);
  moved = std::move(original);

  verifyImageAttributes(moved, kRows, kCols, kCols);
  EXPECT_EQ(moved.dataConstPtr(), original_data);
  verifyTestDepthImage(moved, kRows, kCols);

  EXPECT_EQ(original.rows(), 0);
  EXPECT_EQ(original.cols(), 0);
  EXPECT_EQ(original.dataConstPtr(), nullptr);
}

// Also test with MonoImage (sizeof(uint8_t)==1) and
// ColorImage (sizeof(Color)==3) where bugs may manifest differently.
TEST(ImageMoveTest, MoveConstructMonoImage) {
  constexpr int kMonoRows = 5;
  constexpr int kMonoCols = 7;
  MonoImage original(kMonoRows, kMonoCols, MemoryType::kUnified);
  for (int r = 0; r < kMonoRows; ++r) {
    for (int c = 0; c < kMonoCols; ++c) {
      original(r, c) = static_cast<uint8_t>(r * kMonoCols + c);
    }
  }

  MonoImage moved(std::move(original));

  verifyImageAttributes(moved, kMonoRows, kMonoCols, kMonoCols);
  for (int r = 0; r < kMonoRows; ++r) {
    for (int c = 0; c < kMonoCols; ++c) {
      EXPECT_EQ(moved(r, c), static_cast<uint8_t>(r * kMonoCols + c));
    }
  }
}

TEST(ImageMoveTest, MoveConstructColorImage) {
  constexpr int kColorRows = 4;
  constexpr int kColorCols = 480;
  ColorImage original(kColorRows, kColorCols, MemoryType::kUnified);
  for (int row = 0; row < kColorRows; ++row) {
    for (int col = 0; col < kColorCols; ++col) {
      original(row, col) = Color(row, col % 256, 0);
    }
  }

  ColorImage moved(std::move(original));

  verifyImageAttributes(moved, kColorRows, kColorCols, kColorCols);
  for (int row = 0; row < kColorRows; ++row) {
    for (int col = 0; col < kColorCols; ++col) {
      ASSERT_EQ(moved(row, col).r(), static_cast<uint8_t>(row));
      ASSERT_EQ(moved(row, col).g(), static_cast<uint8_t>(col % 256));
    }
  }
}

// ---------- ImageView copy/move operations ----------

TEST(ImageViewCopyMoveTest, CopyConstruct) {
  DepthImage image = makeTestDepthImage(kRows, kCols);
  DepthImageView original(image);

  DepthImageView copy(original);

  verifyImageAttributes(copy, kRows, kCols, kCols);
  EXPECT_EQ(copy.dataConstPtr(), original.dataConstPtr());
}

TEST(ImageViewCopyMoveTest, CopyAssign) {
  DepthImage image = makeTestDepthImage(kRows, kCols);
  DepthImageView original(image);

  DepthImageView copy;
  copy = original;

  verifyImageAttributes(copy, kRows, kCols, kCols);
  EXPECT_EQ(copy.dataConstPtr(), original.dataConstPtr());
}

TEST(ImageViewCopyMoveTest, MoveConstruct) {
  DepthImage image = makeTestDepthImage(kRows, kCols);
  DepthImageView original(image);
  const float* data_ptr = original.dataConstPtr();

  DepthImageView moved(std::move(original));

  verifyImageAttributes(moved, kRows, kCols, kCols);
  EXPECT_EQ(moved.dataConstPtr(), data_ptr);
}

TEST(ImageViewCopyMoveTest, MoveAssign) {
  DepthImage image = makeTestDepthImage(kRows, kCols);
  DepthImageView original(image);
  const float* data_ptr = original.dataConstPtr();

  DepthImageView moved;
  moved = std::move(original);

  verifyImageAttributes(moved, kRows, kCols, kCols);
  EXPECT_EQ(moved.dataConstPtr(), data_ptr);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
