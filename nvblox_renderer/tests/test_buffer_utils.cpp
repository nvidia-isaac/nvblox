/*
Copyright 2026 NVIDIA CORPORATION

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

#include <limits>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// wouldOverflow tests
// ==============================================================================

TEST(BufferUtilsTest, WouldOverflowZero) {
  // Zero times anything should not overflow
  EXPECT_FALSE(wouldOverflow(0, 0));
  EXPECT_FALSE(wouldOverflow(0, 100));
  EXPECT_FALSE(wouldOverflow(100, 0));
  EXPECT_FALSE(wouldOverflow(0, SIZE_MAX));
  EXPECT_FALSE(wouldOverflow(SIZE_MAX, 0));
}

TEST(BufferUtilsTest, WouldOverflowSmallNumbers) {
  // Small numbers should not overflow
  EXPECT_FALSE(wouldOverflow(1, 1));
  EXPECT_FALSE(wouldOverflow(100, 100));
  EXPECT_FALSE(wouldOverflow(1000, 1000));
  EXPECT_FALSE(wouldOverflow(1024 * 1024, 1024));  // 1 GB
}

TEST(BufferUtilsTest, WouldOverflowLargeNumbers) {
  // Large numbers that would overflow
  EXPECT_TRUE(wouldOverflow(SIZE_MAX, 2));
  EXPECT_TRUE(wouldOverflow(2, SIZE_MAX));
  EXPECT_TRUE(wouldOverflow(SIZE_MAX, SIZE_MAX));

  // Half max times 3 should overflow
  EXPECT_TRUE(wouldOverflow(SIZE_MAX / 2, 3));

  // Half max times 2 should not overflow (exactly SIZE_MAX - 1)
  EXPECT_FALSE(wouldOverflow(SIZE_MAX / 2, 2));
}

TEST(BufferUtilsTest, WouldOverflowEdgeCases) {
  // One times max should not overflow
  EXPECT_FALSE(wouldOverflow(1, SIZE_MAX));
  EXPECT_FALSE(wouldOverflow(SIZE_MAX, 1));

  // floor(sqrt(SIZE_MAX)) squared should not overflow
  // On 64-bit: floor(sqrt(2^64 - 1)) = 2^32 - 1 = 4294967295
  size_t sqrt_max = (static_cast<size_t>(1) << (sizeof(size_t) * 4)) - 1;
  EXPECT_FALSE(wouldOverflow(sqrt_max, sqrt_max));

  // sqrt_max + 1 squared should overflow
  EXPECT_TRUE(wouldOverflow(sqrt_max + 1, sqrt_max + 1));
}

// ==============================================================================
// calculateResizeCapacity tests
// ==============================================================================

TEST(BufferUtilsTest, CalculateResizeCapacitySmall) {
  // Small sizes should be multiplied by growth factor
  size_t result = calculateResizeCapacity(100);

  // 100 * 1.5 = 150
  EXPECT_EQ(result, 150u);
}

TEST(BufferUtilsTest, CalculateResizeCapacityMedium) {
  // Medium sizes
  size_t result = calculateResizeCapacity(1000000);

  // 1M * 1.5 = 1.5M
  EXPECT_EQ(result, 1500000u);
}

TEST(BufferUtilsTest, CalculateResizeCapacityLarge) {
  // Large sizes near the max should be capped
  size_t result = calculateResizeCapacity(kMaxBufferSizeBytes);

  // Should be capped at kMaxBufferSizeBytes
  EXPECT_EQ(result, kMaxBufferSizeBytes);
}

TEST(BufferUtilsTest, CalculateResizeCapacityAboveMax) {
  // Sizes above max (after growth factor) should be capped
  size_t large_size = kMaxBufferSizeBytes - 100;
  size_t result = calculateResizeCapacity(large_size);

  // large_size * 1.5 > kMaxBufferSizeBytes, so should be capped
  EXPECT_EQ(result, kMaxBufferSizeBytes);
}

TEST(BufferUtilsTest, CalculateResizeCapacityZero) {
  // Zero should result in zero (0 * 1.5 = 0)
  EXPECT_EQ(calculateResizeCapacity(0), 0u);
}

// ==============================================================================
// Constant invariant tests (relational, not exact values)
// ==============================================================================

TEST(BufferUtilsTest, ConstantInvariants) {
  // Growth factor must be > 1 for amortized resizing to work
  EXPECT_GT(kBufferGrowthFactor, 1.0f);

  // Max limits must be >= defaults
  EXPECT_GE(kMaxVertexCount, kDefaultVertexBufferSize);
  EXPECT_GE(kMaxPointCount, kDefaultPointBufferSize);

  // Texture dimension range must be valid
  EXPECT_GT(kMaxTextureDimension, kMinTextureDimension);
}

// ==============================================================================
// validateBufferSize tests
// ==============================================================================

TEST(BufferUtilsTest, ValidateBufferSizeValid) {
  auto result = validateBufferSize(100, sizeof(float), kMaxPointCount, "test");
  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.required_size, 100 * sizeof(float));
}

TEST(BufferUtilsTest, ValidateBufferSizeExceedsMax) {
  auto result = validateBufferSize(kMaxPointCount + 1, sizeof(float),
                                   kMaxPointCount, "test");
  EXPECT_FALSE(result.valid);
}

TEST(BufferUtilsTest, ValidateBufferSizeOverflow) {
  auto result = validateBufferSize(SIZE_MAX, sizeof(float), SIZE_MAX, "test");
  EXPECT_FALSE(result.valid);
}

TEST(BufferUtilsTest, ValidateBufferSizeZeroCount) {
  // Zero count should be valid
  auto result = validateBufferSize(0, sizeof(float), kMaxPointCount, "test");
  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.required_size, 0u);
}

TEST(BufferUtilsTest, ValidateBufferSizeExceedsMaxBytes) {
  // Large count that passes max_count but exceeds kMaxBufferSizeBytes
  size_t count = (kMaxBufferSizeBytes / sizeof(float)) + 1;
  auto result = validateBufferSize(count, sizeof(float), SIZE_MAX, "test");
  EXPECT_FALSE(result.valid);
}

}  // namespace test
}  // namespace renderer
}  // namespace nvblox

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
