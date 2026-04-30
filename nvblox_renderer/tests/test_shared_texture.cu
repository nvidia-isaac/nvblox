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

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_texture.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {
namespace test {

class SharedTextureTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default state after construction
TEST_F(SharedTextureTest, DefaultState) {
  SharedTexture texture;

  EXPECT_FALSE(texture.isValid());
  EXPECT_TRUE(texture.image() == VK_NULL_HANDLE);
  EXPECT_TRUE(texture.imageView() == VK_NULL_HANDLE);
  EXPECT_TRUE(texture.sampler() == VK_NULL_HANDLE);
  EXPECT_EQ(texture.cudaArray(), nullptr);
  EXPECT_EQ(texture.width(), 0u);
  EXPECT_EQ(texture.height(), 0u);
}

// Test create() fails with null context
TEST_F(SharedTextureTest, CreateWithNullContext) {
  SharedTexture texture;

  EXPECT_FALSE(
      texture.create(nullptr, 640, 480, SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());
}

// Test create() fails with zero dimensions
TEST_F(SharedTextureTest, CreateWithZeroDimensions) {
  SharedTexture texture;

  EXPECT_FALSE(texture.create(nullptr, 0, 480, SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.create(nullptr, 640, 0, SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.create(nullptr, 0, 0, SharedTexture::Format::kRGBA8));
}

// Test destroy() is safe on uninitialized texture
TEST_F(SharedTextureTest, DestroyUninitialized) {
  SharedTexture texture;

  // Should not crash
  texture.destroy();
  EXPECT_FALSE(texture.isValid());
}

// Test destructor doesn't crash on uninitialized texture
TEST_F(SharedTextureTest, DestructorUninitialized) {
  // Create and immediately destroy - should not crash
  SharedTexture* texture = new SharedTexture();
  delete texture;
}

// Test copyFromCuda() with null data on uninitialized texture
TEST_F(SharedTextureTest, CopyFromCudaUninitialized) {
  SharedTexture texture;
  CudaStreamOwning stream;

  // Should not crash (early return due to invalid texture)
  texture.copyFromCuda(nullptr, stream);
  EXPECT_FALSE(texture.isValid());
}

// Test move constructor
TEST_F(SharedTextureTest, MoveConstructor) {
  SharedTexture texture1;
  // Texture is not initialized, but move should still work

  SharedTexture texture2(std::move(texture1));
  EXPECT_FALSE(texture2.isValid());
}

// Test move assignment
TEST_F(SharedTextureTest, MoveAssignment) {
  SharedTexture texture1;
  SharedTexture texture2;

  texture2 = std::move(texture1);
  EXPECT_FALSE(texture2.isValid());
}

// Test Format enum values are distinct
TEST_F(SharedTextureTest, FormatEnum) {
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kR32F),
            static_cast<int>(SharedTexture::Format::kRGBA8));
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kR32F),
            static_cast<int>(SharedTexture::Format::kRGB8));
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kR32F),
            static_cast<int>(SharedTexture::Format::kR8));
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kRGBA8),
            static_cast<int>(SharedTexture::Format::kRGB8));
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kRGBA8),
            static_cast<int>(SharedTexture::Format::kR8));
  EXPECT_NE(static_cast<int>(SharedTexture::Format::kRGB8),
            static_cast<int>(SharedTexture::Format::kR8));
}

// ==============================================================================
// Edge Case Tests - Dimension Limits
// Note: With null context, create() fails at context validation before
// dimension validation. Dimension limit tests with valid context are in the
// integration test section below.
// ==============================================================================

// Test create() fails with excessive dimensions (fails at context validation
// first)
TEST_F(SharedTextureTest, CreateWithExcessiveDimensionsNullContext) {
  SharedTexture texture;

  // Note: These fail at context validation (null check) before dimension check,
  // but still demonstrate the API behavior with invalid inputs.
  EXPECT_FALSE(texture.create(nullptr, kMaxTextureDimension + 1, 480,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());

  EXPECT_FALSE(texture.create(nullptr, 640, kMaxTextureDimension + 1,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());
}

// Test minimum dimension (1x1 texture) - fails at context validation
TEST_F(SharedTextureTest, CreateWithMinimumDimensionsNullContext) {
  SharedTexture texture;

  // The failure is due to null context, not dimensions
  EXPECT_FALSE(texture.create(nullptr, kMinTextureDimension,
                              kMinTextureDimension,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());
}

// ==============================================================================
// Integration tests with valid VkContext
// ==============================================================================

class SharedTextureIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check CUDA availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    // Initialize Vulkan context
    std::vector<const char*> extensions;
    if (!ctx_.init("test_shared_texture", extensions, false)) {
      GTEST_SKIP() << "Failed to initialize Vulkan";
    }
    if (!ctx_.createDevice()) {
      GTEST_SKIP() << "Failed to create Vulkan device";
    }
  }

  void TearDown() override {}

  VkContext ctx_;
};

// Test texture creation with valid context
TEST_F(SharedTextureIntegrationTest, CreateTextureRGBA8) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 640;
  constexpr uint32_t kHeight = 480;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));

  EXPECT_TRUE(texture.isValid());
  EXPECT_TRUE(texture.image() != VK_NULL_HANDLE);
  EXPECT_TRUE(texture.imageView() != VK_NULL_HANDLE);
  EXPECT_TRUE(texture.sampler() != VK_NULL_HANDLE);
  EXPECT_NE(texture.cudaArray(), nullptr);
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);
}

// Test texture creation with all formats
TEST_F(SharedTextureIntegrationTest, CreateTextureAllFormats) {
  constexpr uint32_t kWidth = 320;
  constexpr uint32_t kHeight = 240;

  // R32F (depth format)
  SharedTexture depth_texture;
  ASSERT_TRUE(depth_texture.create(&ctx_, kWidth, kHeight,
                                   SharedTexture::Format::kR32F));
  EXPECT_TRUE(depth_texture.isValid());

  // RGBA8 (color format)
  SharedTexture rgba_texture;
  ASSERT_TRUE(rgba_texture.create(&ctx_, kWidth, kHeight,
                                  SharedTexture::Format::kRGBA8));
  EXPECT_TRUE(rgba_texture.isValid());

  // RGB8 (converted to RGBA internally)
  SharedTexture rgb_texture;
  ASSERT_TRUE(
      rgb_texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGB8));
  EXPECT_TRUE(rgb_texture.isValid());

  // R8 (grayscale format)
  SharedTexture gray_texture;
  ASSERT_TRUE(
      gray_texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kR8));
  EXPECT_TRUE(gray_texture.isValid());
}

// Test CUDA copy to RGBA texture
TEST_F(SharedTextureIntegrationTest, CopyFromCudaRGBA) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 64;
  constexpr uint32_t kHeight = 64;
  constexpr size_t kPixelSize = 4;  // RGBA8
  CudaStreamOwning stream;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));

  // Allocate and fill CUDA source data with gradient
  size_t pitch = kWidth * kPixelSize;
  size_t total_size = pitch * kHeight;
  unsigned char* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, total_size), cudaSuccess);

  std::vector<unsigned char> h_data(total_size);
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      size_t idx = y * pitch + x * kPixelSize;
      h_data[idx + 0] = static_cast<unsigned char>(x * 4);        // R
      h_data[idx + 1] = static_cast<unsigned char>(y * 4);        // G
      h_data[idx + 2] = static_cast<unsigned char>((x + y) * 2);  // B
      h_data[idx + 3] = 255;                                      // A
    }
  }
  ASSERT_EQ(
      cudaMemcpy(d_src, h_data.data(), total_size, cudaMemcpyHostToDevice),
      cudaSuccess);

  // Copy to shared texture
  EXPECT_TRUE(texture.copyFromCuda(d_src, stream));

  // Synchronize to ensure copy is complete
  stream.synchronize();

  cudaFree(d_src);
}

// Test CUDA copy to depth texture (R32F)
TEST_F(SharedTextureIntegrationTest, CopyFromCudaDepth) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 64;
  constexpr uint32_t kHeight = 64;
  constexpr size_t kPixelSize = 4;  // R32F = 4 bytes
  CudaStreamOwning stream;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kR32F));

  // Allocate and fill CUDA source data
  size_t pitch = kWidth * kPixelSize;
  size_t total_size = pitch * kHeight;
  float* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, total_size), cudaSuccess);

  std::vector<float> h_data(kWidth * kHeight);
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      h_data[y * kWidth + x] = static_cast<float>(x + y) * 0.1f;
    }
  }
  ASSERT_EQ(
      cudaMemcpy(d_src, h_data.data(), total_size, cudaMemcpyHostToDevice),
      cudaSuccess);

  // Copy to shared texture
  EXPECT_TRUE(texture.copyFromCuda(d_src, stream));

  stream.synchronize();
  cudaFree(d_src);
}

// Test RGB to RGBA conversion
TEST_F(SharedTextureIntegrationTest, CopyFromCudaRGB) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 64;
  constexpr uint32_t kHeight = 64;
  constexpr size_t kPixelSize = 3;  // RGB8
  CudaStreamOwning stream;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGB8));

  // Allocate RGB source data
  size_t pitch = kWidth * kPixelSize;
  size_t total_size = pitch * kHeight;
  unsigned char* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, total_size), cudaSuccess);

  std::vector<unsigned char> h_data(total_size);
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      size_t idx = y * pitch + x * kPixelSize;
      h_data[idx + 0] = static_cast<unsigned char>(x * 4);        // R
      h_data[idx + 1] = static_cast<unsigned char>(y * 4);        // G
      h_data[idx + 2] = static_cast<unsigned char>((x + y) * 2);  // B
    }
  }
  ASSERT_EQ(
      cudaMemcpy(d_src, h_data.data(), total_size, cudaMemcpyHostToDevice),
      cudaSuccess);

  // Copy should convert RGB to RGBA
  EXPECT_TRUE(texture.copyFromCuda(d_src, stream));

  stream.synchronize();
  cudaFree(d_src);
}

// Test move semantics with valid texture
TEST_F(SharedTextureIntegrationTest, MoveWithValidTexture) {
  SharedTexture texture1;
  constexpr uint32_t kWidth = 320;
  constexpr uint32_t kHeight = 240;

  ASSERT_TRUE(
      texture1.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));
  VkImage original_image = texture1.image();
  VkImageView original_view = texture1.imageView();
  cudaArray_t original_array = texture1.cudaArray();

  // Move to new texture
  SharedTexture texture2(std::move(texture1));

  // texture2 should have the resources
  EXPECT_TRUE(texture2.isValid());
  EXPECT_EQ(texture2.image(), original_image);
  EXPECT_EQ(texture2.imageView(), original_view);
  EXPECT_EQ(texture2.cudaArray(), original_array);
  EXPECT_EQ(texture2.width(), kWidth);
  EXPECT_EQ(texture2.height(), kHeight);

  // texture1 should be empty
  EXPECT_FALSE(texture1.isValid());
  EXPECT_TRUE(texture1.image() == VK_NULL_HANDLE);
  EXPECT_TRUE(texture1.imageView() == VK_NULL_HANDLE);
  EXPECT_EQ(texture1.cudaArray(), nullptr);
}

// ==============================================================================
// Edge Case Tests - Dimension Limits (with valid context)
// ==============================================================================

// Test create() fails with dimensions exceeding maximum
TEST_F(SharedTextureIntegrationTest, CreateWithExcessiveDimensions) {
  SharedTexture texture;

  // Width exceeds max
  EXPECT_FALSE(texture.create(&ctx_, kMaxTextureDimension + 1, 480,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());

  // Height exceeds max
  EXPECT_FALSE(texture.create(&ctx_, 640, kMaxTextureDimension + 1,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());

  // Both exceed max
  EXPECT_FALSE(texture.create(&ctx_, kMaxTextureDimension + 1,
                              kMaxTextureDimension + 1,
                              SharedTexture::Format::kRGBA8));
  EXPECT_FALSE(texture.isValid());
}

// Test minimum dimension (1x1 texture) works correctly
TEST_F(SharedTextureIntegrationTest, CreateWithMinimumDimensions) {
  SharedTexture texture;

  // 1x1 should be valid and creatable
  ASSERT_TRUE(texture.create(&ctx_, kMinTextureDimension, kMinTextureDimension,
                             SharedTexture::Format::kRGBA8));
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kMinTextureDimension);
  EXPECT_EQ(texture.height(), kMinTextureDimension);
}

// ==============================================================================
// Resize Edge Case Tests
// ==============================================================================

// Test resize to zero dimensions fails gracefully
TEST_F(SharedTextureIntegrationTest, ResizeToZeroDimensions) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 320;
  constexpr uint32_t kHeight = 240;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));

  // Resize to zero width should fail
  EXPECT_FALSE(texture.resize(0, kHeight));
  // Original texture should still be valid
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);

  // Resize to zero height should fail
  EXPECT_FALSE(texture.resize(kWidth, 0));
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);

  // Resize to both zero should fail
  EXPECT_FALSE(texture.resize(0, 0));
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);
}

// Test resize to same dimensions is a no-op
TEST_F(SharedTextureIntegrationTest, ResizeToSameDimensions) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 320;
  constexpr uint32_t kHeight = 240;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));
  VkImage original_image = texture.image();

  // Resize to same dimensions should be a no-op
  ASSERT_TRUE(texture.resize(kWidth, kHeight));
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);
  // Image should not have been recreated
  EXPECT_EQ(texture.image(), original_image);
}

// Test rapid consecutive resizes
TEST_F(SharedTextureIntegrationTest, RapidConsecutiveResizes) {
  SharedTexture texture;
  constexpr uint32_t kInitialWidth = 320;
  constexpr uint32_t kInitialHeight = 240;

  ASSERT_TRUE(texture.create(&ctx_, kInitialWidth, kInitialHeight,
                             SharedTexture::Format::kRGBA8));

  // Perform multiple rapid resizes
  const std::vector<std::pair<uint32_t, uint32_t>> sizes = {
      {640, 480}, {1280, 720}, {800, 600}, {320, 240}, {1920, 1080}};

  for (const auto& [width, height] : sizes) {
    ASSERT_TRUE(texture.resize(width, height))
        << "Failed to resize to " << width << "x" << height;
    EXPECT_TRUE(texture.isValid());
    EXPECT_EQ(texture.width(), width);
    EXPECT_EQ(texture.height(), height);
  }
}

// Test resize to excessive dimensions fails
TEST_F(SharedTextureIntegrationTest, ResizeToExcessiveDimensions) {
  SharedTexture texture;
  constexpr uint32_t kWidth = 320;
  constexpr uint32_t kHeight = 240;

  ASSERT_TRUE(
      texture.create(&ctx_, kWidth, kHeight, SharedTexture::Format::kRGBA8));

  // Resize to excessive width should fail
  EXPECT_FALSE(texture.resize(kMaxTextureDimension + 1, kHeight));
  // Original texture should still be valid
  EXPECT_TRUE(texture.isValid());
  EXPECT_EQ(texture.width(), kWidth);
  EXPECT_EQ(texture.height(), kHeight);

  // Resize to excessive height should fail
  EXPECT_FALSE(texture.resize(kWidth, kMaxTextureDimension + 1));
  EXPECT_TRUE(texture.isValid());
}

// Test resize on uninitialized texture fails
TEST_F(SharedTextureIntegrationTest, ResizeUninitialized) {
  SharedTexture texture;

  EXPECT_FALSE(texture.resize(640, 480));
  EXPECT_FALSE(texture.isValid());
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
