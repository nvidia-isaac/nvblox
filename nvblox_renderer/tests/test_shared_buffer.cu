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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_buffer.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {
namespace test {

class SharedBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default state after construction
TEST_F(SharedBufferTest, DefaultState) {
  SharedBuffer buffer;

  EXPECT_FALSE(buffer.isValid());
  EXPECT_TRUE(buffer.buffer() == VK_NULL_HANDLE);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(buffer.cudaPtr(), nullptr);
}

// Test create() fails with null context
TEST_F(SharedBufferTest, CreateWithNullContext) {
  SharedBuffer buffer;

  EXPECT_FALSE(buffer.create(nullptr, 1024, SharedBuffer::Usage::kVertex));
  EXPECT_FALSE(buffer.isValid());
}

// Test destroy() is safe on uninitialized buffer
TEST_F(SharedBufferTest, DestroyUninitialized) {
  SharedBuffer buffer;

  // Should not crash
  buffer.destroy();
  EXPECT_FALSE(buffer.isValid());
}

// Test destructor doesn't crash on uninitialized buffer
TEST_F(SharedBufferTest, DestructorUninitialized) {
  // Create and immediately destroy - should not crash
  SharedBuffer* buffer = new SharedBuffer();
  delete buffer;
}

// Test resize() fails on uninitialized buffer
TEST_F(SharedBufferTest, ResizeUninitialized) {
  SharedBuffer buffer;

  EXPECT_FALSE(buffer.resize(2048));
}

// Test copyFromCuda() fails on uninitialized buffer
TEST_F(SharedBufferTest, CopyFromCudaUninitialized) {
  SharedBuffer buffer;
  float dummy_data[10] = {0};
  CudaStreamOwning stream;

  EXPECT_FALSE(buffer.copyFromCuda(dummy_data, sizeof(dummy_data), stream));
}

// Test move constructor
TEST_F(SharedBufferTest, MoveConstructor) {
  SharedBuffer buffer1;
  // Buffer is not initialized, but move should still work

  SharedBuffer buffer2(std::move(buffer1));
  EXPECT_FALSE(buffer2.isValid());
}

// Test move assignment
TEST_F(SharedBufferTest, MoveAssignment) {
  SharedBuffer buffer1;
  SharedBuffer buffer2;

  buffer2 = std::move(buffer1);
  EXPECT_FALSE(buffer2.isValid());
}

// Test Usage enum values
TEST_F(SharedBufferTest, UsageEnum) {
  // Verify enum values are distinct
  EXPECT_NE(static_cast<int>(SharedBuffer::Usage::kVertex),
            static_cast<int>(SharedBuffer::Usage::kIndex));
  EXPECT_NE(static_cast<int>(SharedBuffer::Usage::kVertex),
            static_cast<int>(SharedBuffer::Usage::kStorage));
  EXPECT_NE(static_cast<int>(SharedBuffer::Usage::kIndex),
            static_cast<int>(SharedBuffer::Usage::kStorage));
}

// Test create() fails with zero size
TEST_F(SharedBufferTest, CreateWithZeroSize) {
  SharedBuffer buffer;

  // Zero size should fail (no context anyway, but tests the size validation
  // path)
  EXPECT_FALSE(buffer.create(nullptr, 0, SharedBuffer::Usage::kVertex));
  EXPECT_FALSE(buffer.isValid());
}

// Test resize() fails on uninitialized buffer with large size
TEST_F(SharedBufferTest, ResizeUninitializedLargeSize) {
  SharedBuffer buffer;

  // Even large sizes should fail gracefully on uninitialized buffer
  EXPECT_FALSE(buffer.resize(kMaxBufferSizeBytes));
  EXPECT_FALSE(buffer.resize(SIZE_MAX));
}

// ==============================================================================
// Integration tests with valid VkContext
// ==============================================================================

class SharedBufferIntegrationTest : public ::testing::Test {
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
    if (!ctx_.init("test_shared_buffer", extensions, false)) {
      GTEST_SKIP() << "Failed to initialize Vulkan";
    }
    if (!ctx_.createDevice()) {
      GTEST_SKIP() << "Failed to create Vulkan device";
    }
  }

  void TearDown() override {}

  VkContext ctx_;
};

// Test buffer creation with valid context
TEST_F(SharedBufferIntegrationTest, CreateBuffer) {
  SharedBuffer buffer;
  constexpr size_t kBufferSize = 1024;

  ASSERT_TRUE(buffer.create(&ctx_, kBufferSize, SharedBuffer::Usage::kVertex));

  EXPECT_TRUE(buffer.isValid());
  EXPECT_TRUE(buffer.buffer() != VK_NULL_HANDLE);
  EXPECT_EQ(buffer.size(), kBufferSize);
  EXPECT_NE(buffer.cudaPtr(), nullptr);
}

// Test buffer creation with different usage types
TEST_F(SharedBufferIntegrationTest, CreateBufferAllUsages) {
  constexpr size_t kBufferSize = 512;

  SharedBuffer vertex_buffer;
  ASSERT_TRUE(
      vertex_buffer.create(&ctx_, kBufferSize, SharedBuffer::Usage::kVertex));
  EXPECT_TRUE(vertex_buffer.isValid());

  SharedBuffer index_buffer;
  ASSERT_TRUE(
      index_buffer.create(&ctx_, kBufferSize, SharedBuffer::Usage::kIndex));
  EXPECT_TRUE(index_buffer.isValid());

  SharedBuffer storage_buffer;
  ASSERT_TRUE(
      storage_buffer.create(&ctx_, kBufferSize, SharedBuffer::Usage::kStorage));
  EXPECT_TRUE(storage_buffer.isValid());
}

// Test CUDA copy to buffer
TEST_F(SharedBufferIntegrationTest, CopyFromCuda) {
  SharedBuffer buffer;
  constexpr size_t kNumElements = 256;
  constexpr size_t kBufferSize = kNumElements * sizeof(float);
  CudaStreamOwning stream;

  ASSERT_TRUE(buffer.create(&ctx_, kBufferSize, SharedBuffer::Usage::kVertex));

  // Allocate and fill CUDA source data
  float* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, kBufferSize), cudaSuccess);

  std::vector<float> h_data(kNumElements);
  for (size_t i = 0; i < kNumElements; ++i) {
    h_data[i] = static_cast<float>(i);
  }
  ASSERT_EQ(
      cudaMemcpy(d_src, h_data.data(), kBufferSize, cudaMemcpyHostToDevice),
      cudaSuccess);

  // Copy to shared buffer
  EXPECT_TRUE(buffer.copyFromCuda(d_src, kBufferSize, stream));

  // Verify by copying back from shared buffer's CUDA pointer
  stream.synchronize();
  std::vector<float> h_result(kNumElements);
  ASSERT_EQ(cudaMemcpy(h_result.data(), buffer.cudaPtr(), kBufferSize,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
  }

  cudaFree(d_src);
}

// Test buffer resize
TEST_F(SharedBufferIntegrationTest, ResizeBuffer) {
  SharedBuffer buffer;
  constexpr size_t kInitialSize = 512;
  constexpr size_t kLargerSize = 2048;

  ASSERT_TRUE(buffer.create(&ctx_, kInitialSize, SharedBuffer::Usage::kVertex));
  EXPECT_EQ(buffer.size(), kInitialSize);

  // Resize to larger size
  ASSERT_TRUE(buffer.resize(kLargerSize));
  EXPECT_GE(buffer.size(), kLargerSize);
  EXPECT_TRUE(buffer.isValid());
  EXPECT_NE(buffer.cudaPtr(), nullptr);
}

// Test resize to same or smaller size (should be no-op)
TEST_F(SharedBufferIntegrationTest, ResizeSameOrSmaller) {
  SharedBuffer buffer;
  constexpr size_t kInitialSize = 1024;

  ASSERT_TRUE(buffer.create(&ctx_, kInitialSize, SharedBuffer::Usage::kVertex));

  // Resize to same size
  ASSERT_TRUE(buffer.resize(kInitialSize));
  EXPECT_EQ(buffer.size(), kInitialSize);

  // Resize to smaller size (should be no-op)
  ASSERT_TRUE(buffer.resize(512));
  EXPECT_EQ(buffer.size(), kInitialSize);  // Size shouldn't change
}

// Test move semantics with valid buffer
TEST_F(SharedBufferIntegrationTest, MoveWithValidBuffer) {
  SharedBuffer buffer1;
  constexpr size_t kBufferSize = 1024;

  ASSERT_TRUE(buffer1.create(&ctx_, kBufferSize, SharedBuffer::Usage::kVertex));
  VkBuffer original_vk_buffer = buffer1.buffer();
  void* original_cuda_ptr = buffer1.cudaPtr();

  // Move to new buffer
  SharedBuffer buffer2(std::move(buffer1));

  // buffer2 should have the resources
  EXPECT_TRUE(buffer2.isValid());
  EXPECT_EQ(buffer2.buffer(), original_vk_buffer);
  EXPECT_EQ(buffer2.cudaPtr(), original_cuda_ptr);
  EXPECT_EQ(buffer2.size(), kBufferSize);

  // buffer1 should be empty
  EXPECT_FALSE(buffer1.isValid());
  EXPECT_TRUE(buffer1.buffer() == VK_NULL_HANDLE);
  EXPECT_EQ(buffer1.cudaPtr(), nullptr);
}

// ==============================================================================
// Edge Case Tests - Size Limits (with valid context)
// ==============================================================================

// Test create() fails with zero size
TEST_F(SharedBufferIntegrationTest, CreateWithZeroSize) {
  SharedBuffer buffer;

  EXPECT_FALSE(buffer.create(&ctx_, 0, SharedBuffer::Usage::kVertex));
  EXPECT_FALSE(buffer.isValid());
}

// Test moderately large buffer creation (10 MB)
TEST_F(SharedBufferIntegrationTest, CreateLargeBuffer) {
  SharedBuffer buffer;
  constexpr size_t kLargeSize = 10 * 1024 * 1024;  // 10 MB

  ASSERT_TRUE(buffer.create(&ctx_, kLargeSize, SharedBuffer::Usage::kVertex));
  EXPECT_TRUE(buffer.isValid());
  EXPECT_EQ(buffer.size(), kLargeSize);
}

// Note: kMaxBufferSizeBytes (1 GB) is not currently validated in create().
// Testing allocation of buffers near this limit may fail due to GPU memory
// constraints rather than explicit validation.

// ==============================================================================
// Resize Edge Case Tests
// ==============================================================================

// Test rapid consecutive resizes
TEST_F(SharedBufferIntegrationTest, RapidConsecutiveResizes) {
  SharedBuffer buffer;
  constexpr size_t kInitialSize = 512;

  ASSERT_TRUE(buffer.create(&ctx_, kInitialSize, SharedBuffer::Usage::kVertex));

  // Perform multiple rapid resizes (only larger sizes cause actual resize)
  const std::vector<size_t> sizes = {1024, 2048, 4096, 8192, 16384};

  for (size_t size : sizes) {
    ASSERT_TRUE(buffer.resize(size)) << "Failed to resize to " << size;
    EXPECT_TRUE(buffer.isValid());
    EXPECT_GE(buffer.size(), size);
  }
}

// Test resize preserves validity after multiple operations
TEST_F(SharedBufferIntegrationTest, ResizePreservesValidity) {
  SharedBuffer buffer;
  constexpr size_t kInitialSize = 1024;
  CudaStreamOwning stream;

  ASSERT_TRUE(buffer.create(&ctx_, kInitialSize, SharedBuffer::Usage::kVertex));

  // Write data
  float* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, sizeof(float) * 10), cudaSuccess);
  std::vector<float> test_data(10, 1.0f);
  cudaMemcpy(d_src, test_data.data(), sizeof(float) * 10,
             cudaMemcpyHostToDevice);
  ASSERT_TRUE(buffer.copyFromCuda(d_src, sizeof(float) * 10, stream));
  stream.synchronize();

  // Resize larger
  ASSERT_TRUE(buffer.resize(4096));
  EXPECT_TRUE(buffer.isValid());
  EXPECT_NE(buffer.cudaPtr(), nullptr);
  EXPECT_TRUE(buffer.buffer() != VK_NULL_HANDLE);

  // Should still be able to copy data after resize
  EXPECT_TRUE(buffer.copyFromCuda(d_src, sizeof(float) * 10, stream));
  stream.synchronize();

  cudaFree(d_src);
}

// Test buffer usability after resize
TEST_F(SharedBufferIntegrationTest, BufferUsableAfterResize) {
  SharedBuffer buffer;
  constexpr size_t kInitialSize = 256;
  constexpr size_t kLargerSize = 1024;
  constexpr size_t kNumFloats = 64;
  CudaStreamOwning stream;

  ASSERT_TRUE(buffer.create(&ctx_, kInitialSize, SharedBuffer::Usage::kVertex));

  // Resize buffer
  ASSERT_TRUE(buffer.resize(kLargerSize));

  // Allocate and fill test data
  float* d_src;
  ASSERT_EQ(cudaMalloc(&d_src, kNumFloats * sizeof(float)), cudaSuccess);
  std::vector<float> h_data(kNumFloats);
  for (size_t i = 0; i < kNumFloats; ++i) {
    h_data[i] = static_cast<float>(i);
  }
  cudaMemcpy(d_src, h_data.data(), kNumFloats * sizeof(float),
             cudaMemcpyHostToDevice);

  // Copy to resized buffer
  ASSERT_TRUE(buffer.copyFromCuda(d_src, kNumFloats * sizeof(float), stream));
  stream.synchronize();

  // Verify data by reading back from buffer's CUDA pointer
  std::vector<float> h_result(kNumFloats);
  cudaMemcpy(h_result.data(), buffer.cudaPtr(), kNumFloats * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
  }

  cudaFree(d_src);
}

// Test many rapid resizes to check for memory leaks or sync issues
TEST_F(SharedBufferIntegrationTest, StressTestRapidResizes) {
  SharedBuffer buffer;
  CudaStreamOwning stream;
  ASSERT_TRUE(buffer.create(&ctx_, 256, SharedBuffer::Usage::kVertex));

  // Perform 50 resizes with increasing sizes
  for (int i = 0; i < 50; ++i) {
    size_t new_size = 256 * (i + 2);
    ASSERT_TRUE(buffer.resize(new_size)) << "Failed at iteration " << i;

    // Verify buffer is still usable
    float dummy = 1.0f;
    float* d_dummy;
    ASSERT_EQ(cudaMalloc(&d_dummy, sizeof(float)), cudaSuccess);
    cudaMemcpy(d_dummy, &dummy, sizeof(float), cudaMemcpyHostToDevice);
    EXPECT_TRUE(buffer.copyFromCuda(d_dummy, sizeof(float), stream));
    stream.synchronize();
    cudaFree(d_dummy);
  }
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
