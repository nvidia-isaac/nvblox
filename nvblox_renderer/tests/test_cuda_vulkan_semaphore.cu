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
#include "nvblox/renderer/core/cuda_vulkan_semaphore.h"
#include "nvblox/renderer/core/vk_context.h"

namespace nvblox {
namespace renderer {
namespace test {

class CudaVulkanSemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default state after construction
TEST_F(CudaVulkanSemaphoreTest, DefaultState) {
  CudaVulkanSemaphore sem;

  EXPECT_FALSE(sem.isValid());
  EXPECT_TRUE(sem.vulkanSemaphore() == VK_NULL_HANDLE);
  EXPECT_EQ(sem.currentSignalValue(), 0u);
}

// Test create() fails with null context
TEST_F(CudaVulkanSemaphoreTest, CreateWithNullContext) {
  CudaVulkanSemaphore sem;

  EXPECT_FALSE(sem.create(nullptr));
  EXPECT_FALSE(sem.isValid());
}

// Test destroy() is safe on uninitialized semaphore
TEST_F(CudaVulkanSemaphoreTest, DestroyUninitialized) {
  CudaVulkanSemaphore sem;

  // Should not crash
  sem.destroy();
  EXPECT_FALSE(sem.isValid());
}

// Test destructor doesn't crash on uninitialized semaphore
TEST_F(CudaVulkanSemaphoreTest, DestructorUninitialized) {
  // Create and immediately destroy - should not crash
  CudaVulkanSemaphore* sem = new CudaVulkanSemaphore();
  delete sem;
}

// Test signalFromCuda() fails on uninitialized semaphore
TEST_F(CudaVulkanSemaphoreTest, SignalUninitialized) {
  CudaVulkanSemaphore sem;
  CudaStreamOwning stream;

  EXPECT_FALSE(sem.signalFromCuda(stream));
}

// Test waitFromCuda() fails on uninitialized semaphore
TEST_F(CudaVulkanSemaphoreTest, WaitUninitialized) {
  CudaVulkanSemaphore sem;
  CudaStreamOwning stream;

  EXPECT_FALSE(sem.waitFromCuda(stream));
}

// Test move constructor
TEST_F(CudaVulkanSemaphoreTest, MoveConstructor) {
  CudaVulkanSemaphore sem1;
  // Semaphore is not initialized, but move should still work

  CudaVulkanSemaphore sem2(std::move(sem1));
  EXPECT_FALSE(sem2.isValid());
}

// Test move assignment
TEST_F(CudaVulkanSemaphoreTest, MoveAssignment) {
  CudaVulkanSemaphore sem1;
  CudaVulkanSemaphore sem2;

  sem2 = std::move(sem1);
  EXPECT_FALSE(sem2.isValid());
}

// =============================================================================
// Integration tests requiring VkContext
// =============================================================================

class CudaVulkanSemaphoreIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    // Initialize Vulkan context (init() also selects physical device)
    if (!ctx_.init("test_cuda_vulkan_semaphore", {}, true)) {
      GTEST_SKIP() << "Failed to initialize Vulkan instance";
    }
    if (!ctx_.createDevice()) {
      GTEST_SKIP() << "Failed to create Vulkan device";
    }
  }

  void TearDown() override {
    // VkContext destructor handles cleanup
  }

  VkContext ctx_;
};

// Test basic creation and destruction
TEST_F(CudaVulkanSemaphoreIntegrationTest, CreateAndDestroy) {
  CudaVulkanSemaphore sem;

  ASSERT_TRUE(sem.create(&ctx_));
  EXPECT_TRUE(sem.isValid());
  EXPECT_TRUE(sem.vulkanSemaphore() != VK_NULL_HANDLE);
  EXPECT_EQ(sem.currentSignalValue(), 0u);

  sem.destroy();
  EXPECT_FALSE(sem.isValid());
  EXPECT_TRUE(sem.vulkanSemaphore() == VK_NULL_HANDLE);
}

// Test signal increments value
TEST_F(CudaVulkanSemaphoreIntegrationTest, SignalIncrementsValue) {
  CudaVulkanSemaphore sem;
  CudaStreamOwning stream;

  ASSERT_TRUE(sem.create(&ctx_));
  EXPECT_EQ(sem.currentSignalValue(), 0u);

  // Signal should increment value
  ASSERT_TRUE(sem.signalFromCuda(stream));
  EXPECT_EQ(sem.currentSignalValue(), 1u);

  // Signal again
  ASSERT_TRUE(sem.signalFromCuda(stream));
  EXPECT_EQ(sem.currentSignalValue(), 2u);
}

// Test move with valid semaphore
TEST_F(CudaVulkanSemaphoreIntegrationTest, MoveWithValidSemaphore) {
  CudaVulkanSemaphore sem1;
  ASSERT_TRUE(sem1.create(&ctx_));

  VkSemaphore original_handle = sem1.vulkanSemaphore();
  EXPECT_TRUE(original_handle != VK_NULL_HANDLE);

  // Move to new semaphore
  CudaVulkanSemaphore sem2(std::move(sem1));

  // sem2 should have the resources
  EXPECT_TRUE(sem2.isValid());
  EXPECT_EQ(sem2.vulkanSemaphore(), original_handle);

  // sem1 should be empty
  EXPECT_FALSE(sem1.isValid());
  EXPECT_TRUE(sem1.vulkanSemaphore() == VK_NULL_HANDLE);
}

// Test move assignment with valid semaphore
TEST_F(CudaVulkanSemaphoreIntegrationTest, MoveAssignmentWithValidSemaphore) {
  CudaVulkanSemaphore sem1;
  CudaVulkanSemaphore sem2;
  ASSERT_TRUE(sem1.create(&ctx_));

  VkSemaphore original_handle = sem1.vulkanSemaphore();

  sem2 = std::move(sem1);

  // sem2 should have the resources
  EXPECT_TRUE(sem2.isValid());
  EXPECT_EQ(sem2.vulkanSemaphore(), original_handle);

  // sem1 should be empty
  EXPECT_FALSE(sem1.isValid());
}

// Test signal and wait pattern
TEST_F(CudaVulkanSemaphoreIntegrationTest, SignalAndWait) {
  CudaVulkanSemaphore sem;
  CudaStreamOwning stream;

  ASSERT_TRUE(sem.create(&ctx_));

  // Signal
  ASSERT_TRUE(sem.signalFromCuda(stream));

  // Wait (should succeed as signal was issued on same stream)
  ASSERT_TRUE(sem.waitFromCuda(stream));

  // Synchronize to ensure operations complete
  stream.synchronize();
}

// Test multiple signal operations
TEST_F(CudaVulkanSemaphoreIntegrationTest, MultipleSignals) {
  CudaVulkanSemaphore sem;
  CudaStreamOwning stream;

  ASSERT_TRUE(sem.create(&ctx_));

  // Multiple signals should work
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(sem.signalFromCuda(stream));
    EXPECT_EQ(sem.currentSignalValue(), static_cast<uint64_t>(i + 1));
  }

  stream.synchronize();
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
