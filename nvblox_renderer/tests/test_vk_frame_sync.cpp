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

#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_frame_sync.h"

namespace nvblox {
namespace renderer {
namespace test {

class VkFrameSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default state after construction
TEST_F(VkFrameSyncTest, DefaultState) {
  VkFrameSync sync;

  // Should not be valid without calling create()
  EXPECT_FALSE(sync.isValid());

  // Current frame should be 0
  EXPECT_EQ(sync.currentFrame(), 0u);

  // renderTargetImageCount should be 0 (no semaphores created)
  EXPECT_EQ(sync.renderTargetImageCount(), 0u);
}

// Test that create() fails with null device
TEST_F(VkFrameSyncTest, CreateWithNullDevice) {
  VkFrameSync sync;

  EXPECT_FALSE(sync.create(VK_NULL_HANDLE));
  EXPECT_FALSE(sync.isValid());
}

// Test that createRenderTargetSemaphores fails without create()
TEST_F(VkFrameSyncTest, CreateSemaphoresWithoutDevice) {
  VkFrameSync sync;

  // Should fail because device is not set
  EXPECT_FALSE(sync.createRenderTargetSemaphores(3));
}

// Test that destroy() is safe on uninitialized object
TEST_F(VkFrameSyncTest, DestroyUninitialized) {
  VkFrameSync sync;

  // Should not crash
  sync.destroy();
  EXPECT_FALSE(sync.isValid());
}

// Test that destructor doesn't crash on uninitialized object
TEST_F(VkFrameSyncTest, DestructorUninitialized) {
  // Create and immediately destroy - should not crash
  VkFrameSync* sync = new VkFrameSync();
  delete sync;
}

// Test kMaxFramesInFlight constant
TEST_F(VkFrameSyncTest, MaxFramesInFlight) {
  // Verify the constant is set to a reasonable value
  EXPECT_EQ(VkFrameSync::kMaxFramesInFlight, 2);
}

// Test kFenceTimeoutNs constant
TEST_F(VkFrameSyncTest, FenceTimeout) {
  // Verify timeout is 5 seconds in nanoseconds
  EXPECT_EQ(VkFrameSync::kFenceTimeoutNs, 5'000'000'000ull);
}

// ==============================================================================
// Integration tests that require a Vulkan device
// ==============================================================================

class VkFrameSyncIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Vulkan context in headless mode
    std::vector<const char*> extensions;
    if (!ctx_.init("test_vk_frame_sync", extensions, false)) {
      GTEST_SKIP() << "Failed to initialize Vulkan - no GPU available";
    }
    if (!ctx_.createDevice()) {
      GTEST_SKIP() << "Failed to create Vulkan device";
    }
  }

  void TearDown() override {
    // Context cleans itself up via destructor
  }

  VkContext ctx_;
};

// Test creating VkFrameSync with a valid device
TEST_F(VkFrameSyncIntegrationTest, CreateWithValidDevice) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  EXPECT_TRUE(sync.isValid());
  EXPECT_EQ(sync.currentFrame(), 0u);

  // Without render target semaphores, count should still be 0
  EXPECT_EQ(sync.renderTargetImageCount(), 0u);
}

// Test creating render target semaphores
TEST_F(VkFrameSyncIntegrationTest, CreateRenderTargetSemaphores) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));

  // Create semaphores for 3 swapchain images
  constexpr uint32_t kImageCount = 3;
  ASSERT_TRUE(sync.createRenderTargetSemaphores(kImageCount));

  EXPECT_EQ(sync.renderTargetImageCount(), kImageCount);

  // Verify render finished semaphores are valid (one per image)
  for (uint32_t i = 0; i < kImageCount; ++i) {
    EXPECT_TRUE(sync.renderFinishedSemaphore(i) != VK_NULL_HANDLE);
  }

  // Verify current frame semaphores and fences are valid
  EXPECT_TRUE(sync.currentImageAvailableSemaphore() != VK_NULL_HANDLE);
  EXPECT_TRUE(sync.currentInFlightFence() != VK_NULL_HANDLE);
}

// Test frame advancement
TEST_F(VkFrameSyncIntegrationTest, AdvanceFrame) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Initial frame is 0
  EXPECT_EQ(sync.currentFrame(), 0u);

  // Advance frame
  sync.advanceFrame();
  EXPECT_EQ(sync.currentFrame(), 1u);

  // Advance again - should wrap around due to kMaxFramesInFlight
  sync.advanceFrame();
  EXPECT_EQ(sync.currentFrame(), 0u);
}

// Test waiting for fence (should return immediately when fence is signaled)
TEST_F(VkFrameSyncIntegrationTest, WaitForCurrentFrame) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Wait should succeed - fences are created in signaled state
  EXPECT_TRUE(sync.waitForCurrentFrame());
}

// Test destroying and recreating
TEST_F(VkFrameSyncIntegrationTest, DestroyAndRecreate) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Destroy
  sync.destroy();
  EXPECT_FALSE(sync.isValid());
  EXPECT_EQ(sync.renderTargetImageCount(), 0u);

  // Recreate
  ASSERT_TRUE(sync.create(ctx_.device()));
  EXPECT_TRUE(sync.isValid());

  ASSERT_TRUE(sync.createRenderTargetSemaphores(2));
  EXPECT_EQ(sync.renderTargetImageCount(), 2u);
}

// Test recreating render target semaphores with different count
TEST_F(VkFrameSyncIntegrationTest, RecreateSemaphores) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));

  // Create with 2 images
  ASSERT_TRUE(sync.createRenderTargetSemaphores(2));
  EXPECT_EQ(sync.renderTargetImageCount(), 2u);

  // Recreate with 4 images
  ASSERT_TRUE(sync.createRenderTargetSemaphores(4));
  EXPECT_EQ(sync.renderTargetImageCount(), 4u);

  // Verify all render finished semaphores are valid
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(sync.renderFinishedSemaphore(i) != VK_NULL_HANDLE);
  }

  // Verify current frame semaphores are valid
  EXPECT_TRUE(sync.currentImageAvailableSemaphore() != VK_NULL_HANDLE);
}

// Test resetCurrentFence after waiting on a signaled fence
TEST_F(VkFrameSyncIntegrationTest, ResetCurrentFence) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Fences are created in signaled state, wait first
  ASSERT_TRUE(sync.waitForCurrentFrame());

  // Reset should succeed on a signaled fence
  EXPECT_TRUE(sync.resetCurrentFence());
}

// Test markImageInFlight and waitForImageInFlight cycle
TEST_F(VkFrameSyncIntegrationTest, MarkAndWaitImageInFlight) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Mark image 0 as in flight (associates current frame's fence with image 0)
  sync.markImageInFlight(0);

  // Advance to the next frame
  sync.advanceFrame();

  // Wait for image 0's fence - it was the previous frame's fence which is
  // still in signaled state (no actual GPU work submitted), so this should
  // return immediately.
  EXPECT_TRUE(sync.waitForImageInFlight(0));
}

// Test waitForImageInFlight with no prior markImageInFlight (null fence path)
TEST_F(VkFrameSyncIntegrationTest, WaitForImageInFlightNoFence) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // No mark was called, images_in_flight_[0] is VK_NULL_HANDLE.
  // Should return true immediately without waiting.
  EXPECT_TRUE(sync.waitForImageInFlight(0));
}

// Test waitForImageInFlight with an out-of-range image index
TEST_F(VkFrameSyncIntegrationTest, WaitForImageInFlightOutOfRange) {
  VkFrameSync sync;

  ASSERT_TRUE(sync.create(ctx_.device()));
  ASSERT_TRUE(sync.createRenderTargetSemaphores(3));

  // Index 999 is far beyond images_in_flight_ size.
  // Should return true (out-of-range path, no fence to wait on).
  EXPECT_TRUE(sync.waitForImageInFlight(999));
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
