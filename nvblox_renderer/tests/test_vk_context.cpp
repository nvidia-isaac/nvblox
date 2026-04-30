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

#include <memory>
#include <type_traits>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_render_target.h"

namespace nvblox {
namespace renderer {
namespace test {

// =============================================================================
// MockRenderTarget - Lightweight IVkRenderTarget for testing VkContext
// without depending on VkHeadlessTarget (MR 4).
// =============================================================================

class MockRenderTarget : public IVkRenderTarget {
 public:
  explicit MockRenderTarget(uint32_t image_count = 3, uint32_t width = 64,
                            uint32_t height = 64)
      : image_count_(image_count) {
    extent_.width = width;
    extent_.height = height;
  }

  ~MockRenderTarget() override = default;

  bool resize(uint32_t width, uint32_t height) override {
    if (width == 0 || height == 0) {
      return false;
    }
    extent_.width = width;
    extent_.height = height;
    return true;
  }

  void destroy() override {}

  bool acquireImage(VkSemaphore /*semaphore*/, uint32_t* image_index) override {
    if (!image_index || image_count_ == 0) {
      return false;
    }
    *image_index = current_image_;
    current_image_ = (current_image_ + 1) % image_count_;
    return true;
  }

  bool presentImage(VkSemaphore /*wait_semaphore*/,
                    uint32_t /*image_index*/) override {
    return true;
  }

  VkRenderPass renderPass() const override { return VK_NULL_HANDLE; }

  VkFramebuffer framebuffer(uint32_t /*index*/) const override {
    return VK_NULL_HANDLE;
  }

  VkExtent2D extent() const override { return extent_; }

  VkFormat colorFormat() const override { return VK_FORMAT_R8G8B8A8_SRGB; }

  uint32_t imageCount() const override { return image_count_; }

  bool requiresPresentation() const override { return false; }

 private:
  uint32_t image_count_ = 3;
  uint32_t current_image_ = 0;
  VkExtent2D extent_ = {0, 0};
};

// =============================================================================
// VkContext Unit Tests - No GPU required
// =============================================================================

class VkContextTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default state before init()
TEST_F(VkContextTest, DefaultState) {
  VkContext ctx;

  EXPECT_TRUE(ctx.instance() == VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.physicalDevice() == VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.device() == VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.graphicsQueue() == VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.commandPool() == VK_NULL_HANDLE);
  EXPECT_TRUE(ctx.pipelineCache() == VK_NULL_HANDLE);
  EXPECT_EQ(ctx.cudaDeviceIndex(), -1);
  EXPECT_FALSE(ctx.hasRenderTarget());
  EXPECT_EQ(ctx.renderTarget(), nullptr);
}

// Test hasRenderTarget before setting a target
TEST_F(VkContextTest, HasRenderTargetBeforeSet) {
  VkContext ctx;

  EXPECT_FALSE(ctx.hasRenderTarget());
  EXPECT_EQ(ctx.renderTarget(), nullptr);
}

// Test setRenderTarget with null pointer returns false
TEST_F(VkContextTest, SetRenderTargetNull) {
  VkContext ctx;

  EXPECT_FALSE(ctx.setRenderTarget(nullptr));
  EXPECT_FALSE(ctx.hasRenderTarget());
}

// Test beginFrame with null image_index returns false
TEST_F(VkContextTest, BeginFrameNullImageIndex) {
  VkContext ctx;

  EXPECT_FALSE(ctx.beginFrame(nullptr));
}

// Test VkContext is non-copyable
TEST_F(VkContextTest, NonCopyable) {
  static_assert(!std::is_copy_constructible_v<VkContext>,
                "VkContext must not be copy-constructible");
  static_assert(!std::is_copy_assignable_v<VkContext>,
                "VkContext must not be copy-assignable");
}

// =============================================================================
// VkContext Integration Tests - Require Vulkan device
// =============================================================================

class VkContextIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<const char*> extensions;
    if (!ctx_.init("test_vk_context", extensions, false)) {
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

// Test init + createDevice lifecycle and verify accessors
TEST_F(VkContextIntegrationTest, InitAndCreateDevice) {
  // After SetUp, instance, device, etc. should all be valid
  EXPECT_TRUE(ctx_.instance() != VK_NULL_HANDLE);
  EXPECT_TRUE(ctx_.physicalDevice() != VK_NULL_HANDLE);
  EXPECT_TRUE(ctx_.device() != VK_NULL_HANDLE);
  EXPECT_TRUE(ctx_.graphicsQueue() != VK_NULL_HANDLE);
  EXPECT_TRUE(ctx_.commandPool() != VK_NULL_HANDLE);

  // CUDA interop info should be populated
  EXPECT_GE(ctx_.cudaDeviceIndex(), 0);
  EXPECT_NE(ctx_.cudaDeviceUuid(), nullptr);

  // No render target yet
  EXPECT_FALSE(ctx_.hasRenderTarget());
}

// Test waitIdle before setting a render target (no work, should not crash)
TEST_F(VkContextIntegrationTest, WaitIdleBeforeRenderTarget) {
  // Should complete without error - device exists but no render target or work
  ctx_.waitIdle();
}

// Test beginSingleTimeCommands + endSingleTimeCommands roundtrip
TEST_F(VkContextIntegrationTest, SingleTimeCommands) {
  VkCommandBuffer cmd = ctx_.beginSingleTimeCommands();
  ASSERT_TRUE(cmd != VK_NULL_HANDLE);

  // Submit and wait (no actual commands recorded, just open/close)
  ctx_.endSingleTimeCommands(cmd);
}

// Test endSingleTimeCommands with VK_NULL_HANDLE is a no-op
TEST_F(VkContextIntegrationTest, EndSingleTimeCommandsNull) {
  // Should not crash
  ctx_.endSingleTimeCommands(VK_NULL_HANDLE);
}

// Test setting a mock render target
TEST_F(VkContextIntegrationTest, SetMockRenderTarget) {
  constexpr uint32_t kImageCount = 3;
  auto target = std::make_unique<MockRenderTarget>(kImageCount, 64, 64);

  ASSERT_TRUE(ctx_.setRenderTarget(std::move(target)));
  EXPECT_TRUE(ctx_.hasRenderTarget());
  EXPECT_NE(ctx_.renderTarget(), nullptr);
  EXPECT_EQ(ctx_.renderTargetImageCount(), kImageCount);
  EXPECT_EQ(ctx_.renderTargetFormat(), VK_FORMAT_R8G8B8A8_SRGB);

  VkExtent2D ext = ctx_.renderTargetExtent();
  EXPECT_EQ(ext.width, 64u);
  EXPECT_EQ(ext.height, 64u);
}

// Test destroyRenderTarget clears the target
TEST_F(VkContextIntegrationTest, DestroyRenderTarget) {
  auto target = std::make_unique<MockRenderTarget>(2, 32, 32);
  ASSERT_TRUE(ctx_.setRenderTarget(std::move(target)));
  ASSERT_TRUE(ctx_.hasRenderTarget());

  ctx_.destroyRenderTarget();
  EXPECT_FALSE(ctx_.hasRenderTarget());
  EXPECT_EQ(ctx_.renderTarget(), nullptr);
}

// Test endFrame with an invalid image index
TEST_F(VkContextIntegrationTest, EndFrameInvalidImageIndex) {
  auto target = std::make_unique<MockRenderTarget>(3, 64, 64);
  ASSERT_TRUE(ctx_.setRenderTarget(std::move(target)));

  // endFrame with image_index out of bounds should return false
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  EXPECT_FALSE(ctx_.endFrame(999, cmd));
}

// Test init with validation layers enabled (may or may not have layers
// installed, but should not crash)
TEST_F(VkContextIntegrationTest, InitWithValidation) {
  VkContext ctx_with_validation;
  std::vector<const char*> extensions;
  // This may succeed or fall back to no-validation; either way, no crash.
  bool init_ok = ctx_with_validation.init("test_validation", extensions, true);
  if (init_ok) {
    EXPECT_TRUE(ctx_with_validation.instance() != VK_NULL_HANDLE);
  }
  // Context destructor cleans up
}

// Test that VkContext destructor properly cleans up all resources
TEST_F(VkContextIntegrationTest, DestructorCleansUp) {
  // Create a fully-initialized context in a nested scope
  {
    VkContext scoped_ctx;
    std::vector<const char*> extensions;
    ASSERT_TRUE(scoped_ctx.init("test_destructor", extensions, false));
    ASSERT_TRUE(scoped_ctx.createDevice());

    auto target = std::make_unique<MockRenderTarget>(2, 32, 32);
    ASSERT_TRUE(scoped_ctx.setRenderTarget(std::move(target)));
    ASSERT_TRUE(scoped_ctx.hasRenderTarget());
  }
  // scoped_ctx goes out of scope here - destructor should clean up without
  // crashes or validation errors.
}

// Test beginFrame without a render target set returns false
TEST_F(VkContextIntegrationTest, BeginFrameWithoutRenderTarget) {
  uint32_t image_index = 0;
  EXPECT_FALSE(ctx_.beginFrame(&image_index));
}

// Test endFrame with zero image count returns false
TEST_F(VkContextIntegrationTest, EndFrameZeroImageCount) {
  // Mock target with 0 images
  auto target = std::make_unique<MockRenderTarget>(0, 64, 64);
  ASSERT_TRUE(ctx_.setRenderTarget(std::move(target)));

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  EXPECT_FALSE(ctx_.endFrame(0, cmd));
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
