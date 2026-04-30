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

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Error Check Tests - Unit tests for error handling utilities
// ==============================================================================

class ErrorCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test checkVkResult with success
TEST_F(ErrorCheckTest, CheckVkResultSuccess) {
  EXPECT_TRUE(checkVkResult(VK_SUCCESS, "test operation"));
}

// Test checkVkResult with various error codes
TEST_F(ErrorCheckTest, CheckVkResultErrors) {
  EXPECT_FALSE(checkVkResult(VK_ERROR_OUT_OF_HOST_MEMORY, "allocation"));
  EXPECT_FALSE(checkVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY, "allocation"));
  EXPECT_FALSE(checkVkResult(VK_ERROR_INITIALIZATION_FAILED, "init"));
  EXPECT_FALSE(checkVkResult(VK_ERROR_DEVICE_LOST, "device"));
  EXPECT_FALSE(checkVkResult(VK_NOT_READY, "not ready"));
  EXPECT_FALSE(checkVkResult(VK_TIMEOUT, "timeout"));
  EXPECT_FALSE(checkVkResult(VK_ERROR_OUT_OF_DATE_KHR, "swapchain"));
  EXPECT_FALSE(checkVkResult(VK_SUBOPTIMAL_KHR, "suboptimal"));
}

// Test checkVkResultWarn with success
TEST_F(ErrorCheckTest, CheckVkResultWarnSuccess) {
  EXPECT_TRUE(checkVkResultWarn(VK_SUCCESS, "test operation"));
}

// Test checkVkResultWarn with errors (should not throw, just log warning)
TEST_F(ErrorCheckTest, CheckVkResultWarnErrors) {
  EXPECT_FALSE(checkVkResultWarn(VK_ERROR_OUT_OF_HOST_MEMORY, "allocation"));
  EXPECT_FALSE(checkVkResultWarn(VK_ERROR_DEVICE_LOST, "device"));
}

// ==============================================================================
// VkUtils Tests - Unit tests for utility constants
// ==============================================================================

class VkUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test NVIDIA vendor ID constant
TEST_F(VkUtilsTest, NvidiaVendorId) { EXPECT_EQ(kNvidiaVendorId, 0x10DE); }

// Test Image2DCreateInfo default values
TEST_F(VkUtilsTest, Image2DCreateInfoDefaults) {
  Image2DCreateInfo info{};

  EXPECT_EQ(info.width, 0u);
  EXPECT_EQ(info.height, 0u);
  EXPECT_EQ(info.format, VK_FORMAT_UNDEFINED);
  EXPECT_EQ(info.usage, 0u);
  EXPECT_EQ(info.memory_properties, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// Test Image2DResult default values
TEST_F(VkUtilsTest, Image2DResultDefaults) {
  Image2DResult result{};

  EXPECT_TRUE(result.image == VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory == VK_NULL_HANDLE);
  EXPECT_TRUE(result.view == VK_NULL_HANDLE);
}

// ==============================================================================
// VkUtils Integration Tests - Require Vulkan device
// ==============================================================================

class VkUtilsIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Vulkan context in headless mode
    std::vector<const char*> extensions;
    if (!ctx_.init("test_vk_utils", extensions, false)) {
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

// Test findMemoryType returns valid type for common flags
TEST_F(VkUtilsIntegrationTest, FindMemoryTypeDeviceLocal) {
  uint32_t type = findMemoryType(ctx_.physicalDevice(), 0xFFFFFFFF,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  // Should find a valid memory type on any Vulkan-capable GPU
  EXPECT_NE(type, kMemoryTypeNotFound);
}

// Test findMemoryType returns valid type for host visible memory
TEST_F(VkUtilsIntegrationTest, FindMemoryTypeHostVisible) {
  uint32_t type = findMemoryType(ctx_.physicalDevice(), 0xFFFFFFFF,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  EXPECT_NE(type, kMemoryTypeNotFound);
}

// Test findMemoryType with impossible requirements returns kMemoryTypeNotFound
TEST_F(VkUtilsIntegrationTest, FindMemoryTypeInvalidFilter) {
  // Request with type_filter = 0 (no types allowed)
  uint32_t type = findMemoryType(ctx_.physicalDevice(), 0,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  EXPECT_EQ(type, kMemoryTypeNotFound);
}

// Test createImageView2D with null device
TEST_F(VkUtilsIntegrationTest, CreateImageViewNullDevice) {
  VkImageView view = VK_NULL_HANDLE;
  VkResult result = createImageView2D(VK_NULL_HANDLE, VK_NULL_HANDLE,
                                      VK_FORMAT_R8G8B8A8_UNORM,
                                      VK_IMAGE_ASPECT_COLOR_BIT, &view);
  EXPECT_NE(result, VK_SUCCESS);
  EXPECT_TRUE(view == VK_NULL_HANDLE);
}

// Test createDefaultSampler2D
TEST_F(VkUtilsIntegrationTest, CreateDefaultSampler) {
  VkSampler sampler = VK_NULL_HANDLE;
  VkResult result = createDefaultSampler2D(ctx_.device(), &sampler);

  EXPECT_EQ(result, VK_SUCCESS);
  EXPECT_TRUE(sampler != VK_NULL_HANDLE);

  // Cleanup
  if (sampler != VK_NULL_HANDLE) {
    vkDestroySampler(ctx_.device(), sampler, nullptr);
  }
}

// Test createDefaultSampler2D with null device
TEST_F(VkUtilsIntegrationTest, CreateDefaultSamplerNullDevice) {
  VkSampler sampler = VK_NULL_HANDLE;
  VkResult result = createDefaultSampler2D(VK_NULL_HANDLE, &sampler);
  EXPECT_NE(result, VK_SUCCESS);
}

// Test createImage2D with valid parameters
TEST_F(VkUtilsIntegrationTest, CreateImage2DWithMemory) {
  Image2DCreateInfo create_info{};
  create_info.width = 64;
  create_info.height = 64;
  create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  create_info.usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  create_info.memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  Image2DResult result{};
  bool success = createImage2D(ctx_.device(), ctx_.physicalDevice(),
                               create_info, VK_IMAGE_ASPECT_COLOR_BIT, &result);

  EXPECT_TRUE(success);
  EXPECT_TRUE(result.image != VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory != VK_NULL_HANDLE);
  EXPECT_TRUE(result.view != VK_NULL_HANDLE);

  // Cleanup
  destroyImage2D(ctx_.device(), &result);
  EXPECT_TRUE(result.image == VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory == VK_NULL_HANDLE);
  EXPECT_TRUE(result.view == VK_NULL_HANDLE);
}

// Test createImage2D without image view
TEST_F(VkUtilsIntegrationTest, CreateImage2DWithMemoryNoView) {
  Image2DCreateInfo create_info{};
  create_info.width = 32;
  create_info.height = 32;
  create_info.format = VK_FORMAT_D32_SFLOAT;
  create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  create_info.memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  Image2DResult result{};
  // aspect_mask = 0 skips view creation
  bool success = createImage2D(ctx_.device(), ctx_.physicalDevice(),
                               create_info, 0, &result);

  EXPECT_TRUE(success);
  EXPECT_TRUE(result.image != VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory != VK_NULL_HANDLE);
  EXPECT_TRUE(result.view == VK_NULL_HANDLE);  // No view created

  destroyImage2D(ctx_.device(), &result);
}

// Test createImage2D with zero dimensions
TEST_F(VkUtilsIntegrationTest, CreateImage2DWithMemoryZeroDimensions) {
  Image2DCreateInfo create_info{};
  create_info.width = 0;
  create_info.height = 0;
  create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  Image2DResult result{};
  bool success = createImage2D(ctx_.device(), ctx_.physicalDevice(),
                               create_info, VK_IMAGE_ASPECT_COLOR_BIT, &result);

  // Should fail with zero dimensions
  EXPECT_FALSE(success);
  EXPECT_TRUE(result.image == VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory == VK_NULL_HANDLE);
  EXPECT_TRUE(result.view == VK_NULL_HANDLE);
}

// Test destroyImage2D with partially initialized result
TEST_F(VkUtilsIntegrationTest, DestroyPartiallyInitialized) {
  Image2DResult result{};
  result.image = VK_NULL_HANDLE;
  result.memory = VK_NULL_HANDLE;
  result.view = VK_NULL_HANDLE;

  // Should not crash
  destroyImage2D(ctx_.device(), &result);

  // Values should remain null
  EXPECT_TRUE(result.image == VK_NULL_HANDLE);
  EXPECT_TRUE(result.memory == VK_NULL_HANDLE);
  EXPECT_TRUE(result.view == VK_NULL_HANDLE);
}

// Test destroyImage2D with null device
TEST_F(VkUtilsIntegrationTest, DestroyWithNullDevice) {
  Image2DResult result{};
  // Should not crash even with null device
  destroyImage2D(VK_NULL_HANDLE, &result);
}

// Test creating multiple images
TEST_F(VkUtilsIntegrationTest, CreateMultipleImages) {
  constexpr int kNumImages = 3;
  Image2DResult results[kNumImages];

  Image2DCreateInfo create_info{};
  create_info.width = 128;
  create_info.height = 128;
  create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  // Create multiple images
  for (int i = 0; i < kNumImages; ++i) {
    bool success =
        createImage2D(ctx_.device(), ctx_.physicalDevice(), create_info,
                      VK_IMAGE_ASPECT_COLOR_BIT, &results[i]);
    ASSERT_TRUE(success);
    EXPECT_TRUE(results[i].image != VK_NULL_HANDLE);
  }

  // Verify all images are distinct
  for (int i = 0; i < kNumImages; ++i) {
    for (int j = i + 1; j < kNumImages; ++j) {
      EXPECT_NE(results[i].image, results[j].image);
      EXPECT_NE(results[i].memory, results[j].memory);
      EXPECT_NE(results[i].view, results[j].view);
    }
  }

  // Cleanup
  for (int i = 0; i < kNumImages; ++i) {
    destroyImage2D(ctx_.device(), &results[i]);
  }
}

// Test createImageView2D with null output pointer
TEST_F(VkUtilsIntegrationTest, CreateImageViewNullOutView) {
  VkResult result =
      createImageView2D(ctx_.device(), VK_NULL_HANDLE, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_ASPECT_COLOR_BIT, nullptr);
  EXPECT_EQ(result, VK_ERROR_INITIALIZATION_FAILED);
}

// Test createDefaultSampler2D with null output pointer
TEST_F(VkUtilsIntegrationTest, CreateSamplerNullOutSampler) {
  VkResult result = createDefaultSampler2D(ctx_.device(), nullptr);
  EXPECT_EQ(result, VK_ERROR_INITIALIZATION_FAILED);
}

// Test createImage2D with null result pointer
TEST_F(VkUtilsIntegrationTest, CreateImage2DNullResult) {
  Image2DCreateInfo create_info{};
  create_info.width = 32;
  create_info.height = 32;
  create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  bool success = createImage2D(ctx_.device(), ctx_.physicalDevice(),
                               create_info, VK_IMAGE_ASPECT_COLOR_BIT, nullptr);
  EXPECT_FALSE(success);
}

// ==============================================================================
// Debug Callback Tests - Test defaultDebugCallback directly
// ==============================================================================

// Test defaultDebugCallback with warning severity returns VK_FALSE
TEST_F(VkUtilsTest, DefaultDebugCallbackWarning) {
  VkDebugUtilsMessengerCallbackDataEXT callback_data{};
  callback_data.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
  callback_data.pMessage = "test warning message";

  VkBool32 result = defaultDebugCallback(
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &callback_data, nullptr);
  EXPECT_EQ(result, VK_FALSE);
}

// Test defaultDebugCallback with error severity returns VK_FALSE
TEST_F(VkUtilsTest, DefaultDebugCallbackError) {
  VkDebugUtilsMessengerCallbackDataEXT callback_data{};
  callback_data.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
  callback_data.pMessage = "test error message";

  VkBool32 result = defaultDebugCallback(
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, &callback_data, nullptr);
  EXPECT_EQ(result, VK_FALSE);
}

// Test defaultDebugCallback with info severity (below warning) returns VK_FALSE
TEST_F(VkUtilsTest, DefaultDebugCallbackInfo) {
  VkDebugUtilsMessengerCallbackDataEXT callback_data{};
  callback_data.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
  callback_data.pMessage = "test info message";

  VkBool32 result = defaultDebugCallback(
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &callback_data, nullptr);
  EXPECT_EQ(result, VK_FALSE);
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
