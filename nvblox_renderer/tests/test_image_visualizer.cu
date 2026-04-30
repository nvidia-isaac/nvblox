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
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/render_targets/vk_headless_target.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/visualizers/image_visualizer.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Test Fixture with VkContext and Headless Target
// ==============================================================================

class ImageVisualizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    std::vector<const char*> extensions;
    ASSERT_TRUE(ctx_.init("image_visualizer_test", extensions, false));
    ASSERT_TRUE(ctx_.createDevice());

    auto headless_target = std::make_unique<VkHeadlessTarget>();
    ASSERT_TRUE(headless_target->create(ctx_.device(), ctx_.physicalDevice(),
                                        kRenderWidth, kRenderHeight));
    ASSERT_TRUE(ctx_.setRenderTarget(std::move(headless_target)));

    stream_ = std::make_shared<CudaStreamOwning>();
  }

  void TearDown() override { stream_.reset(); }

  VkCommandBuffer allocateCommandBuffer() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx_.commandPool();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(ctx_.device(), &alloc_info, &cmd) !=
        VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    return cmd;
  }

  void freeCommandBuffer(VkCommandBuffer cmd) {
    if (cmd != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(ctx_.device(), ctx_.commandPool(), 1, &cmd);
    }
  }

  bool beginRenderPass(VkCommandBuffer cmd, uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
      return false;
    }

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = ctx_.renderPass();
    rp_info.framebuffer = ctx_.framebuffer(image_index);
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = ctx_.renderTargetExtent();

    VkClearValue clear_values[2];
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};
    rp_info.clearValueCount = 2;
    rp_info.pClearValues = clear_values;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    return true;
  }

  void endRenderPass(VkCommandBuffer cmd) {
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  static constexpr uint32_t kRenderWidth = 800;
  static constexpr uint32_t kRenderHeight = 600;

  VkContext ctx_;
  std::shared_ptr<CudaStream> stream_;
};

// ==============================================================================
// ImageVisualizer Tests
// ==============================================================================

TEST_F(ImageVisualizerTest, InitAndDestroy) {
  ImageVisualizer visualizer;

  ImageVisualizerConfig config{640, 480, 640, 480};
  EXPECT_TRUE(visualizer.init(&ctx_, config));
  EXPECT_TRUE(visualizer.hasData());

  visualizer.destroy();
}

TEST_F(ImageVisualizerTest, InitWithoutContextFails) {
  ImageVisualizer visualizer;
  EXPECT_FALSE(
      visualizer.init(nullptr, ImageVisualizerConfig{640, 480, 640, 480}));
}

TEST_F(ImageVisualizerTest, UpdateDepthImage) {
  ImageVisualizer visualizer;
  const uint32_t width = 64;
  const uint32_t height = 48;

  ASSERT_TRUE(visualizer.init(&ctx_, ImageVisualizerConfig(width, height)));

  float* d_depth = nullptr;
  ASSERT_EQ(cudaMalloc(&d_depth, width * height * sizeof(float)), cudaSuccess);

  std::vector<float> h_depth(width * height, 1.0f);
  ASSERT_EQ(
      cudaMemcpyAsync(d_depth, h_depth.data(), width * height * sizeof(float),
                      cudaMemcpyHostToDevice, *stream_),
      cudaSuccess);

  visualizer.updateDepth(d_depth, *stream_);
  stream_->synchronize();

  cudaFree(d_depth);
  visualizer.destroy();
}

TEST_F(ImageVisualizerTest, RenderWithDepthData) {
  ImageVisualizer visualizer;
  const uint32_t width = 64;
  const uint32_t height = 48;

  ASSERT_TRUE(visualizer.init(&ctx_, ImageVisualizerConfig(width, height)));

  float* d_depth = nullptr;
  ASSERT_EQ(cudaMalloc(&d_depth, width * height * sizeof(float)), cudaSuccess);

  std::vector<float> h_depth(width * height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      h_depth[y * width + x] = 0.5f + 0.5f * static_cast<float>(x) / width;
    }
  }
  ASSERT_EQ(
      cudaMemcpyAsync(d_depth, h_depth.data(), width * height * sizeof(float),
                      cudaMemcpyHostToDevice, *stream_),
      cudaSuccess);

  visualizer.updateDepth(d_depth, *stream_);
  stream_->synchronize();

  uint32_t image_index;
  ASSERT_TRUE(ctx_.beginFrame(&image_index));

  VkCommandBuffer cmd = allocateCommandBuffer();
  ASSERT_NE(cmd, VK_NULL_HANDLE);
  ASSERT_TRUE(beginRenderPass(cmd, image_index));

  visualizer.render(cmd, nullptr, kRenderWidth, kRenderHeight);

  endRenderPass(cmd);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  freeCommandBuffer(cmd);
  cudaFree(d_depth);
  visualizer.destroy();
}

TEST_F(ImageVisualizerTest, SetDepthRange) {
  ImageVisualizer visualizer;
  const uint32_t width = 64;
  const uint32_t height = 48;

  ASSERT_TRUE(visualizer.init(&ctx_, ImageVisualizerConfig(width, height)));

  visualizer.setDepthRange(0.1f, 10.0f);

  visualizer.destroy();
}

TEST_F(ImageVisualizerTest, InitWithZeroDimensionsFails) {
  ImageVisualizer visualizer;
  EXPECT_FALSE(visualizer.init(&ctx_, ImageVisualizerConfig{0, 480, 640, 480}));
  EXPECT_FALSE(visualizer.init(&ctx_, ImageVisualizerConfig{640, 0, 640, 480}));
  EXPECT_FALSE(visualizer.init(&ctx_, ImageVisualizerConfig{640, 480, 0, 480}));
  EXPECT_FALSE(visualizer.init(&ctx_, ImageVisualizerConfig{640, 480, 640, 0}));
}

TEST_F(ImageVisualizerTest, InitWithExcessiveDimensionsFails) {
  ImageVisualizer visualizer;
  const uint32_t too_large = kMaxTextureDimension + 1;
  EXPECT_FALSE(
      visualizer.init(&ctx_, ImageVisualizerConfig{too_large, 480, 640, 480}));
  EXPECT_FALSE(
      visualizer.init(&ctx_, ImageVisualizerConfig{640, too_large, 640, 480}));
}

TEST_F(ImageVisualizerTest, StateTracking) {
  ImageVisualizer visualizer;

  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
  ASSERT_TRUE(visualizer.init(&ctx_, ImageVisualizerConfig(64, 48)));
  EXPECT_EQ(visualizer.state(), VisualizerState::kReady);
  EXPECT_TRUE(visualizer.isReady());

  visualizer.destroy();
  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
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
