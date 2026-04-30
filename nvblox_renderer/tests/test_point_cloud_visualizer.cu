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
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Test Fixture with VkContext and Headless Target
// ==============================================================================

class PointCloudVisualizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    std::vector<const char*> extensions;
    ASSERT_TRUE(ctx_.init("pointcloud_visualizer_test", extensions, false));
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
// PointCloudVisualizer Tests
// ==============================================================================

TEST_F(PointCloudVisualizerTest, InitAndDestroy) {
  PointCloudVisualizer visualizer;

  EXPECT_TRUE(visualizer.init(&ctx_));
  EXPECT_FALSE(visualizer.hasData());
  EXPECT_EQ(visualizer.numPoints(), 0u);

  visualizer.destroy();
}

TEST_F(PointCloudVisualizerTest, InitWithoutContextFails) {
  PointCloudVisualizer visualizer;
  EXPECT_FALSE(visualizer.init(nullptr));
}

TEST_F(PointCloudVisualizerTest, SetPointSize) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  visualizer.setPointSize(5.0f);

  visualizer.destroy();
}

TEST_F(PointCloudVisualizerTest, UpdatePoints) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  const size_t num_points = 100;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  std::vector<PointCloudVisualizer::Point> h_points(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    h_points[i].x = static_cast<float>(i % 10) * 0.1f;
    h_points[i].y = static_cast<float>(i / 10) * 0.1f;
    h_points[i].z = 1.0f;
    h_points[i].r = 255;
    h_points[i].g = 128;
    h_points[i].b = 64;
    h_points[i].a = 255;
  }
  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            num_points * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  visualizer.updatePoints(d_points, num_points, *stream_);
  stream_->synchronize();

  EXPECT_TRUE(visualizer.hasData());
  EXPECT_EQ(visualizer.numPoints(), num_points);

  cudaFree(d_points);
  visualizer.destroy();
}

TEST_F(PointCloudVisualizerTest, RenderWithData) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  const size_t num_points = 10;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  std::vector<PointCloudVisualizer::Point> h_points(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    h_points[i] = {static_cast<float>(i) * 0.1f, 0.0f, 1.0f, 255, 0, 0, 255};
  }
  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            num_points * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  visualizer.updatePoints(d_points, num_points, *stream_);
  stream_->synchronize();

  uint32_t image_index;
  ASSERT_TRUE(ctx_.beginFrame(&image_index));

  VkCommandBuffer cmd = allocateCommandBuffer();
  ASSERT_NE(cmd, VK_NULL_HANDLE);
  ASSERT_TRUE(beginRenderPass(cmd, image_index));

  ViewCamera camera;
  camera.setAspect(static_cast<float>(kRenderWidth) / kRenderHeight);
  visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

  endRenderPass(cmd);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  freeCommandBuffer(cmd);
  cudaFree(d_points);
  visualizer.destroy();
}

TEST_F(PointCloudVisualizerTest, RenderWithoutData) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  uint32_t image_index;
  ASSERT_TRUE(ctx_.beginFrame(&image_index));

  VkCommandBuffer cmd = allocateCommandBuffer();
  ASSERT_NE(cmd, VK_NULL_HANDLE);
  ASSERT_TRUE(beginRenderPass(cmd, image_index));

  ViewCamera camera;
  visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

  endRenderPass(cmd);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  freeCommandBuffer(cmd);
  visualizer.destroy();
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
