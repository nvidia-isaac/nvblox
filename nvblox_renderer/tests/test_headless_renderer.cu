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
#include "nvblox/renderer/renderer.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// VkContext Tests
// ==============================================================================

class VkContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure CUDA is available
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
  }
};

TEST_F(VkContextTest, InitWithoutExtensions) {
  VkContext ctx;

  // Initialize without window extensions (headless mode)
  std::vector<const char*> extensions;
  ASSERT_TRUE(ctx.init("test_app", extensions, false));

  EXPECT_NE(ctx.instance(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.physicalDevice(), VK_NULL_HANDLE);
  EXPECT_GE(ctx.cudaDeviceIndex(), 0);
}

TEST_F(VkContextTest, CreateDevice) {
  VkContext ctx;

  std::vector<const char*> extensions;
  ASSERT_TRUE(ctx.init("test_app", extensions, false));
  ASSERT_TRUE(ctx.createDevice());

  EXPECT_NE(ctx.device(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.graphicsQueue(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.commandPool(), VK_NULL_HANDLE);
}

TEST_F(VkContextTest, SingleTimeCommands) {
  VkContext ctx;

  std::vector<const char*> extensions;
  ASSERT_TRUE(ctx.init("test_app", extensions, false));
  ASSERT_TRUE(ctx.createDevice());

  // Test single-time command buffer
  VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
  EXPECT_NE(cmd, VK_NULL_HANDLE);

  // End command buffer (submits and waits)
  ctx.endSingleTimeCommands(cmd);
}

// ==============================================================================
// VkHeadlessTarget Tests
// ==============================================================================

class VkHeadlessTargetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    std::vector<const char*> extensions;
    ASSERT_TRUE(ctx_.init("test_app", extensions, false));
    ASSERT_TRUE(ctx_.createDevice());
  }

  void TearDown() override {
    // Context destructor handles cleanup
  }

  VkContext ctx_;
};

TEST_F(VkHeadlessTargetTest, CreateBasic) {
  VkHeadlessTarget target;

  ASSERT_TRUE(target.create(ctx_.device(), ctx_.physicalDevice(), 800, 600));

  EXPECT_NE(target.renderPass(), VK_NULL_HANDLE);
  EXPECT_EQ(target.extent().width, 800u);
  EXPECT_EQ(target.extent().height, 600u);
  EXPECT_EQ(target.imageCount(), VkHeadlessTarget::kDefaultImageCount);
  EXPECT_FALSE(target.requiresPresentation());
}

TEST_F(VkHeadlessTargetTest, AcquireAndPresent) {
  VkHeadlessTarget target;
  ASSERT_TRUE(target.create(ctx_.device(), ctx_.physicalDevice(), 800, 600));

  // Acquire first image
  uint32_t image_index = UINT32_MAX;
  ASSERT_TRUE(target.acquireImage(VK_NULL_HANDLE, &image_index));
  EXPECT_LT(image_index, target.imageCount());

  // Present (no-op for headless but should not fail)
  EXPECT_TRUE(target.presentImage(VK_NULL_HANDLE, image_index));
}

TEST_F(VkHeadlessTargetTest, Resize) {
  VkHeadlessTarget target;
  ASSERT_TRUE(target.create(ctx_.device(), ctx_.physicalDevice(), 800, 600));

  // Resize to different dimensions
  ASSERT_TRUE(target.resize(1024, 768));

  EXPECT_EQ(target.extent().width, 1024u);
  EXPECT_EQ(target.extent().height, 768u);
}

TEST_F(VkHeadlessTargetTest, FramebufferAccess) {
  VkHeadlessTarget target;
  ASSERT_TRUE(target.create(ctx_.device(), ctx_.physicalDevice(), 800, 600));

  // All framebuffers should be valid
  for (uint32_t i = 0; i < target.imageCount(); ++i) {
    EXPECT_NE(target.framebuffer(i), VK_NULL_HANDLE);
    EXPECT_NE(target.colorImage(i), VK_NULL_HANDLE);
  }

  // Invalid index should return null
  EXPECT_EQ(target.framebuffer(target.imageCount()), VK_NULL_HANDLE);
}

TEST_F(VkHeadlessTargetTest, SetAsRenderTarget) {
  auto target = std::make_unique<VkHeadlessTarget>();
  ASSERT_TRUE(target->create(ctx_.device(), ctx_.physicalDevice(), 800, 600));

  ASSERT_TRUE(ctx_.setRenderTarget(std::move(target)));
  EXPECT_TRUE(ctx_.hasRenderTarget());
  EXPECT_EQ(ctx_.renderTargetExtent().width, 800u);
  EXPECT_EQ(ctx_.renderTargetExtent().height, 600u);
}

// ==============================================================================
// NvbloxRenderer Headless Tests
// ==============================================================================

class HeadlessRendererTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
  }
};

TEST_F(HeadlessRendererTest, InitHeadless) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(800, 600));
  EXPECT_TRUE(renderer.isInitialized());
  EXPECT_TRUE(renderer.isHeadless());

  // Context should be valid
  EXPECT_NE(renderer.context(), nullptr);
  EXPECT_NE(renderer.context()->device(), VK_NULL_HANDLE);
}

TEST_F(HeadlessRendererTest, InitWithConfig) {
  NvbloxRenderer renderer;

  RendererConfig config;
  config.width = 1024;
  config.height = 768;
  config.headless = true;

  ASSERT_TRUE(renderer.init(config));
  EXPECT_TRUE(renderer.isInitialized());
  EXPECT_TRUE(renderer.isHeadless());
}

TEST_F(HeadlessRendererTest, ShouldCloseAlwaysFalse) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(800, 600));

  // Headless mode should never request close
  EXPECT_FALSE(renderer.shouldClose());
}

TEST_F(HeadlessRendererTest, InitVisualizersHeadless) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(800, 600));

  // Should be able to initialize all visualizers in headless mode
  EXPECT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));
  EXPECT_NE(renderer.pointCloudVisualizer(), nullptr);

  EXPECT_TRUE(renderer.initVisualizer(RenderMode::kMesh));
  EXPECT_NE(renderer.meshVisualizer(), nullptr);

  EXPECT_TRUE(renderer.initVisualizer(RenderMode::kImage));
  EXPECT_NE(renderer.imageVisualizer(), nullptr);
}

TEST_F(HeadlessRendererTest, RenderHeadless) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(800, 600));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  // Set render mode
  renderer.setRenderMode(RenderMode::kPointCloud);

  // Should be able to render without data (no-op)
  EXPECT_TRUE(renderer.render());
}

TEST_F(HeadlessRendererTest, ResizeHeadless) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(800, 600));

  // Resize in headless mode
  renderer.resizeWindow(1024, 768);

  // Verify new size
  EXPECT_EQ(renderer.context()->renderTargetExtent().width, 1024u);
  EXPECT_EQ(renderer.context()->renderTargetExtent().height, 768u);
}

// ==============================================================================
// Integration Test with CUDA Data
// ==============================================================================

class HeadlessRenderingIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    ASSERT_TRUE(renderer_.initHeadless(800, 600));
    stream_ = std::make_shared<CudaStreamOwning>();
  }

  void TearDown() override { stream_.reset(); }

  NvbloxRenderer renderer_;
  std::shared_ptr<CudaStream> stream_;
};

TEST_F(HeadlessRenderingIntegrationTest, RenderPointCloudFromCuda) {
  ASSERT_TRUE(renderer_.initVisualizer(RenderMode::kPointCloud));
  renderer_.setRenderMode(RenderMode::kPointCloud);

  // Create test point cloud data on GPU
  const size_t num_points = 100;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  // Initialize points (simple grid)
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

  // Update visualizer via accessor
  renderer_.pointCloudVisualizer()->updatePoints(d_points, num_points,
                                                 *stream_);

  // Synchronize before render
  stream_->synchronize();

  // Render
  EXPECT_TRUE(renderer_.render());

  cudaFree(d_points);
}

TEST_F(HeadlessRenderingIntegrationTest, RenderMeshFromCuda) {
  ASSERT_TRUE(renderer_.initVisualizer(RenderMode::kMesh));
  renderer_.setRenderMode(RenderMode::kMesh);

  // Create a simple triangle
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  // Allocate GPU memory
  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);

  // Initialize data
  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f,   // vertex 0
                                    1.0f, 0.0f, 1.0f,   // vertex 1
                                    0.5f, 1.0f, 1.0f};  // vertex 2
  std::vector<uint8_t> h_colors = {255, 0,   0,         // red
                                   0,   255, 0,         // green
                                   0,   0,   255};      // blue
  std::vector<int> h_triangles = {0, 1, 2};

  ASSERT_EQ(cudaMemcpyAsync(d_positions, h_positions.data(),
                            h_positions.size() * sizeof(float),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(d_colors, h_colors.data(),
                            h_colors.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(d_triangles, h_triangles.data(),
                            h_triangles.size() * sizeof(int),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // Update visualizer via accessor
  renderer_.meshVisualizer()->updateMesh(d_positions, d_colors, d_triangles,
                                         num_vertices, num_triangles, *stream_);

  // Synchronize before render
  stream_->synchronize();

  // Render
  EXPECT_TRUE(renderer_.render());

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
}

TEST_F(HeadlessRenderingIntegrationTest, RenderMultipleFrames) {
  ASSERT_TRUE(renderer_.initVisualizer(RenderMode::kPointCloud));
  renderer_.setRenderMode(RenderMode::kPointCloud);

  // Render multiple frames to test frame synchronization
  for (int frame = 0; frame < 10; ++frame) {
    EXPECT_TRUE(renderer_.render()) << "Failed on frame " << frame;
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
