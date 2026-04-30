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
#include "nvblox/renderer/visualizers/image_visualizer.h"
#include "nvblox/renderer/visualizers/mesh_visualizer.h"
#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Test Fixture with VkContext and Headless Target
// ==============================================================================

class VisualizerTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure CUDA is available
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    // Initialize Vulkan context
    std::vector<const char*> extensions;
    ASSERT_TRUE(ctx_.init("visualizer_test", extensions, false));
    ASSERT_TRUE(ctx_.createDevice());

    // Create headless render target
    auto headless_target = std::make_unique<VkHeadlessTarget>();
    ASSERT_TRUE(headless_target->create(ctx_.device(), ctx_.physicalDevice(),
                                        kRenderWidth, kRenderHeight));
    ASSERT_TRUE(ctx_.setRenderTarget(std::move(headless_target)));

    // Create CUDA stream
    stream_ = std::make_shared<CudaStreamOwning>();
  }

  void TearDown() override { stream_.reset(); }

  // Helper to allocate a command buffer for rendering tests
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

  // Begin command buffer with render pass
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

class PointCloudVisualizerTest : public VisualizerTestFixture {};

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
  // Verify setting doesn't crash - actual value is internal

  visualizer.destroy();
}

TEST_F(PointCloudVisualizerTest, UpdatePoints) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create test data
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

  // Create test data
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

  // Render
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
  // Should be no-op when hasData() is false
  visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

  endRenderPass(cmd);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  freeCommandBuffer(cmd);
  visualizer.destroy();
}

// ==============================================================================
// MeshVisualizer Tests
// ==============================================================================

class MeshVisualizerTest : public VisualizerTestFixture {};

TEST_F(MeshVisualizerTest, InitAndDestroy) {
  MeshVisualizer visualizer;

  EXPECT_TRUE(visualizer.init(&ctx_));
  EXPECT_FALSE(visualizer.hasData());

  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, InitWithoutContextFails) {
  MeshVisualizer visualizer;
  EXPECT_FALSE(visualizer.init(nullptr));
}

TEST_F(MeshVisualizerTest, UpdateMesh) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create a simple triangle
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
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

  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_);
  stream_->synchronize();

  EXPECT_TRUE(visualizer.hasData());

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, RenderWithData) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create a simple triangle
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
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

  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_);
  stream_->synchronize();

  // Render
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
  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  visualizer.destroy();
}

// ==============================================================================
// ImageVisualizer Tests
// ==============================================================================

class ImageVisualizerTest : public VisualizerTestFixture {};

TEST_F(ImageVisualizerTest, InitAndDestroy) {
  ImageVisualizer visualizer;

  ImageVisualizerConfig config{640, 480, 640, 480};
  EXPECT_TRUE(visualizer.init(&ctx_, config));
  // hasData() returns true after init because textures are created
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

  // Create test depth data
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

  // Create test depth data
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

  // Render
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

  // Set depth range
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

// ==============================================================================
// Multiple Frame Rendering Tests
// ==============================================================================

TEST_F(PointCloudVisualizerTest, RenderMultipleFrames) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create test data
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

  ViewCamera camera;
  camera.setAspect(static_cast<float>(kRenderWidth) / kRenderHeight);

  // Render multiple frames
  for (int frame = 0; frame < 5; ++frame) {
    uint32_t image_index;
    ASSERT_TRUE(ctx_.beginFrame(&image_index)) << "Failed on frame " << frame;

    VkCommandBuffer cmd = allocateCommandBuffer();
    ASSERT_NE(cmd, VK_NULL_HANDLE);
    ASSERT_TRUE(beginRenderPass(cmd, image_index));

    visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

    endRenderPass(cmd);
    ASSERT_TRUE(ctx_.endFrame(image_index, cmd)) << "Failed on frame " << frame;

    freeCommandBuffer(cmd);
  }

  cudaFree(d_points);
  visualizer.destroy();
}

// ==============================================================================
// Buffer Auto-Resize Tests
// ==============================================================================

TEST_F(PointCloudVisualizerTest, BufferAutoResize) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Start with a small number of points
  const size_t small_count = 100;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, small_count * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  std::vector<PointCloudVisualizer::Point> h_points(small_count);
  for (size_t i = 0; i < small_count; ++i) {
    h_points[i] = {static_cast<float>(i), 0.0f, 0.0f, 255, 0, 0, 255};
  }
  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            small_count * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  visualizer.updatePoints(d_points, small_count, *stream_);
  stream_->synchronize();
  EXPECT_EQ(visualizer.numPoints(), small_count);

  cudaFree(d_points);

  // Now update with a much larger number to trigger resize
  // Use a count larger than kDefaultPointBufferSize (100000)
  const size_t large_count = 200000;
  ASSERT_EQ(
      cudaMalloc(&d_points, large_count * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  h_points.resize(large_count);
  for (size_t i = 0; i < large_count; ++i) {
    h_points[i] = {static_cast<float>(i), 0.0f, 0.0f, 255, 0, 0, 255};
  }
  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            large_count * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // This should trigger a buffer resize
  visualizer.updatePoints(d_points, large_count, *stream_);
  stream_->synchronize();

  EXPECT_EQ(visualizer.numPoints(), large_count);
  EXPECT_TRUE(visualizer.hasData());

  cudaFree(d_points);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, BufferAutoResize) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Start with a small mesh
  const size_t small_vertices = 100;
  const size_t small_triangles = 100;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, small_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, small_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, small_triangles * 3 * sizeof(int)),
            cudaSuccess);

  std::vector<float> h_positions(small_vertices * 3, 0.0f);
  std::vector<uint8_t> h_colors(small_vertices * 3, 128);
  std::vector<int> h_triangles(small_triangles * 3);
  for (size_t i = 0; i < small_triangles * 3; ++i) {
    h_triangles[i] = static_cast<int>(i % small_vertices);
  }

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

  visualizer.updateMesh(d_positions, d_colors, d_triangles, small_vertices,
                        small_triangles, *stream_);
  stream_->synchronize();
  EXPECT_EQ(visualizer.numVertices(), small_vertices);

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);

  // Now update with larger mesh to trigger resize
  const size_t large_vertices = 200000;
  const size_t large_triangles = 200000;

  ASSERT_EQ(cudaMalloc(&d_positions, large_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, large_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, large_triangles * 3 * sizeof(int)),
            cudaSuccess);

  h_positions.resize(large_vertices * 3, 0.0f);
  h_colors.resize(large_vertices * 3, 128);
  h_triangles.resize(large_triangles * 3);
  for (size_t i = 0; i < large_triangles * 3; ++i) {
    h_triangles[i] = static_cast<int>(i % large_vertices);
  }

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

  visualizer.updateMesh(d_positions, d_colors, d_triangles, large_vertices,
                        large_triangles, *stream_);
  stream_->synchronize();

  EXPECT_EQ(visualizer.numVertices(), large_vertices);
  EXPECT_TRUE(visualizer.hasData());

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  visualizer.destroy();
}

// ==============================================================================
// State Tracking Tests
// ==============================================================================

TEST_F(PointCloudVisualizerTest, StateTracking) {
  PointCloudVisualizer visualizer;

  // Initially uninitialized
  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
  EXPECT_FALSE(visualizer.isReady());

  // After successful init, should be ready
  ASSERT_TRUE(visualizer.init(&ctx_));
  EXPECT_EQ(visualizer.state(), VisualizerState::kReady);
  EXPECT_TRUE(visualizer.isReady());

  // After destroy, should be uninitialized again
  visualizer.destroy();
  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
  EXPECT_FALSE(visualizer.isReady());
}

TEST_F(PointCloudVisualizerTest, StateTrackingOnInitFailure) {
  PointCloudVisualizer visualizer;

  // Init with null context should fail and set error state
  EXPECT_FALSE(visualizer.init(nullptr));
  EXPECT_EQ(visualizer.state(), VisualizerState::kError);
  EXPECT_FALSE(visualizer.isReady());
}

TEST_F(MeshVisualizerTest, StateTracking) {
  MeshVisualizer visualizer;

  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
  ASSERT_TRUE(visualizer.init(&ctx_));
  EXPECT_EQ(visualizer.state(), VisualizerState::kReady);
  EXPECT_TRUE(visualizer.isReady());

  visualizer.destroy();
  EXPECT_EQ(visualizer.state(), VisualizerState::kUninitialized);
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

// ==============================================================================
// Empty/Zero Data Tests
// ==============================================================================

TEST_F(PointCloudVisualizerTest, UpdateWithZeroPoints) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // First add some points
  const size_t num_points = 100;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  std::vector<PointCloudVisualizer::Point> h_points(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    h_points[i] = {static_cast<float>(i), 0.0f, 0.0f, 255, 0, 0, 255};
  }
  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            num_points * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  visualizer.updatePoints(d_points, num_points, *stream_);
  stream_->synchronize();
  EXPECT_TRUE(visualizer.hasData());
  EXPECT_EQ(visualizer.numPoints(), num_points);

  // Now update with zero points - should clear data
  visualizer.updatePoints(d_points, 0, *stream_);
  stream_->synchronize();
  EXPECT_FALSE(visualizer.hasData());
  EXPECT_EQ(visualizer.numPoints(), 0u);

  cudaFree(d_points);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, UpdateWithEmptyMesh) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // First add a mesh
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
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

  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_);
  stream_->synchronize();
  EXPECT_TRUE(visualizer.hasData());

  // Update with empty mesh - should clear data
  visualizer.updateMesh(d_positions, d_colors, d_triangles, 0, 0, *stream_);
  stream_->synchronize();
  EXPECT_FALSE(visualizer.hasData());
  EXPECT_EQ(visualizer.numVertices(), 0u);
  EXPECT_EQ(visualizer.numTriangles(), 0u);

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  visualizer.destroy();
}

// ==============================================================================
// MeshVisualizer Hybrid Rendering Tests (Texture + UV)
// ==============================================================================

TEST_F(MeshVisualizerTest, InitialStateNoTexture) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  EXPECT_FALSE(visualizer.hasTexture());

  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, UpdateTextureValid) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create a small 4x4 RGBA texture on device
  const uint32_t width = 4;
  const uint32_t height = 4;
  const size_t num_pixels = width * height;
  const size_t data_size = num_pixels * 4;  // RGBA

  uint8_t* d_texture = nullptr;
  ASSERT_EQ(cudaMalloc(&d_texture, data_size), cudaSuccess);

  std::vector<uint8_t> h_texture(data_size, 128);  // Gray
  ASSERT_EQ(cudaMemcpyAsync(d_texture, h_texture.data(), data_size,
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);
  stream_->synchronize();

  EXPECT_TRUE(visualizer.updateTexture(d_texture, width, height, *stream_));
  stream_->synchronize();

  EXPECT_TRUE(visualizer.hasTexture());

  cudaFree(d_texture);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, UpdateTextureInvalidParams) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Null source
  EXPECT_FALSE(visualizer.updateTexture(nullptr, 4, 4, *stream_));

  // Zero dimensions
  uint8_t dummy = 0;
  EXPECT_FALSE(visualizer.updateTexture(&dummy, 0, 4, *stream_));
  EXPECT_FALSE(visualizer.updateTexture(&dummy, 4, 0, *stream_));

  EXPECT_FALSE(visualizer.hasTexture());

  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, UpdateTextureResize) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Create initial 4x4 texture
  uint8_t* d_texture = nullptr;
  ASSERT_EQ(cudaMalloc(&d_texture, 4 * 4 * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_texture, 128, 4 * 4 * 4), cudaSuccess);

  EXPECT_TRUE(visualizer.updateTexture(d_texture, 4, 4, *stream_));
  stream_->synchronize();
  EXPECT_TRUE(visualizer.hasTexture());

  // Resize to 8x8
  cudaFree(d_texture);
  ASSERT_EQ(cudaMalloc(&d_texture, 8 * 8 * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_texture, 200, 8 * 8 * 4), cudaSuccess);

  EXPECT_TRUE(visualizer.updateTexture(d_texture, 8, 8, *stream_));
  stream_->synchronize();
  EXPECT_TRUE(visualizer.hasTexture());

  cudaFree(d_texture);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, UpdateMeshWithUVs) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;
  float* d_uvs = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_uvs, num_vertices * 2 * sizeof(float)), cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
  std::vector<int> h_triangles = {0, 1, 2};
  std::vector<float> h_uvs = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};

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
  ASSERT_EQ(cudaMemcpyAsync(d_uvs, h_uvs.data(), h_uvs.size() * sizeof(float),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_, d_uvs);
  stream_->synchronize();

  EXPECT_TRUE(visualizer.hasData());
  EXPECT_EQ(visualizer.numVertices(), num_vertices);
  EXPECT_EQ(visualizer.numTriangles(), num_triangles);

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  cudaFree(d_uvs);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, RenderWithTextureAndUVs) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Upload a simple texture
  const uint32_t tex_width = 4;
  const uint32_t tex_height = 4;
  uint8_t* d_texture = nullptr;
  ASSERT_EQ(cudaMalloc(&d_texture, tex_width * tex_height * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_texture, 255, tex_width * tex_height * 4),
            cudaSuccess);
  EXPECT_TRUE(
      visualizer.updateTexture(d_texture, tex_width, tex_height, *stream_));
  stream_->synchronize();
  EXPECT_TRUE(visualizer.hasTexture());

  // Upload mesh with UVs
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;
  float* d_uvs = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_uvs, num_vertices * 2 * sizeof(float)), cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
  std::vector<int> h_triangles = {0, 1, 2};
  std::vector<float> h_uvs = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};

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
  ASSERT_EQ(cudaMemcpyAsync(d_uvs, h_uvs.data(), h_uvs.size() * sizeof(float),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_, d_uvs);
  stream_->synchronize();

  // Render — should use texture path since hasTexture is true and UVs are valid
  uint32_t image_index;
  ASSERT_TRUE(ctx_.beginFrame(&image_index));

  VkCommandBuffer cmd = allocateCommandBuffer();
  ASSERT_NE(cmd, VK_NULL_HANDLE);
  ASSERT_TRUE(beginRenderPass(cmd, image_index));

  ViewCamera camera;
  camera.setAspect(static_cast<float>(kRenderWidth) / kRenderHeight);

  // Should not crash — exercises the full texture+UV render path
  visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

  vkCmdEndRenderPass(cmd);
  ASSERT_EQ(vkEndCommandBuffer(cmd), VK_SUCCESS);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  cudaFree(d_uvs);
  cudaFree(d_texture);
  visualizer.destroy();
}

TEST_F(MeshVisualizerTest, RenderMeshWithoutUVsFallsBackToVertexColor) {
  MeshVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Upload texture but NO UVs — should fall back to vertex color
  uint8_t* d_texture = nullptr;
  ASSERT_EQ(cudaMalloc(&d_texture, 4 * 4 * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_texture, 255, 4 * 4 * 4), cudaSuccess);
  EXPECT_TRUE(visualizer.updateTexture(d_texture, 4, 4, *stream_));
  stream_->synchronize();

  // Upload mesh WITHOUT UVs (uvs = nullptr)
  const size_t num_vertices = 3;
  const size_t num_triangles = 1;

  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, num_vertices * 3 * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, num_vertices * 3 * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, num_triangles * 3 * sizeof(int)),
            cudaSuccess);

  std::vector<float> h_positions = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
                                    1.0f, 0.5f, 1.0f, 1.0f};
  std::vector<uint8_t> h_colors = {255, 0, 0, 0, 255, 0, 0, 0, 255};
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

  // No UVs — should set UV sentinel (-1,-1), shader falls back to vertex color
  visualizer.updateMesh(d_positions, d_colors, d_triangles, num_vertices,
                        num_triangles, *stream_, nullptr);
  stream_->synchronize();

  // Render — should not crash, uses vertex color despite texture being bound
  uint32_t image_index;
  ASSERT_TRUE(ctx_.beginFrame(&image_index));

  VkCommandBuffer cmd = allocateCommandBuffer();
  ASSERT_NE(cmd, VK_NULL_HANDLE);
  ASSERT_TRUE(beginRenderPass(cmd, image_index));

  ViewCamera camera;
  camera.setAspect(static_cast<float>(kRenderWidth) / kRenderHeight);

  visualizer.render(cmd, &camera, kRenderWidth, kRenderHeight);

  vkCmdEndRenderPass(cmd);
  ASSERT_EQ(vkEndCommandBuffer(cmd), VK_SUCCESS);
  ASSERT_TRUE(ctx_.endFrame(image_index, cmd));

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
  cudaFree(d_texture);
  visualizer.destroy();
}

// ==============================================================================
// PointCloudVisualizer Tests (continued)
// ==============================================================================

TEST_F(PointCloudVisualizerTest, UpdateWithNullPointer) {
  PointCloudVisualizer visualizer;
  ASSERT_TRUE(visualizer.init(&ctx_));

  // Attempting to update with null pointer and non-zero count should be handled
  // gracefully (logs warning, doesn't crash)
  visualizer.updatePoints(nullptr, 100, *stream_);
  stream_->synchronize();

  // Should still be in valid state with no data
  EXPECT_FALSE(visualizer.hasData());
  EXPECT_EQ(visualizer.numPoints(), 0u);

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
