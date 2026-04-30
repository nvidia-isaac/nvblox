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

/**
 * @file test_e2e_pipeline.cu
 * @brief End-to-end pipeline tests for nvblox_renderer.
 *
 * These tests verify the complete rendering pipeline from CUDA data input
 * through to rendered output. They test the integration of all components
 * working together, not just individual units.
 */

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/renderer.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace renderer {
namespace test {

// Test fixture for end-to-end pipeline tests
class E2EPipelineTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kRenderWidth = 640;
  static constexpr uint32_t kRenderHeight = 480;

  void SetUp() override {
    // Ensure CUDA is available
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    stream_ = std::make_shared<CudaStreamOwning>();
  }

  void TearDown() override { stream_.reset(); }

  std::shared_ptr<CudaStream> stream_;
};

// ==============================================================================
// Image Pipeline Tests
// ==============================================================================

TEST_F(E2EPipelineTest, DepthImagePipeline) {
  // Full pipeline: Create renderer -> init image visualizer -> update depth ->
  // render
  NvbloxRenderer renderer;

  // Initialize headless renderer
  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));
  ASSERT_TRUE(renderer.isInitialized());
  ASSERT_TRUE(renderer.isHeadless());

  // Initialize image visualizer
  const uint32_t depth_width = 320;
  const uint32_t depth_height = 240;
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));
  ASSERT_TRUE(renderer.imageVisualizer()->resizeDepthTexture(depth_width,
                                                             depth_height));

  // Create test depth data on GPU
  float* d_depth = nullptr;
  const size_t depth_pitch = depth_width * sizeof(float);
  ASSERT_EQ(cudaMalloc(&d_depth, depth_height * depth_pitch), cudaSuccess);

  // Generate synthetic depth data (gradient from near to far)
  std::vector<float> h_depth(depth_width * depth_height);
  for (uint32_t y = 0; y < depth_height; ++y) {
    for (uint32_t x = 0; x < depth_width; ++x) {
      // Depth ranges from 0.5m to 5.0m
      h_depth[y * depth_width + x] =
          0.5f + 4.5f * static_cast<float>(x) / static_cast<float>(depth_width);
    }
  }

  ASSERT_EQ(cudaMemcpyAsync(d_depth, h_depth.data(), depth_height * depth_pitch,
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // Update visualizer with depth data
  ASSERT_NE(renderer.imageVisualizer(), nullptr);
  renderer.imageVisualizer()->updateDepth(d_depth, *stream_);

  // Synchronize before render (as required by the API)
  stream_->synchronize();

  // Render frame
  renderer.setRenderMode(RenderMode::kImage);
  EXPECT_TRUE(renderer.render());

  // Cleanup
  cudaFree(d_depth);
}

TEST_F(E2EPipelineTest, RGBDImagePipeline) {
  // Full pipeline with both depth and color images
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));

  const uint32_t depth_width = 320;
  const uint32_t depth_height = 240;
  const uint32_t color_width = 640;
  const uint32_t color_height = 480;
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));
  ASSERT_TRUE(renderer.imageVisualizer()->resizeDepthTexture(depth_width,
                                                             depth_height));
  ASSERT_TRUE(renderer.imageVisualizer()->resizeColorTexture(color_width,
                                                             color_height));

  // Create depth data
  float* d_depth = nullptr;
  const size_t depth_pitch = depth_width * sizeof(float);
  ASSERT_EQ(cudaMalloc(&d_depth, depth_height * depth_pitch), cudaSuccess);

  std::vector<float> h_depth(depth_width * depth_height, 2.0f);  // Uniform 2m
  ASSERT_EQ(cudaMemcpyAsync(d_depth, h_depth.data(), depth_height * depth_pitch,
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // Create color data (RGBA)
  uint8_t* d_color = nullptr;
  const size_t color_pitch = color_width * 4;  // RGBA
  ASSERT_EQ(cudaMalloc(&d_color, color_height * color_pitch), cudaSuccess);

  std::vector<uint8_t> h_color(color_width * color_height * 4);
  for (uint32_t y = 0; y < color_height; ++y) {
    for (uint32_t x = 0; x < color_width; ++x) {
      size_t idx = (y * color_width + x) * 4;
      h_color[idx + 0] = static_cast<uint8_t>(255 * x / color_width);   // R
      h_color[idx + 1] = static_cast<uint8_t>(255 * y / color_height);  // G
      h_color[idx + 2] = 128;                                           // B
      h_color[idx + 3] = 255;                                           // A
    }
  }
  ASSERT_EQ(cudaMemcpyAsync(d_color, h_color.data(), color_height * color_pitch,
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // Update both depth and color via visualizer
  ASSERT_NE(renderer.imageVisualizer(), nullptr);
  renderer.imageVisualizer()->updateDepth(d_depth, *stream_);
  renderer.imageVisualizer()->updateColor(d_color, *stream_);

  stream_->synchronize();

  // Render
  renderer.setRenderMode(RenderMode::kImage);
  EXPECT_TRUE(renderer.render());

  // Cleanup
  cudaFree(d_depth);
  cudaFree(d_color);
}

// ==============================================================================
// Point Cloud Pipeline Tests
// ==============================================================================

TEST_F(E2EPipelineTest, PointCloudPipeline) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  // Create point cloud data
  const size_t num_points = 10000;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  // Generate a sphere of points
  std::vector<PointCloudVisualizer::Point> h_points(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    float phi =
        static_cast<float>(i) / static_cast<float>(num_points) * 2.0f * M_PI;
    float theta = std::acos(1.0f - 2.0f * static_cast<float>(i % 100) / 100.0f);
    float r = 2.0f;  // 2m radius sphere

    h_points[i].x = r * std::sin(theta) * std::cos(phi);
    h_points[i].y = r * std::sin(theta) * std::sin(phi);
    h_points[i].z = r * std::cos(theta) + 3.0f;  // Offset 3m in Z
    h_points[i].r =
        static_cast<uint8_t>(255 * (h_points[i].x + r) / (2.0f * r));
    h_points[i].g =
        static_cast<uint8_t>(255 * (h_points[i].y + r) / (2.0f * r));
    h_points[i].b =
        static_cast<uint8_t>(255 * (h_points[i].z - 3.0f + r) / (2.0f * r));
    h_points[i].a = 255;
  }

  ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                            num_points * sizeof(PointCloudVisualizer::Point),
                            cudaMemcpyHostToDevice, *stream_),
            cudaSuccess);

  // Update and render via visualizer accessor
  ASSERT_NE(renderer.pointCloudVisualizer(), nullptr);
  renderer.pointCloudVisualizer()->updatePoints(d_points, num_points, *stream_);
  stream_->synchronize();

  renderer.setRenderMode(RenderMode::kPointCloud);
  EXPECT_TRUE(renderer.render());

  // Verify visualizer state
  EXPECT_TRUE(renderer.pointCloudVisualizer()->hasData());
  EXPECT_EQ(renderer.pointCloudVisualizer()->numPoints(), num_points);

  cudaFree(d_points);
}

// ==============================================================================
// Mesh Pipeline Tests
// ==============================================================================

TEST_F(E2EPipelineTest, MeshPipeline) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kMesh));

  // Create a simple cube mesh
  const size_t num_vertices = 8;
  const size_t num_triangles = 12;

  // Cube vertices (2m cube centered at origin, offset 3m in Z)
  std::vector<float> h_positions = {
      -1.0f, -1.0f, 2.0f,  // 0
      1.0f,  -1.0f, 2.0f,  // 1
      1.0f,  1.0f,  2.0f,  // 2
      -1.0f, 1.0f,  2.0f,  // 3
      -1.0f, -1.0f, 4.0f,  // 4
      1.0f,  -1.0f, 4.0f,  // 5
      1.0f,  1.0f,  4.0f,  // 6
      -1.0f, 1.0f,  4.0f,  // 7
  };

  // Vertex colors (different color per vertex)
  std::vector<uint8_t> h_colors = {
      255, 0,   0,    // Red
      0,   255, 0,    // Green
      0,   0,   255,  // Blue
      255, 255, 0,    // Yellow
      255, 0,   255,  // Magenta
      0,   255, 255,  // Cyan
      255, 255, 255,  // White
      128, 128, 128,  // Gray
  };

  // Cube triangles (two triangles per face, 6 faces)
  std::vector<int> h_triangles = {
      0, 1, 2, 0, 2, 3,  // Front
      5, 4, 7, 5, 7, 6,  // Back
      4, 0, 3, 4, 3, 7,  // Left
      1, 5, 6, 1, 6, 2,  // Right
      3, 2, 6, 3, 6, 7,  // Top
      4, 5, 1, 4, 1, 0,  // Bottom
  };

  // Allocate GPU memory
  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;

  ASSERT_EQ(cudaMalloc(&d_positions, h_positions.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, h_colors.size() * sizeof(uint8_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, h_triangles.size() * sizeof(int)),
            cudaSuccess);

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

  // Update and render via visualizer accessor
  ASSERT_NE(renderer.meshVisualizer(), nullptr);
  renderer.meshVisualizer()->updateMesh(d_positions, d_colors, d_triangles,
                                        num_vertices, num_triangles, *stream_);
  stream_->synchronize();

  renderer.setRenderMode(RenderMode::kMesh);
  EXPECT_TRUE(renderer.render());

  // Verify visualizer state
  EXPECT_NE(renderer.meshVisualizer(), nullptr);
  EXPECT_TRUE(renderer.meshVisualizer()->hasData());
  EXPECT_EQ(renderer.meshVisualizer()->numVertices(), num_vertices);
  EXPECT_EQ(renderer.meshVisualizer()->numTriangles(), num_triangles);

  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
}

// ==============================================================================
// Multi-Frame Tests
// ==============================================================================

TEST_F(E2EPipelineTest, MultiFrameRendering) {
  // Test rendering multiple frames with updating data
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  PointCloudVisualizer::Point* d_points = nullptr;
  const size_t num_points = 1000;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  std::vector<PointCloudVisualizer::Point> h_points(num_points);

  // Render 10 frames with different data each time
  const int num_frames = 10;
  for (int frame = 0; frame < num_frames; ++frame) {
    // Generate different point positions for each frame
    for (size_t i = 0; i < num_points; ++i) {
      float t = static_cast<float>(frame) / static_cast<float>(num_frames);
      float angle =
          static_cast<float>(i) / static_cast<float>(num_points) * 2.0f * M_PI;
      h_points[i].x = std::cos(angle + t * M_PI);
      h_points[i].y = std::sin(angle + t * M_PI);
      h_points[i].z = 2.0f + t;
      h_points[i].r = static_cast<uint8_t>(255 * t);
      h_points[i].g = static_cast<uint8_t>(255 * (1.0f - t));
      h_points[i].b = 128;
      h_points[i].a = 255;
    }

    ASSERT_EQ(cudaMemcpyAsync(d_points, h_points.data(),
                              num_points * sizeof(PointCloudVisualizer::Point),
                              cudaMemcpyHostToDevice, *stream_),
              cudaSuccess);

    renderer.pointCloudVisualizer()->updatePoints(d_points, num_points,
                                                  *stream_);
    stream_->synchronize();

    renderer.setRenderMode(RenderMode::kPointCloud);
    EXPECT_TRUE(renderer.render()) << "Frame " << frame << " failed to render";
  }

  cudaFree(d_points);
}

// ==============================================================================
// Render Mode Switching Tests
// ==============================================================================

TEST_F(E2EPipelineTest, RenderModeSwitching) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));

  // Initialize all visualizers
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));
  ASSERT_TRUE(renderer.imageVisualizer()->resizeDepthTexture(320, 240));
  ASSERT_TRUE(renderer.imageVisualizer()->resizeColorTexture(320, 240));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kMesh));

  // Create minimal test data for each type
  float* d_depth = nullptr;
  ASSERT_EQ(cudaMalloc(&d_depth, 320 * 240 * sizeof(float)), cudaSuccess);
  std::vector<float> h_depth(320 * 240, 1.5f);
  cudaMemcpy(d_depth, h_depth.data(), 320 * 240 * sizeof(float),
             cudaMemcpyHostToDevice);

  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(cudaMalloc(&d_points, 100 * sizeof(PointCloudVisualizer::Point)),
            cudaSuccess);
  std::vector<PointCloudVisualizer::Point> h_points(100,
                                                    {0, 0, 2, 255, 0, 0, 255});
  cudaMemcpy(d_points, h_points.data(),
             100 * sizeof(PointCloudVisualizer::Point), cudaMemcpyHostToDevice);

  float positions[9] = {0, 0, 2, 1, 0, 2, 0.5f, 1, 2};
  uint8_t colors[9] = {255, 0, 0, 0, 255, 0, 0, 0, 255};
  int triangles[3] = {0, 1, 2};
  float* d_positions = nullptr;
  uint8_t* d_colors = nullptr;
  int* d_triangles = nullptr;
  ASSERT_EQ(cudaMalloc(&d_positions, sizeof(positions)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_colors, sizeof(colors)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_triangles, sizeof(triangles)), cudaSuccess);
  cudaMemcpy(d_positions, positions, sizeof(positions), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colors, colors, sizeof(colors), cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, triangles, sizeof(triangles), cudaMemcpyHostToDevice);

  // Update all visualizers via accessors
  renderer.imageVisualizer()->updateDepth(d_depth, *stream_);
  renderer.pointCloudVisualizer()->updatePoints(d_points, 100, *stream_);
  renderer.meshVisualizer()->updateMesh(d_positions, d_colors, d_triangles, 3,
                                        1, *stream_);
  stream_->synchronize();

  // Test switching between modes
  renderer.setRenderMode(RenderMode::kImage);
  EXPECT_TRUE(renderer.render());

  renderer.setRenderMode(RenderMode::kPointCloud);
  EXPECT_TRUE(renderer.render());

  renderer.setRenderMode(RenderMode::kMesh);
  EXPECT_TRUE(renderer.render());

  // Switch back to first mode
  renderer.setRenderMode(RenderMode::kImage);
  EXPECT_TRUE(renderer.render());

  // Cleanup
  cudaFree(d_depth);
  cudaFree(d_points);
  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_triangles);
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(E2EPipelineTest, UpdateBeforeVisualizerInit) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));

  // Try to update without initializing the visualizer - should fail gracefully
  DepthImage depth_image(10, 10, MemoryType::kDevice);

  // updateDepth should return false when image visualizer not initialized
  EXPECT_FALSE(renderer.updateDepth(depth_image, *stream_));
}

TEST_F(E2EPipelineTest, InitVisualizerBeforeRendererInit) {
  NvbloxRenderer renderer;

  // initVisualizer should fail before renderer is initialized
  EXPECT_FALSE(renderer.initVisualizer(RenderMode::kImage));
  EXPECT_FALSE(renderer.initVisualizer(RenderMode::kPointCloud));
  EXPECT_FALSE(renderer.initVisualizer(RenderMode::kMesh));
}

TEST_F(E2EPipelineTest, RenderModeWithoutVisualizer) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initHeadless(kRenderWidth, kRenderHeight));

  // Initialize only image visualizer
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));

  // Try to render with point cloud mode (not initialized) - should log warning
  renderer.setRenderMode(RenderMode::kPointCloud);
  EXPECT_TRUE(renderer.render());  // Still returns true but renders nothing

  // Try mesh mode
  renderer.setRenderMode(RenderMode::kMesh);
  EXPECT_TRUE(renderer.render());

  // RGBD mode should work
  renderer.setRenderMode(RenderMode::kImage);
  EXPECT_TRUE(renderer.render());
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
