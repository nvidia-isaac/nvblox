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

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/renderer.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Test Fixture for Window Mode Tests
// ==============================================================================

class WindowRendererTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Check if CUDA is available
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);

    // Check if a display is available by trying to init GLFW
    // This will fail in headless CI environments
    display_available_ = (glfwInit() == GLFW_TRUE);
    if (display_available_) {
      glfwTerminate();  // Clean up, actual window creation will re-init
    }
  }

  void SetUp() override {
    if (!cuda_available_) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    if (!display_available_) {
      GTEST_SKIP() << "No display available (headless environment)";
    }
  }

  static bool cuda_available_;
  static bool display_available_;
};

bool WindowRendererTest::cuda_available_ = false;
bool WindowRendererTest::display_available_ = false;

// ==============================================================================
// Window Mode Initialization Tests
// ==============================================================================

TEST_F(WindowRendererTest, InitWithWindow) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Test Window"));
  EXPECT_TRUE(renderer.isInitialized());
  EXPECT_FALSE(renderer.isHeadless());

  // Context should be valid
  EXPECT_NE(renderer.context(), nullptr);
  EXPECT_NE(renderer.context()->device(), VK_NULL_HANDLE);
  EXPECT_TRUE(renderer.context()->hasRenderTarget());
}

TEST_F(WindowRendererTest, InitWithConfig) {
  NvbloxRenderer renderer;

  RendererConfig config;
  config.width = 1024;
  config.height = 768;
  config.title = "Config Test Window";
  config.headless = false;  // Window mode

  ASSERT_TRUE(renderer.init(config));
  EXPECT_TRUE(renderer.isInitialized());
  EXPECT_FALSE(renderer.isHeadless());
}

TEST_F(WindowRendererTest, DoubleInitFails) {
  NvbloxRenderer renderer;

  ASSERT_TRUE(renderer.initWithWindow(800, 600, "First Init"));
  EXPECT_TRUE(renderer.isInitialized());

  // Second init should fail
  EXPECT_FALSE(renderer.initWithWindow(800, 600, "Second Init"));

  // Original state should be preserved
  EXPECT_TRUE(renderer.isInitialized());
}

TEST_F(WindowRendererTest, DestroyAndReinit) {
  NvbloxRenderer renderer;

  // First init
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "First Window"));
  EXPECT_TRUE(renderer.isInitialized());

  // Destroy
  renderer.destroy();
  EXPECT_FALSE(renderer.isInitialized());

  // Reinit should work
  ASSERT_TRUE(renderer.initWithWindow(640, 480, "Second Window"));
  EXPECT_TRUE(renderer.isInitialized());
}

// ==============================================================================
// Window Rendering Tests
// ==============================================================================

TEST_F(WindowRendererTest, RenderEmptyFrame) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Render Test"));

  // Should be able to render without any visualizers
  EXPECT_TRUE(renderer.render());
}

TEST_F(WindowRendererTest, RenderWithPointCloudVisualizer) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Point Cloud Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  renderer.setRenderMode(RenderMode::kPointCloud);

  // Render multiple frames
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(renderer.render()) << "Failed on frame " << i;
  }
}

TEST_F(WindowRendererTest, RenderWithMeshVisualizer) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Mesh Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kMesh));

  renderer.setRenderMode(RenderMode::kMesh);

  // Render multiple frames
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(renderer.render()) << "Failed on frame " << i;
  }
}

TEST_F(WindowRendererTest, RenderWithImageVisualizer) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Image Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));

  renderer.setRenderMode(RenderMode::kImage);

  // Render multiple frames
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(renderer.render()) << "Failed on frame " << i;
  }
}

// ==============================================================================
// Window Event Tests
// ==============================================================================

TEST_F(WindowRendererTest, ShouldCloseInitiallyFalse) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Close Test"));

  // Window should not request close immediately
  EXPECT_FALSE(renderer.shouldClose());
}

TEST_F(WindowRendererTest, PollEventsDoesNotCrash) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Poll Test"));

  // Poll events multiple times should not crash
  for (int i = 0; i < 10; ++i) {
    renderer.pollEvents();
  }
}

// ==============================================================================
// Camera Controls Tests
// ==============================================================================

TEST_F(WindowRendererTest, CameraControlsEnabled) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Camera Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  // Camera controls should be enabled by default
  EXPECT_TRUE(renderer.cameraControlsEnabled());

  // View camera should be initialized
  EXPECT_NE(renderer.viewCamera(), nullptr);
}

TEST_F(WindowRendererTest, DisableCameraControls) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Camera Disable Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));

  renderer.setCameraControlsEnabled(false);
  EXPECT_FALSE(renderer.cameraControlsEnabled());

  renderer.setCameraControlsEnabled(true);
  EXPECT_TRUE(renderer.cameraControlsEnabled());
}

// ==============================================================================
// Render Mode Switching Tests (also covers render-mode-tests todo)
// ==============================================================================

TEST_F(WindowRendererTest, SwitchRenderModes) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Mode Switch Test"));

  // Initialize all visualizers
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kImage));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kMesh));

  // Test switching between all modes
  const RenderMode modes[] = {
      RenderMode::kImage, RenderMode::kPointCloud, RenderMode::kMesh,
      RenderMode::kImage  // Switch back
  };

  for (RenderMode mode : modes) {
    renderer.setRenderMode(mode);
    EXPECT_EQ(renderer.renderMode(), mode);
    EXPECT_TRUE(renderer.render())
        << "Failed to render in mode " << static_cast<int>(mode);
  }
}

TEST_F(WindowRendererTest, SwitchModesWithoutVisualizer) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "Mode Switch No Viz Test"));

  // Don't initialize any visualizers
  // Switching modes should still work (just won't render anything)
  renderer.setRenderMode(RenderMode::kPointCloud);
  EXPECT_EQ(renderer.renderMode(), RenderMode::kPointCloud);
  EXPECT_TRUE(renderer.render());  // Should succeed (no-op render)

  renderer.setRenderMode(RenderMode::kMesh);
  EXPECT_EQ(renderer.renderMode(), RenderMode::kMesh);
  EXPECT_TRUE(renderer.render());
}

// ==============================================================================
// Integration Test with CUDA Data
// ==============================================================================

TEST_F(WindowRendererTest, RenderPointCloudFromCuda) {
  NvbloxRenderer renderer;
  ASSERT_TRUE(renderer.initWithWindow(800, 600, "CUDA Point Cloud Test"));
  ASSERT_TRUE(renderer.initVisualizer(RenderMode::kPointCloud));
  renderer.setRenderMode(RenderMode::kPointCloud);

  auto stream = std::make_shared<CudaStreamOwning>();

  // Create test point cloud data on GPU
  const size_t num_points = 100;
  PointCloudVisualizer::Point* d_points = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_points, num_points * sizeof(PointCloudVisualizer::Point)),
      cudaSuccess);

  // Initialize points
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
                            cudaMemcpyHostToDevice, *stream),
            cudaSuccess);

  // Update visualizer via accessor
  renderer.pointCloudVisualizer()->updatePoints(d_points, num_points, *stream);

  // Synchronize before render
  stream->synchronize();

  // Render
  EXPECT_TRUE(renderer.render());

  cudaFree(d_points);
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
