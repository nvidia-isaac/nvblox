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
#pragma once

#include <memory>
#include <string>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/render_targets/vk_window.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/image_visualizer.h"
#include "nvblox/renderer/visualizers/mesh_visualizer.h"
#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace renderer {

/// Render mode for the viewer.
/// Determines which visualizer is used during render() and initVisualizer().
enum class RenderMode {
  kImage,       ///< Render depth and/or color images as 2D quads.
  kPointCloud,  ///< Render 3D colored point cloud.
  kMesh         ///< Render 3D colored mesh.
};

/// Configuration for renderer initialization.
/// Pass to NvbloxRenderer::init() to configure the renderer.
struct RendererConfig {
  uint32_t width = 1920;   ///< Render target width in pixels.
  uint32_t height = 1080;  ///< Render target height in pixels.
  std::string title =
      "nvblox renderer";  ///< Window title (ignored in headless mode).
  bool headless = false;  ///< If true, create headless renderer (no window).
};

/// Main Vulkan renderer class for visualizing nvblox data.
///
/// Manages Vulkan context, render targets, and visualizers. Supports both
/// windowed and headless rendering modes.
///
/// ## Lifecycle
/// 1. Construct, then call init() / initWithWindow() / initHeadless()
/// 2. Call initVisualizer(RenderMode) for each visualizer needed
/// 3. Render loop: update* -> render() -> pollEvents()
/// 4. destroy() or let destructor clean up
///
/// ## Error Handling
/// Init and render methods return false on failure and log via glog.
///
/// ## Thread Safety
/// Not thread-safe. All calls must be from the same thread.
///
/// ## Synchronization
/// Caller must call stream.synchronize() after CUDA updates and before
/// render(). CPU sync is used for simplicity and portability; GPU-GPU sync
/// via timeline semaphores is possible by modifying VkContext::endFrame().
///
class NvbloxRenderer {
 public:
  NvbloxRenderer() = default;
  ~NvbloxRenderer();

  // Non-copyable
  NvbloxRenderer(const NvbloxRenderer&) = delete;
  NvbloxRenderer& operator=(const NvbloxRenderer&) = delete;

  /// Initialize the renderer with a configuration.
  /// @param config Renderer configuration (window or headless mode).
  /// @return True if initialization succeeded.
  bool init(const RendererConfig& config);

  /// Initialize the renderer with a window.
  /// @param width Window width.
  /// @param height Window height.
  /// @param title Window title.
  /// @return True if initialization succeeded.
  bool initWithWindow(uint32_t width, uint32_t height,
                      const std::string& title);

  /// Initialize headless renderer (no window).
  /// Useful for offscreen rendering, testing, or server-side use.
  /// @param width Render target width.
  /// @param height Render target height.
  /// @return True if initialization succeeded.
  bool initHeadless(uint32_t width, uint32_t height);

  /// Destroy renderer resources.
  void destroy();

  /// Initialize a visualizer for the given render mode.
  /// @param mode Which visualizer to initialize.
  /// @return True if initialization succeeded.
  bool initVisualizer(RenderMode mode);

  // ============================================================================
  // Data update methods (nvblox types)
  // ============================================================================
  // For raw pointer access, use the individual visualizers directly via
  // imageVisualizer(), pointCloudVisualizer(), meshVisualizer() accessors.

  /// Update depth image for RGBD visualization.
  /// @param depth_image Device-resident depth image.
  /// @param stream CUDA stream.
  /// @return True if update was initiated, false if visualizer not initialized.
  /// @note Data is copied asynchronously. Call stream.synchronize()
  ///       before render() to ensure data transfer is complete.
  bool updateDepth(const DepthImage& depth_image, const CudaStream& stream);

  /// Update color image for RGBD visualization.
  /// @param color_image Device-resident color image.
  /// @param stream CUDA stream.
  /// @return True if update was initiated, false if visualizer not initialized.
  /// @note Data is copied asynchronously. Call stream.synchronize()
  ///       before render() to ensure data transfer is complete.
  bool updateColor(const ColorImage& color_image, const CudaStream& stream);

  /// Update point cloud from RGBD images.
  /// Converts depth + color images into a colored point cloud on the GPU
  /// using the depthToColoredPointCloud kernel.
  /// @param depth_image Device-resident depth image.
  /// @param color_image Device-resident color image.
  /// @param depth_cam Camera intrinsics for depth sensor.
  /// @param color_cam Camera intrinsics for color sensor.
  /// @param stream CUDA stream.
  /// @return True if update was initiated, false if visualizer not initialized.
  /// @note Internally synchronizes the stream to read back the point count.
  bool updatePointCloud(const DepthImage& depth_image,
                        const ColorImage& color_image, const Camera& depth_cam,
                        const Camera& color_cam, const CudaStream& stream);

  /// Update mesh from nvblox ColorMesh.
  /// @param mesh Device-resident colored mesh with vertices, colors, and
  ///        triangles. The triangles vector contains flat vertex indices where
  ///        every 3 consecutive indices form one triangle. Triangle count is
  ///        mesh.triangles.size() / 3.
  /// @param stream CUDA stream.
  /// @return True if update was initiated, false if visualizer not initialized.
  /// @note Data is copied asynchronously. Call stream.synchronize()
  ///       before render() to ensure data transfer is complete.
  /// @note If mesh.triangles.size() is not divisible by 3, a warning is logged
  ///       and the remaining indices are ignored.
  bool updateMesh(const ColorMesh& mesh, const CudaStream& stream);

  /// Update the texture atlas for textured mesh rendering.
  /// @param atlas_image Color image containing the texture atlas (device
  /// memory).
  /// @param stream CUDA stream.
  /// @return True if upload succeeded.
  bool updateMeshTexture(const ColorImage& atlas_image,
                         const CudaStream& stream);

  /// Set depth range for point cloud conversion.
  /// Points outside this range are discarded by updatePointCloud().
  /// @param min_depth Minimum valid depth in meters (default: 0.1).
  /// @param max_depth Maximum valid depth in meters (default: 10.0).
  void setDepthRange(float min_depth, float max_depth) {
    min_depth_ = min_depth;
    max_depth_ = max_depth;
  }

  /// Render a frame.
  /// @note Caller must ensure all CUDA operations that update visualizer data
  ///       have completed (via cudaStreamSynchronize) before calling this
  ///       method.
  /// @return True if rendering succeeded.
  bool render();

  /// Poll window events.
  /// @note In headless mode, this is a no-op since there is no window.
  void pollEvents();

  /// Check if window should close.
  /// @note In headless mode, this always returns false since there is no
  ///       window close event. For headless rendering loops, use your own
  ///       termination condition.
  bool shouldClose() const;

  /// Check if renderer is initialized.
  bool isInitialized() const { return initialized_; }

  /// Check if renderer is in headless mode.
  bool isHeadless() const { return headless_; }

  /// Get the Vulkan context.
  VkContext* context() { return ctx_.get(); }
  const VkContext* context() const { return ctx_.get(); }

  /// Get the image visualizer.
  ImageVisualizer* imageVisualizer() { return image_visualizer_.get(); }
  const ImageVisualizer* imageVisualizer() const {
    return image_visualizer_.get();
  }

  /// Get the point cloud visualizer.
  PointCloudVisualizer* pointCloudVisualizer() {
    return point_cloud_visualizer_.get();
  }
  const PointCloudVisualizer* pointCloudVisualizer() const {
    return point_cloud_visualizer_.get();
  }

  /// Get the mesh visualizer.
  MeshVisualizer* meshVisualizer() { return mesh_visualizer_.get(); }
  const MeshVisualizer* meshVisualizer() const {
    return mesh_visualizer_.get();
  }

  /// Get the view camera.
  ViewCamera* viewCamera() { return view_camera_.get(); }
  const ViewCamera* viewCamera() const { return view_camera_.get(); }

  /// Enable/disable camera controls.
  /// @note In headless mode, camera controls have no effect since there is
  ///       no input. Use ViewCamera methods directly to control the camera.
  void setCameraControlsEnabled(bool enabled) {
    camera_controls_enabled_ = enabled;
  }

  /// Check if camera controls are enabled.
  bool cameraControlsEnabled() const { return camera_controls_enabled_; }

  /// Set the active render mode.
  void setRenderMode(RenderMode mode) { render_mode_ = mode; }

  /// Get the active render mode.
  RenderMode renderMode() const { return render_mode_; }

  /// Set the background clear color (RGBA, linear, range [0, 1]).
  void setClearColor(float r, float g, float b, float a = 1.0f) {
    clear_color_ = {r, g, b, a};
  }

  /// Resize the window (or render target in headless mode).
  /// @param width New width.
  /// @param height New height.
  void resizeWindow(uint32_t width, uint32_t height);

  /// Set callback for key events (for app-specific handling).
  /// @note In headless mode, this has no effect since there is no window
  ///       to receive key events.
  void setKeyCallback(VkWindow::KeyCallback callback);

 private:
  void onResize(int width, int height);
  void onMouseButton(int button, int action, int mods);
  void onMouseMove(double x, double y);
  void onScroll(double x_offset, double y_offset);
  void onKey(int key, int action, int mods);
  void setupInputCallbacks();

  /// Handle render target resize after resize or acquire/present failure.
  /// @return True if resize succeeded (or was skipped due to minimized window).
  bool handleRenderTargetResize();

  /// Common initialization after render target is set up.
  bool initCommon();

  /// Ensure view camera is initialized with default settings.
  /// Called by 3D visualizer init methods (point cloud, mesh).
  void ensureViewCameraInitialized();

  bool initialized_ = false;
  bool headless_ = false;
  bool framebuffer_resized_ = false;

  std::unique_ptr<VkWindow> window_;
  std::unique_ptr<VkContext> ctx_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;

  // Command buffers
  std::vector<VkCommandBuffer> command_buffers_;

  // Visualizers
  std::unique_ptr<ImageVisualizer> image_visualizer_;
  std::unique_ptr<PointCloudVisualizer> point_cloud_visualizer_;
  std::unique_ptr<MeshVisualizer> mesh_visualizer_;

  // Camera for 3D visualization
  std::unique_ptr<ViewCamera> view_camera_;

  // Input state for camera controls
  bool camera_controls_enabled_ = true;
  bool left_mouse_pressed_ = false;
  bool right_mouse_pressed_ = false;
  double last_mouse_x_ = 0.0;
  double last_mouse_y_ = 0.0;

  // Active render mode
  RenderMode render_mode_ = RenderMode::kImage;

  // Background clear color (RGBA).
  std::array<float, 4> clear_color_ = kDefaultClearColor;

  // Point cloud conversion buffer (for updatePointCloud with depth+color)
  device_vector<PointCloudVisualizer::Point> pointcloud_buffer_;
  device_vector<int> num_points_device_{1};  // single int for atomic counter
  float min_depth_ = 0.1f;
  float max_depth_ = 10.0f;

  // User key callback
  VkWindow::KeyCallback user_key_callback_;
};

}  // namespace renderer
}  // namespace nvblox
