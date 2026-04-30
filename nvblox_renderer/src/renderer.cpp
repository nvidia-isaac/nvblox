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

#include "nvblox/renderer/renderer.h"

#include <array>

#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/kernels/depth_to_pointcloud.h"
#include "nvblox/renderer/render_targets/vk_headless_target.h"
#include "nvblox/renderer/render_targets/vk_window_target.h"
#include "nvblox/renderer/utils/renderer_constants.h"
#include "nvblox/renderer/utils/view_camera.h"
#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"

namespace nvblox {
namespace renderer {

// Verify type layout assumptions for reinterpret_cast in updateMesh()
static_assert(sizeof(Vector3f) == 3 * sizeof(float),
              "Vector3f must be 3 contiguous floats for mesh vertex data");
static_assert(sizeof(Color) == 3,
              "Color must be 3 bytes for mesh vertex color data");

NvbloxRenderer::~NvbloxRenderer() { destroy(); }

void NvbloxRenderer::destroy() {
  if (ctx_) {
    ctx_->waitIdle();
  }

  image_visualizer_.reset();
  point_cloud_visualizer_.reset();
  mesh_visualizer_.reset();
  view_camera_.reset();

  // Point cloud conversion buffers (device_vector) are cleaned up by
  // their destructors automatically.

  if (ctx_ && ctx_->device()) {
    if (!command_buffers_.empty()) {
      vkFreeCommandBuffers(ctx_->device(), ctx_->commandPool(),
                           static_cast<uint32_t>(command_buffers_.size()),
                           command_buffers_.data());
      command_buffers_.clear();
    }
  }

  // Destroy render target BEFORE surface (Vulkan spec requirement for window
  // targets)
  if (ctx_) {
    ctx_->destroyRenderTarget();
  }

  // Surface only exists in window mode
  if (surface_ != VK_NULL_HANDLE && ctx_ &&
      ctx_->instance() != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(ctx_->instance(), surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }

  ctx_.reset();
  window_.reset();
  initialized_ = false;
  headless_ = false;
}

bool NvbloxRenderer::init(const RendererConfig& config) {
  if (config.headless) {
    return initHeadless(config.width, config.height);
  } else {
    return initWithWindow(config.width, config.height, config.title);
  }
}

bool NvbloxRenderer::initWithWindow(uint32_t width, uint32_t height,
                                    const std::string& title) {
  // Check for double-init
  if (initialized_) {
    LOG(ERROR) << "NvbloxRenderer already initialized. Call destroy() first.";
    return false;
  }

  // Get required Vulkan extensions for window presentation
  auto required_extensions = VkWindow::getRequiredVulkanExtensions();

  // Create window (GLFW uses int for dimensions)
  window_ = std::make_unique<VkWindow>();
  if (!window_->create(static_cast<int>(width), static_cast<int>(height),
                       title)) {
    LOG(ERROR) << "Failed to create window";
    return false;
  }

  // Set resize callback
  window_->setResizeCallback([this](int w, int h) { onResize(w, h); });

  // Set up input callbacks
  setupInputCallbacks();

  // Create Vulkan context with window extensions
  ctx_ = std::make_unique<VkContext>();

#ifdef ENABLE_VALIDATION_LAYERS
  bool enable_validation = true;
#else
  bool enable_validation = false;
#endif

  if (!ctx_->init(title, required_extensions, enable_validation)) {
    LOG(ERROR) << "Failed to initialize Vulkan context";
    return false;
  }

  // Create surface
  if (!window_->createSurface(ctx_->instance(), &surface_)) {
    LOG(ERROR) << "Failed to create Vulkan surface";
    return false;
  }

  // Create device
  if (!ctx_->createDevice()) {
    LOG(ERROR) << "Failed to create Vulkan device";
    return false;
  }

  // Create window render target and set it on the context
  int fb_width, fb_height;
  window_->getFramebufferSize(&fb_width, &fb_height);

  auto window_target = std::make_unique<VkWindowTarget>();
  if (!window_target->create(ctx_->device(), ctx_->physicalDevice(), surface_,
                             ctx_->graphicsQueueFamily(), ctx_->graphicsQueue(),
                             fb_width, fb_height)) {
    LOG(ERROR) << "Failed to create window target";
    return false;
  }

  if (!ctx_->setRenderTarget(std::move(window_target))) {
    LOG(ERROR) << "Failed to set render target";
    return false;
  }

  headless_ = false;
  return initCommon();
}

bool NvbloxRenderer::initHeadless(uint32_t width, uint32_t height) {
  // Check for double-init
  if (initialized_) {
    LOG(ERROR) << "NvbloxRenderer already initialized. Call destroy() first.";
    return false;
  }

  // Create Vulkan context without window extensions
  ctx_ = std::make_unique<VkContext>();

#ifdef ENABLE_VALIDATION_LAYERS
  bool enable_validation = true;
#else
  bool enable_validation = false;
#endif

  // No window extensions needed for headless
  std::vector<const char*> extensions;
  if (!ctx_->init("nvblox_headless", extensions, enable_validation)) {
    LOG(ERROR) << "Failed to initialize Vulkan context for headless rendering";
    return false;
  }

  // Create device
  if (!ctx_->createDevice()) {
    LOG(ERROR) << "Failed to create Vulkan device";
    return false;
  }

  // Create headless render target
  auto headless_target = std::make_unique<VkHeadlessTarget>();
  if (!headless_target->create(ctx_->device(), ctx_->physicalDevice(), width,
                               height)) {
    LOG(ERROR) << "Failed to create headless render target";
    return false;
  }

  if (!ctx_->setRenderTarget(std::move(headless_target))) {
    LOG(ERROR) << "Failed to set headless render target";
    return false;
  }

  headless_ = true;
  return initCommon();
}

bool NvbloxRenderer::initCommon() {
  // Create command buffers
  command_buffers_.resize(ctx_->renderTargetImageCount());
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = ctx_->commandPool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount =
      static_cast<uint32_t>(command_buffers_.size());

  VkResult result = vkAllocateCommandBuffers(ctx_->device(), &alloc_info,
                                             command_buffers_.data());
  if (!checkVkResult(result, "vkAllocateCommandBuffers")) {
    LOG(ERROR) << "Failed to allocate command buffers";
    command_buffers_.clear();
    return false;
  }

  initialized_ = true;
  LOG(INFO) << "NvbloxRenderer initialized"
            << (headless_ ? " (headless mode)" : " (window mode)");
  return true;
}

void NvbloxRenderer::ensureViewCameraInitialized() {
  if (!view_camera_) {
    view_camera_ = std::make_unique<ViewCamera>();
    view_camera_->setDistance(kDefaultCameraDistanceM);
    view_camera_->setTarget(0.0f, 0.0f, 0.0f);
  }
}

bool NvbloxRenderer::initVisualizer(RenderMode mode) {
  if (!initialized_) {
    LOG(ERROR) << "Renderer not initialized";
    return false;
  }

  switch (mode) {
    case RenderMode::kImage: {
      if (image_visualizer_) {
        LOG(WARNING) << "ImageVisualizer already initialized, destroying and "
                        "re-creating";
        image_visualizer_.reset();
      }
      image_visualizer_ = std::make_unique<ImageVisualizer>();
      if (!image_visualizer_->init(ctx_.get())) {
        LOG(ERROR) << "Failed to initialize image visualizer";
        image_visualizer_.reset();
        return false;
      }
      return true;
    }
    case RenderMode::kPointCloud: {
      if (point_cloud_visualizer_) {
        LOG(WARNING) << "PointCloudVisualizer already initialized, destroying "
                        "and re-creating";
        point_cloud_visualizer_.reset();
      }
      ensureViewCameraInitialized();
      point_cloud_visualizer_ = std::make_unique<PointCloudVisualizer>();
      if (!point_cloud_visualizer_->init(ctx_.get())) {
        LOG(ERROR) << "Failed to initialize point cloud visualizer";
        point_cloud_visualizer_.reset();
        return false;
      }
      return true;
    }
    case RenderMode::kMesh: {
      if (mesh_visualizer_) {
        LOG(WARNING) << "MeshVisualizer already initialized, destroying and "
                        "re-creating";
        mesh_visualizer_.reset();
      }
      ensureViewCameraInitialized();
      mesh_visualizer_ = std::make_unique<MeshVisualizer>();
      if (!mesh_visualizer_->init(ctx_.get())) {
        LOG(ERROR) << "Failed to initialize mesh visualizer";
        mesh_visualizer_.reset();
        return false;
      }
      return true;
    }
  }
  LOG(ERROR) << "Unknown RenderMode";
  return false;
}

// ============================================================================
// Data update methods
// ============================================================================

bool NvbloxRenderer::updateDepth(const DepthImage& depth_image,
                                 const CudaStream& stream) {
  if (!image_visualizer_) {
    LOG(WARNING) << "updateDepth called but image visualizer not initialized";
    return false;
  }
  image_visualizer_->updateDepth(depth_image.dataConstPtr(), stream);
  return true;
}

bool NvbloxRenderer::updateColor(const ColorImage& color_image,
                                 const CudaStream& stream) {
  if (!image_visualizer_) {
    LOG(WARNING) << "updateColor called but image visualizer not initialized";
    return false;
  }
  image_visualizer_->updateColor(color_image.dataConstPtr(), stream);
  return true;
}

bool NvbloxRenderer::updatePointCloud(const DepthImage& depth_image,
                                      const ColorImage& color_image,
                                      const Camera& depth_cam,
                                      const Camera& color_cam,
                                      const CudaStream& stream) {
  if (!point_cloud_visualizer_) {
    LOG(WARNING)
        << "updatePointCloud called but point cloud visualizer not initialized";
    return false;
  }

  // Ensure intermediate buffer is allocated
  const size_t max_points =
      static_cast<size_t>(depth_image.width()) * depth_image.height();
  if (max_points > pointcloud_buffer_.capacity()) {
    pointcloud_buffer_.reserveAsync(max_points, stream);
  }

  // Run depth-to-pointcloud conversion on GPU
  if (!depthToColoredPointCloud(
          depth_image.dataConstPtr(),
          reinterpret_cast<const uint8_t*>(color_image.dataConstPtr()),
          depth_cam, color_cam, nullptr, pointcloud_buffer_.data(),
          static_cast<int>(max_points), num_points_device_.data(), min_depth_,
          max_depth_, stream)) {
    LOG(ERROR) << "depthToColoredPointCloud failed";
    return false;
  }

  // Read back the point count (requires stream sync)
  int num_points = 0;
  stream.synchronize();
  checkCudaErrors(cudaMemcpy(&num_points, num_points_device_.data(),
                             sizeof(int), cudaMemcpyDeviceToHost));

  // Update the visualizer with the converted points
  point_cloud_visualizer_->updatePoints(
      pointcloud_buffer_.data(), static_cast<size_t>(num_points), stream);
  return true;
}

bool NvbloxRenderer::updateMesh(const ColorMesh& mesh,
                                const CudaStream& stream) {
  if (!mesh_visualizer_) {
    LOG(WARNING) << "updateMesh called but mesh visualizer not initialized";
    return false;
  }

  // nvblox ColorMesh stores triangles as a flat array of vertex indices,
  // where every 3 consecutive indices form one triangle:
  //   triangles = [v0, v1, v2,  v3, v4, v5,  ...]
  //               |  tri 0  |  |  tri 1  |  ...
  if (mesh.triangles.size() % 3 != 0) {
    LOG(WARNING) << "updateMesh: mesh.triangles.size() = "
                 << mesh.triangles.size()
                 << " is not divisible by 3. Truncating to complete triangles.";
  }

  // Extract raw pointers from nvblox mesh
  // Note: unified_vector::data() returns device pointer for device memory
  const float* positions = reinterpret_cast<const float*>(mesh.vertices.data());
  const uint8_t* colors =
      reinterpret_cast<const uint8_t*>(mesh.vertex_appearances.data());
  const int* triangles = mesh.triangles.data();

  // UVs are optional -- pass nullptr if vertex_uvs is empty (all vertices
  // use vertex color). When populated, each vertex has a 2D UV coordinate.
  const float* uvs = nullptr;
  if (!mesh.vertex_uvs.empty()) {
    uvs = reinterpret_cast<const float*>(mesh.vertex_uvs.data());
  }

  mesh_visualizer_->updateMesh(positions, colors, triangles,
                               mesh.vertices.size(), mesh.triangles.size() / 3,
                               stream, uvs);
  return true;
}

bool NvbloxRenderer::updateMeshTexture(const ColorImage& atlas_image,
                                       const CudaStream& stream) {
  if (!mesh_visualizer_) {
    LOG(WARNING) << "updateMeshTexture called but mesh visualizer not "
                    "initialized. Call initVisualizer(RenderMode::kMesh) "
                    "first.";
    return false;
  }
  return mesh_visualizer_->updateTexture(atlas_image.dataConstPtr(),
                                         atlas_image.width(),
                                         atlas_image.height(), stream);
}

bool NvbloxRenderer::handleRenderTargetResize() {
  if (headless_) {
    // Headless mode doesn't resize automatically
    return true;
  }

  int width, height;
  window_->getFramebufferSize(&width, &height);
  if (width > 0 && height > 0) {
    return ctx_->resizeRenderTarget(width, height);
  }
  // Window is minimized or has zero size - skip resize
  return true;
}

bool NvbloxRenderer::render() {
  if (!initialized_) {
    return false;
  }

  // Validate that the current render mode has an initialized visualizer.
  // This catches common errors where the user forgets to init the right
  // visualizer.
  bool visualizer_available = false;
  switch (render_mode_) {
    case RenderMode::kImage:
      visualizer_available = (image_visualizer_ != nullptr);
      if (!visualizer_available) {
        LOG_FIRST_N(WARNING, 1)
            << "render() called with RenderMode::kImage but "
            << "image visualizer not initialized. Call "
            << "initVisualizer(RenderMode::kImage) first.";
      }
      break;
    case RenderMode::kPointCloud:
      visualizer_available = (point_cloud_visualizer_ != nullptr);
      if (!visualizer_available) {
        LOG_FIRST_N(WARNING, 1)
            << "render() called with RenderMode::kPointCloud but "
            << "point cloud visualizer not initialized. Call "
            << "initVisualizer(RenderMode::kPointCloud) first.";
      }
      break;
    case RenderMode::kMesh:
      visualizer_available = (mesh_visualizer_ != nullptr);
      if (!visualizer_available) {
        LOG_FIRST_N(WARNING, 1) << "render() called with RenderMode::kMesh but "
                                << "mesh visualizer not initialized. Call "
                                << "initVisualizer(RenderMode::kMesh) first.";
      }
      break;
    default:
      LOG(ERROR) << "Unknown RenderMode";
      return false;
  }

  // Handle minimized window (only in window mode)
  if (!headless_ && window_->isMinimized()) {
    return true;
  }

  // Handle resize (only in window mode)
  if (framebuffer_resized_) {
    if (!handleRenderTargetResize()) {
      LOG(WARNING) << "Render target resize deferred after window resize";
    }
    framebuffer_resized_ = false;
  }

  // Begin frame
  uint32_t image_index;
  if (!ctx_->beginFrame(&image_index)) {
    // Need to resize render target (only possible in window mode)
    if (!handleRenderTargetResize()) {
      LOG(ERROR) << "Failed to resize render target";
      return false;
    }
    return true;  // Successfully resized, skip this frame
  }

  // Record command buffer
  // Note on error handling: Command buffer operations use checkVkErrors (fatal)
  // because failures here indicate programmer error or device failure that
  // cannot be recovered from. The command buffer would be in an undefined
  // state and continuing would cause undefined behavior.
  if (image_index >= command_buffers_.size()) {
    LOG(ERROR) << "Invalid image_index " << image_index
               << " (command_buffers_.size() = " << command_buffers_.size()
               << ")";
    return false;
  }
  VkCommandBuffer cmd = command_buffers_[image_index];
  checkVkErrors(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  checkVkErrors(vkBeginCommandBuffer(cmd, &begin_info));

  // Begin render pass
  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = ctx_->renderPass();

  VkFramebuffer fb = ctx_->framebuffer(image_index);
  if (fb == VK_NULL_HANDLE) {
    LOG(ERROR) << "Invalid framebuffer for image index: " << image_index;
    return false;
  }
  render_pass_info.framebuffer = fb;
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = ctx_->renderTargetExtent();

  std::array<VkClearValue, 2> clear_values{};
  clear_values[0].color = {
      {clear_color_[0], clear_color_[1], clear_color_[2], clear_color_[3]}};
  clear_values[1].depthStencil = {kDefaultDepthClear, 0};
  render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

  // Render only the active visualizer based on render mode
  if (render_mode_ == RenderMode::kImage && image_visualizer_) {
    image_visualizer_->render(cmd, nullptr, ctx_->renderTargetExtent().width,
                              ctx_->renderTargetExtent().height);
  } else if (render_mode_ == RenderMode::kPointCloud &&
             point_cloud_visualizer_ && view_camera_) {
    point_cloud_visualizer_->render(cmd, view_camera_.get(),
                                    ctx_->renderTargetExtent().width,
                                    ctx_->renderTargetExtent().height);
  } else if (render_mode_ == RenderMode::kMesh && mesh_visualizer_ &&
             view_camera_) {
    mesh_visualizer_->render(cmd, view_camera_.get(),
                             ctx_->renderTargetExtent().width,
                             ctx_->renderTargetExtent().height);
  }

  vkCmdEndRenderPass(cmd);
  checkVkErrors(vkEndCommandBuffer(cmd));

  // End frame
  if (!ctx_->endFrame(image_index, cmd)) {
    if (!handleRenderTargetResize()) {
      LOG(WARNING) << "Render target resize deferred after endFrame failure";
    }
  }

  return true;
}

void NvbloxRenderer::pollEvents() {
  // No events to poll in headless mode
  if (!headless_ && window_) {
    window_->pollEvents();
  }
}

bool NvbloxRenderer::shouldClose() const {
  // Headless mode never closes (application controls lifecycle)
  if (headless_) {
    return false;
  }
  return window_ ? window_->shouldClose() : true;
}

void NvbloxRenderer::resizeWindow(uint32_t width, uint32_t height) {
  if (headless_) {
    // In headless mode, resize the render target directly
    if (ctx_ && width > 0 && height > 0) {
      ctx_->resizeRenderTarget(width, height);
    }
  } else if (window_) {
    // GLFW uses int for dimensions
    window_->resize(static_cast<int>(width), static_cast<int>(height));
  }
}

void NvbloxRenderer::onResize(int /*width*/, int /*height*/) {
  framebuffer_resized_ = true;

  // Update view camera aspect ratio
  if (view_camera_) {
    int fb_width, fb_height;
    window_->getFramebufferSize(&fb_width, &fb_height);
    if (fb_width > 0 && fb_height > 0) {
      view_camera_->setAspect(static_cast<float>(fb_width) /
                              static_cast<float>(fb_height));
    }
  }
}

void NvbloxRenderer::setupInputCallbacks() {
  // Mouse button callback
  window_->setMouseButtonCallback([this](int button, int action, int mods) {
    onMouseButton(button, action, mods);
  });

  // Mouse move callback
  window_->setMouseMoveCallback(
      [this](double x, double y) { onMouseMove(x, y); });

  // Scroll callback
  window_->setScrollCallback([this](double x_offset, double y_offset) {
    onScroll(x_offset, y_offset);
  });

  // Key callback
  window_->setKeyCallback(
      [this](int key, int action, int mods) { onKey(key, action, mods); });
}

void NvbloxRenderer::setKeyCallback(VkWindow::KeyCallback callback) {
  user_key_callback_ = std::move(callback);
}

void NvbloxRenderer::onMouseButton(int button, int action, int /*mods*/) {
  if (!camera_controls_enabled_ || !view_camera_) {
    return;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    left_mouse_pressed_ = (action == GLFW_PRESS);
    if (left_mouse_pressed_) {
      window_->getCursorPos(&last_mouse_x_, &last_mouse_y_);
    }
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    right_mouse_pressed_ = (action == GLFW_PRESS);
    if (right_mouse_pressed_) {
      window_->getCursorPos(&last_mouse_x_, &last_mouse_y_);
    }
  }
}

void NvbloxRenderer::onMouseMove(double x, double y) {
  if (!camera_controls_enabled_ || !view_camera_) {
    return;
  }

  double dx = x - last_mouse_x_;
  double dy = y - last_mouse_y_;
  last_mouse_x_ = x;
  last_mouse_y_ = y;

  if (left_mouse_pressed_) {
    // Orbit camera - drag to look in that direction
    view_camera_->orbit(-static_cast<float>(dx) * kCameraOrbitSpeedRadPerPx,
                        static_cast<float>(dy) * kCameraOrbitSpeedRadPerPx);
  } else if (right_mouse_pressed_) {
    // Pan camera - drag to move scene in that direction
    float distance = view_camera_->distance();
    view_camera_->pan(
        -static_cast<float>(dx) * kCameraPanSpeedMPerPx * distance,
        static_cast<float>(dy) * kCameraPanSpeedMPerPx * distance);
  }
}

void NvbloxRenderer::onScroll(double /*x_offset*/, double y_offset) {
  if (!camera_controls_enabled_ || !view_camera_) {
    return;
  }

  // Zoom camera
  view_camera_->zoom(static_cast<float>(y_offset));
}

void NvbloxRenderer::onKey(int key, int action, int mods) {
  if (!view_camera_) {
    // Forward to user callback if no view camera
    if (user_key_callback_) {
      user_key_callback_(key, action, mods);
    }
    return;
  }

  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_R:
        // Reset camera (target at typical depth, looking from behind)
        view_camera_->setTarget(0.0f, 0.0f, kCameraResetTargetZM);
        view_camera_->setDistance(kDefaultCameraDistanceM);
        view_camera_->setOrbitAngles(kCameraResetYawRad, kCameraResetPitchRad);
        break;
      default:
        break;
    }
  }

  // Forward to user callback
  if (user_key_callback_) {
    user_key_callback_(key, action, mods);
  }
}

}  // namespace renderer
}  // namespace nvblox
