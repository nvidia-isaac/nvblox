/*
Copyright 2025 NVIDIA CORPORATION

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

// Real implementation of FuserVisualizer used when BUILD_RENDERER=ON.
// Wraps NvbloxRenderer and manages its own CudaStream for async data copies.

#include "nvblox/fuser/fuser_visualizer.h"

#include <glog/logging.h>

#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/projective_texturing/camera_view.h"
#include "nvblox/projective_texturing/projective_texture_mapper.h"
#include "nvblox/renderer/renderer.h"

namespace nvblox {

struct FuserVisualizer::Impl {
  renderer::NvbloxRenderer renderer;
  std::shared_ptr<CudaStreamOwning> stream;  // shared with the Fuser's mapper
  bool is_initialized = false;

  // Runtime options:
  bool texture_mapping_enabled = true;
  bool follow_orientation_enabled = true;
  bool paused = false;

  // Projective texture mapper.
  ProjectiveTextureMapper texture_mapper;

  // Storing the texture mapped mesh
  ColorMesh flat_mesh;
};

FuserVisualizer::FuserVisualizer() : impl_(std::make_unique<Impl>()) {}

FuserVisualizer::~FuserVisualizer() {
  if (impl_ && impl_->is_initialized) {
    impl_->renderer.destroy();
  }
}

constexpr int kDefaultWindowWidth = 1280;
constexpr int kDefaultWindowHeight = 720;

bool FuserVisualizer::init(const std::string& title,
                           std::shared_ptr<CudaStreamOwning> stream) {
  impl_->stream = std::move(stream);
  if (!impl_->renderer.initWithWindow(kDefaultWindowWidth, kDefaultWindowHeight,
                                      title)) {
    LOG(ERROR) << "FuserVisualizer: Failed to open visualization window.";
    return false;
  }

  // Initialize mesh and point-cloud visualizers
  if (!impl_->renderer.initVisualizer(renderer::RenderMode::kMesh)) {
    LOG(ERROR) << "FuserVisualizer: Failed to initialize mesh visualizer.";
    impl_->renderer.destroy();
    return false;
  }
  if (!impl_->renderer.initVisualizer(renderer::RenderMode::kPointCloud)) {
    LOG(ERROR)
        << "FuserVisualizer: Failed to initialize point-cloud visualizer.";
    impl_->renderer.destroy();
    return false;
  }
  impl_->renderer.setRenderMode(renderer::RenderMode::kMesh);
  impl_->renderer.setCameraControlsEnabled(true);
  impl_->renderer.setClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  // Setup initial camera view
  if (!impl_->renderer.viewCamera()) {
    LOG(ERROR) << "FuserVisualizer: View camera not initialized.";
    impl_->renderer.destroy();
    return false;
  }
  // Initial view pose expressed in the global frame, looking at the origin from
  // a preset distance. This will be overriden if follow-camera mode is
  // activated.
  impl_->renderer.viewCamera()->setTarget(0.0f, 0.0f, 2.0f);
  impl_->renderer.viewCamera()->setDistance(3.0f);
  impl_->renderer.viewCamera()->setOrbitAngles(0.0f, -30.0f);
  impl_->renderer.viewCamera()->setAspect(
      static_cast<float>(kDefaultWindowWidth) /
      static_cast<float>(kDefaultWindowHeight));

  // Setup key callbacks for runtime options
  impl_->renderer.setKeyCallback([&](int key, int action, int /*mods*/) {
    if (action == GLFW_PRESS && key == GLFW_KEY_T) {
      impl_->texture_mapping_enabled = !impl_->texture_mapping_enabled;
      LOG(INFO) << "Texture mapping: "
                << (impl_->texture_mapping_enabled ? "ON" : "OFF");
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_F) {
      impl_->follow_orientation_enabled = !impl_->follow_orientation_enabled;
      LOG(INFO) << "Follow orientation: "
                << (impl_->follow_orientation_enabled ? "ON" : "OFF");
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_W) {
      if (impl_->renderer.meshVisualizer()) {
        impl_->renderer.meshVisualizer()->toggleWireframe();
        LOG(INFO) << "Wireframe: "
                  << (impl_->renderer.meshVisualizer()->wireframe() ? "ON"
                                                                    : "OFF");
      }
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
      impl_->paused = !impl_->paused;
      LOG(INFO) << "Reconstruction: " << (impl_->paused ? "PAUSED" : "RESUMED");
    }
  });
  std::cout << "--------------------------------" << std::endl;
  std::cout << "NVBLOX VISUALIZER CONTROLS:" << std::endl;
  std::cout << "  ESC - Quit" << std::endl;
  std::cout << "  T   - Toggle projective texture mapping" << std::endl;
  std::cout << "  W   - Toggle wireframe" << std::endl;
  std::cout << "  F   - Toggle follow orientation" << std::endl;
  std::cout << "  SPC - Pause/resume reconstruction" << std::endl;
  std::cout << "  Left mouse - Orbit camera" << std::endl;
  std::cout << "  Right mouse - Pan camera" << std::endl;
  std::cout << "  Scroll - Zoom" << std::endl;
  std::cout << "--------------------------------" << std::endl;

  impl_->is_initialized = true;
  return true;
}

void FuserVisualizer::updateMesh(const ColorMesh& mesh, const Camera& color_cam,
                                 const Transform& T_C_L,
                                 const ColorImage& color_frame,
                                 const DepthImage& depth_frame) {
  if (!impl_->is_initialized) {
    return;
  }
  if (mesh.vertices.empty()) {
    return;
  }

  if (impl_->texture_mapping_enabled && color_frame.numel() > 0) {
    // Copy mesh to the working buffer
    impl_->flat_mesh.vertices.copyFromAsync(mesh.vertices, *impl_->stream);
    impl_->flat_mesh.vertex_appearances.copyFromAsync(mesh.vertex_appearances,
                                                      *impl_->stream);
    impl_->flat_mesh.triangles.copyFromAsync(mesh.triangles, *impl_->stream);
    impl_->flat_mesh.vertex_normals.clearNoDeallocate();
    impl_->flat_mesh.vertex_uvs.clearNoDeallocate();

    // Build texture atlas from the current image
    CameraView view(color_cam, T_C_L, color_frame, depth_frame);
    impl_->texture_mapper.buildAtlasAsync({view}, *impl_->stream);
    impl_->texture_mapper.mapMesh(&impl_->flat_mesh, *impl_->stream);

    // Update the rendered mesh and texture
    impl_->renderer.updateMesh(impl_->flat_mesh, *impl_->stream);
    impl_->renderer.updateMeshTexture(impl_->texture_mapper.atlasImage(),
                                      *impl_->stream);
  } else {
    impl_->renderer.updateMesh(mesh, *impl_->stream);
  }

  if (impl_->follow_orientation_enabled && impl_->renderer.viewCamera()) {
    // Get layer->camera transform
    const Transform T_L_C = T_C_L.inverse();

    // Sensor is OpenCV (+Z forward, +Y down); ViewCamera is OpenGL (-Z forward,
    // +Y up). A 180° rotation around X flips both Y and Z to match.
    static const Eigen::Quaternionf kFlipYZ(
        Eigen::AngleAxisf(static_cast<float>(M_PI), Eigen::Vector3f::UnitX()));
    const Eigen::Quaternionf target_rot =
        (Eigen::Quaternionf(T_L_C.rotation()) * kFlipYZ).normalized();

    // kFollowAlpha determines the smoothing factor for camera follow behavior:
    // 0.0f means no movement (no following), 1.0f means instant snapping to the
    // target. Intermediate values result in gradual interpolation for smoother
    // camera tracking.
    constexpr float kFollowAlpha = 0.1f;

    // Slerp rotation for smooth tracking.
    const Eigen::Quaternionf current_rot =
        impl_->renderer.viewCamera()->rotation();
    impl_->renderer.viewCamera()->setRotation(
        current_rot.slerp(kFollowAlpha, target_rot));

    // Lerp target position toward camera translation.
    const Eigen::Vector3f target_pos = T_L_C.translation();
    const Eigen::Vector3f current_target =
        impl_->renderer.viewCamera()->target();
    impl_->renderer.viewCamera()->setTarget(
        current_target + (target_pos - current_target) * kFollowAlpha);
  }
}

bool FuserVisualizer::renderAndPoll() {
  if (!impl_->is_initialized) return true;
  impl_->stream->synchronize();
  if (!impl_->renderer.render()) {
    LOG(ERROR) << "FuserVisualizer: render() failed.";
  }
  impl_->renderer.pollEvents();
  return !impl_->renderer.shouldClose();
}

bool FuserVisualizer::isPaused() const { return impl_->paused; }

bool FuserVisualizer::isAvailable() const { return true; }

}  // namespace nvblox
