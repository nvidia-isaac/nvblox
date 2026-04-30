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
#pragma once

#include <memory>
#include <string>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// Thin wrapper around NvbloxRenderer for use inside the Fuser.
///
/// Visualizes a textured mesh.
///
/// Two implementations exist with identical interfaces:
///   - fuser_visualizer.cpp        (compiled when renderer is available)
///   - fuser_visualizer_stub.cpp   (compiled when renderer is not available)
class FuserVisualizer {
 public:
  FuserVisualizer();
  ~FuserVisualizer();

  // Non-copyable
  FuserVisualizer(const FuserVisualizer&) = delete;
  FuserVisualizer& operator=(const FuserVisualizer&) = delete;

  /// Open a visualization window.
  /// @param stream CUDA stream shared with the mapper; used for all GPU ops.
  /// Returns true on success.
  bool init(const std::string& title, std::shared_ptr<CudaStreamOwning> stream);

  /// Update mesh visualization from a pre-built flat ColorMesh.
  /// Applies projective texture mapping when enabled.
  void updateMesh(const ColorMesh& mesh, const Camera& color_cam,
                  const Transform& T_C_L, const ColorImage& color_frame,
                  const DepthImage& depth_frame);

  /// Render one frame and poll window events.
  /// Returns false when the user closes the window.
  bool renderAndPoll();

  /// Returns true if reconstruction is currently paused (toggled with SPACE).
  bool isPaused() const;

  /// Returns true if the nvblox renderer is available.
  bool isAvailable() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvblox
