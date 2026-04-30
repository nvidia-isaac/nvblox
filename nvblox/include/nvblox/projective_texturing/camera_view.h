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

#include "nvblox/core/types.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// A single camera view for projective texture mapping.
/// Bundles camera intrinsics, extrinsics, image views, and atlas layout.
/// Does NOT own image memory -- the underlying buffers must remain valid
/// for the duration of texture mapping operations.
/// @note Depth and color images must be aligned and undistorted.
struct CameraView {
  Camera camera;                    ///< Camera intrinsics
  Transform T_C_W;                  ///< World-to-camera transform
  Vector3f camera_position_W;       ///< Precomputed camera position in world
                                    ///< frame (avoids T_C_W.inverse() on GPU)
  ColorImageConstView color_image;  ///< Color image view (non-owning)
  DepthImageConstView depth_image;  ///< Aligned depth image view (non-owning,
                                    ///< used for occlusion checking)
  Vector2f atlas_uv_offset{0.0f, 0.0f};  ///< Atlas UV offset (set by atlas)
  Vector2f atlas_uv_scale{1.0f, 1.0f};   ///< Atlas UV scale (set by atlas)

  CameraView() = default;

  CameraView(const Camera& cam, const Transform& transform,
             const ColorImageConstView& color, const DepthImageConstView& depth)
      : camera(cam),
        T_C_W(transform),
        camera_position_W(transform.inverse().translation()),
        color_image(color),
        depth_image(depth) {}
};

}  // namespace nvblox
