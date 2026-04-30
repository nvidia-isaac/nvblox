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

/// @file push_constants.h
/// @brief C++ structs mirroring GLSL push_constant layouts.
///
/// These structs define the data passed from C++ to shaders via Vulkan
/// push constants. They must exactly match the corresponding GLSL layouts:
///
/// - MeshPushConstants      -> shaders/mesh.vert
/// - PointCloudPushConstants -> shaders/point_cloud.vert
///
/// The static_asserts ensure compile-time verification of size matching.
/// If a shader layout changes, update the corresponding struct here.

namespace nvblox {
namespace renderer {

/// Push constants for mesh visualizer.
/// Matches: shaders/mesh.vert and shaders/mesh.frag
///
/// GLSL layout:
/// @code
/// layout(push_constant) uniform PushConstants {
///     mat4 viewProj;
///     uint hasTexture;  // 0 = vertex color only, 1 = texture bound
/// } pc;
/// @endcode
struct MeshPushConstants {
  float viewProj[16];    ///< View-projection matrix (mat4)
  uint32_t hasTexture;   ///< 0 = vertex color only, 1 = texture atlas bound
  uint32_t reserved[3];  ///< Reserved for future flags (16-byte alignment)
};
static_assert(sizeof(MeshPushConstants) == 80,
              "MeshPushConstants size mismatch with mesh shaders");

/// Push constants for point cloud visualizer.
/// Matches: shaders/point_cloud.vert
///
/// GLSL layout:
/// @code
/// layout(push_constant) uniform PushConstants {
///     mat4 viewProj;
///     float pointSize;
/// } pc;
/// @endcode
///
/// Note: padding is required for 16-byte alignment per Vulkan spec.
struct PointCloudPushConstants {
  float viewProj[16];  ///< View-projection matrix (mat4)
  float pointSize;     ///< Point size in pixels
  float padding[3];    ///< Padding for 16-byte alignment
};
static_assert(
    sizeof(PointCloudPushConstants) == 80,
    "PointCloudPushConstants size mismatch with point_cloud.vert shader");

/// Push constants for image visualizer.
/// Matches: shaders/image_quad.frag
///
/// GLSL layout:
/// @code
/// layout(push_constant) uniform PushConstants {
///     float minDepth;
///     float maxDepth;
///     float colormap;  // 0=grayscale, 1=jet, 2=turbo
///     float layout_;   // 0=side-by-side, 1=color-only, 2=depth-only,
///     3=overlay
/// } pc;
/// @endcode
struct ImagePushConstants {
  float minDepth;       ///< Minimum depth for colormap normalization
  float maxDepth;       ///< Maximum depth for colormap normalization
  float colormap;       ///< Colormap type: 0=grayscale, 1=jet, 2=turbo
  float displayLayout;  ///< Layout type: 0=side-by-side, 1=color, 2=depth,
                        ///< 3=overlay
};
static_assert(sizeof(ImagePushConstants) == 16,
              "ImagePushConstants size mismatch with image_quad.frag shader");

}  // namespace renderer
}  // namespace nvblox
