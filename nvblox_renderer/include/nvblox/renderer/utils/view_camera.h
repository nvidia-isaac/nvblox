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

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "nvblox/renderer/utils/renderer_constants.h"

namespace nvblox {
namespace renderer {

/// 3D view camera with arcball rotation (quaternion-based, no gimbal lock).
/// Provides view and projection matrices for rendering 3D content.
class ViewCamera {
 public:
  ViewCamera();

  /// Set camera target (look-at point).
  void setTarget(float x, float y, float z);
  /// @copydoc setTarget(float,float,float)
  void setTarget(const Eigen::Vector3f& target);

  /// Set camera distance from target.
  void setDistance(float distance);

  /// Set camera orbit angles (for initial setup, converts to quaternion).
  /// @param azimuth Horizontal angle in radians.
  /// @param elevation Vertical angle in radians.
  void setOrbitAngles(float azimuth, float elevation);

  /// Rotate camera using arcball rotation (quaternion-based, no gimbal lock).
  /// @param delta_x Horizontal rotation delta (in radians).
  /// @param delta_y Vertical rotation delta (in radians).
  void rotate(float delta_x, float delta_y);

  /// Legacy orbit function (maps to rotate).
  void orbit(float delta_azimuth, float delta_elevation);

  /// Zoom camera (change distance).
  void zoom(float delta);

  /// Pan camera (move target in screen space).
  void pan(float delta_x, float delta_y);

  /// Set rotation directly (quaternion-based, no gimbal lock).
  void setRotation(const Eigen::Quaternionf& rotation);

  /// Reset camera to default orientation.
  void reset();

  /// Set perspective projection parameters.
  /// @param fov_y Field of view in radians (must be positive).
  /// @param aspect Aspect ratio (must be positive).
  /// @param near Near clip plane distance (must be positive).
  /// @param far Far clip plane distance (must be > near).
  void setPerspective(float fov_y, float aspect, float near, float far);

  /// Set aspect ratio.
  /// @param aspect Aspect ratio (must be positive).
  void setAspect(float aspect);

  /// Get view matrix.
  const Eigen::Matrix4f& viewMatrix() const { return view_matrix_; }

  /// Get projection matrix.
  const Eigen::Matrix4f& projMatrix() const { return proj_matrix_; }

  /// Get view-projection matrix.
  const Eigen::Matrix4f& viewProjMatrix() const { return view_proj_matrix_; }

  /// Get view-projection matrix as float pointer (for Vulkan push constants).
  /// Eigen stores matrices in column-major order, matching Vulkan/OpenGL.
  const float* viewProjMatrixPtr() const { return view_proj_matrix_.data(); }

  /// Get camera position.
  const Eigen::Vector3f& position() const { return position_; }

  /// Get distance from target.
  float distance() const { return distance_; }
  /// Get the look-at point.
  const Eigen::Vector3f& target() const { return target_; }
  /// Get the orientation quaternion.
  const Eigen::Quaternionf& rotation() const { return rotation_; }

 private:
  void updateMatrices();

  /// Build a right-handed look-at view matrix (OpenGL convention).
  static Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye,
                                const Eigen::Vector3f& center,
                                const Eigen::Vector3f& up);

  /// Build a perspective projection matrix with depth mapped to [0, 1]
  /// (Vulkan convention).
  static Eigen::Matrix4f perspectiveVk(float fov_y, float aspect, float near,
                                       float far);

  Eigen::Quaternionf rotation_ = Eigen::Quaternionf::Identity();

  float distance_ = kDefaultCameraDistanceM;

  Eigen::Vector3f target_ = Eigen::Vector3f::Zero();

  Eigen::Vector3f position_ =
      Eigen::Vector3f(0.0f, 0.0f, kDefaultCameraDistanceM);

  // Projection parameters
  float fov_y_ = 60.0f * static_cast<float>(M_PI) / 180.0f;  // 60 degrees
  float aspect_ = 16.0f / 9.0f;
  float near_ = 0.01f;
  float far_ = 100.0f;

  Eigen::Matrix4f view_matrix_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f proj_matrix_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f view_proj_matrix_ = Eigen::Matrix4f::Identity();

  // Constants
  static constexpr float kMinDistance = 0.1f;
  static constexpr float kZoomSensitivity = 0.1f;
};

}  // namespace renderer
}  // namespace nvblox
