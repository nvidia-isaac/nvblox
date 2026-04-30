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

#include <algorithm>
#include <cmath>

#include <glog/logging.h>

#include "nvblox/renderer/utils/view_camera.h"

namespace nvblox {
namespace renderer {

ViewCamera::ViewCamera() { updateMatrices(); }

void ViewCamera::setTarget(float x, float y, float z) {
  target_ = Eigen::Vector3f(x, y, z);
  updateMatrices();
}

void ViewCamera::setTarget(const Eigen::Vector3f& target) {
  target_ = target;
  updateMatrices();
}

void ViewCamera::setDistance(float distance) {
  distance_ = std::max(kMinDistance, distance);
  updateMatrices();
}

void ViewCamera::setOrbitAngles(float azimuth, float elevation) {
  Eigen::Quaternionf q_azimuth(
      Eigen::AngleAxisf(azimuth, Eigen::Vector3f::UnitY()));
  Eigen::Quaternionf q_elevation(
      Eigen::AngleAxisf(elevation, Eigen::Vector3f::UnitX()));
  rotation_ = q_azimuth * q_elevation;
  updateMatrices();
}

void ViewCamera::rotate(float delta_x, float delta_y) {
  Eigen::Quaternionf q_yaw(
      Eigen::AngleAxisf(-delta_x, Eigen::Vector3f::UnitY()));
  Eigen::Vector3f right = rotation_ * Eigen::Vector3f::UnitX();
  Eigen::Quaternionf q_pitch(Eigen::AngleAxisf(-delta_y, right));
  rotation_ = (q_pitch * q_yaw * rotation_).normalized();
  updateMatrices();
}

void ViewCamera::orbit(float delta_azimuth, float delta_elevation) {
  rotate(delta_azimuth, delta_elevation);
}

void ViewCamera::zoom(float delta) {
  distance_ =
      std::max(kMinDistance, distance_ - delta * distance_ * kZoomSensitivity);
  updateMatrices();
}

void ViewCamera::pan(float delta_x, float delta_y) {
  Eigen::Vector3f right = rotation_ * Eigen::Vector3f::UnitX();
  Eigen::Vector3f up = rotation_ * Eigen::Vector3f::UnitY();
  float scale = distance_ * kCameraPanSpeedMPerPx;
  target_ += right * delta_x * scale + up * delta_y * scale;
  updateMatrices();
}

void ViewCamera::setRotation(const Eigen::Quaternionf& rotation) {
  rotation_ = rotation.normalized();
  updateMatrices();
}

void ViewCamera::reset() {
  rotation_ = Eigen::Quaternionf::Identity();
  distance_ = kDefaultCameraDistanceM;
  target_ = Eigen::Vector3f::Zero();
  updateMatrices();
}

void ViewCamera::setPerspective(float fov_y, float aspect, float near,
                                float far) {
  if (fov_y <= 0.0f || fov_y >= static_cast<float>(M_PI)) {
    LOG(WARNING) << "setPerspective: invalid fov_y=" << fov_y
                 << " (must be in (0, pi))";
    return;
  }
  if (aspect <= 0.0f) {
    LOG(WARNING) << "setPerspective: invalid aspect=" << aspect
                 << " (must be > 0)";
    return;
  }
  if (near <= 0.0f || far <= near) {
    LOG(WARNING) << "setPerspective: invalid near/far=" << near << "/" << far
                 << " (need 0 < near < far)";
    return;
  }
  fov_y_ = fov_y;
  aspect_ = aspect;
  near_ = near;
  far_ = far;
  updateMatrices();
}

void ViewCamera::setAspect(float aspect) {
  if (aspect <= 0.0f) {
    LOG(WARNING) << "setAspect: invalid aspect=" << aspect << " (must be > 0)";
    return;
  }
  aspect_ = aspect;
  updateMatrices();
}

void ViewCamera::updateMatrices() {
  Eigen::Vector3f forward = rotation_ * Eigen::Vector3f::UnitZ();
  position_ = target_ + forward * distance_;

  Eigen::Vector3f up = rotation_ * Eigen::Vector3f::UnitY();

  view_matrix_ = lookAt(position_, target_, up);

  proj_matrix_ = perspectiveVk(fov_y_, aspect_, near_, far_);
  proj_matrix_(1, 1) *= -1;  // Flip Y for Vulkan

  view_proj_matrix_ = proj_matrix_ * view_matrix_;
}

Eigen::Matrix4f ViewCamera::lookAt(const Eigen::Vector3f& eye,
                                   const Eigen::Vector3f& center,
                                   const Eigen::Vector3f& up) {
  Eigen::Vector3f f = (center - eye).normalized();
  Eigen::Vector3f s = f.cross(up).normalized();
  Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result(0, 0) = s.x();
  result(0, 1) = s.y();
  result(0, 2) = s.z();
  result(1, 0) = u.x();
  result(1, 1) = u.y();
  result(1, 2) = u.z();
  result(2, 0) = -f.x();
  result(2, 1) = -f.y();
  result(2, 2) = -f.z();
  result(0, 3) = -s.dot(eye);
  result(1, 3) = -u.dot(eye);
  result(2, 3) = f.dot(eye);
  return result;
}

Eigen::Matrix4f ViewCamera::perspectiveVk(float fov_y, float aspect, float near,
                                          float far) {
  float tan_half_fov = std::tan(fov_y / 2.0f);

  Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
  result(0, 0) = 1.0f / (aspect * tan_half_fov);
  result(1, 1) = 1.0f / tan_half_fov;
  result(2, 2) = far / (near - far);
  result(2, 3) = -(far * near) / (far - near);
  result(3, 2) = -1.0f;
  return result;
}

}  // namespace renderer
}  // namespace nvblox
