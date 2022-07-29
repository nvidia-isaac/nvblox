/*
Copyright 2022 NVIDIA CORPORATION

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

#include <glog/logging.h>

namespace nvblox {

Camera::Camera(float fu, float fv, float cu, float cv, int width, int height)
    : fu_(fu), fv_(fv), cu_(cu), cv_(cv), width_(width), height_(height) {}
Camera::Camera(float fu, float fv, int width, int height)
    : Camera(fu, fv, width / 2.0, height / 2.0, width, height) {}

bool Camera::project(const Eigen::Vector3f& p_C, Eigen::Vector2f* u_C) const {
  // Point is behind the camera.
  if (p_C.z() <= 0.0f) {
    return false;
  }
  const float inv_z = 1.0f / p_C.z();
  *u_C = inv_z * Eigen::Vector2f(p_C.x(), p_C.y());

  u_C->x() = u_C->x() * fu_ + cu_;
  u_C->y() = u_C->y() * fv_ + cv_;

  if (u_C->x() > width_ || u_C->y() > height_ || u_C->x() < 0 || u_C->y() < 0) {
    return false;
  }
  return true;
}

float Camera::getDepth(const Vector3f& p_C) const {
  return p_C.z();
}

Vector3f Camera::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  // NOTE(alexmillane): We allow u_C values up to the outer edges of pixels,
  // hence the GE and LE checks.
  DCHECK_GE(u_C[0], 0.0f);
  DCHECK_LE(u_C[0], width_);
  DCHECK_GE(u_C[1], 0.0f);
  DCHECK_LE(u_C[1], height_);
  return Vector3f((u_C[0] - cu_) / fu_,  // NOLINT
                  (u_C[1] - cv_) / fv_,  // NOLINT
                  1.0f);
}

Vector3f Camera::vectorFromPixelIndices(const Index2D& u_C) const {
  // NOTE(alexmillane): The +0.5 here takes us from image plane indices, which
  // are equal to the coordinates of the lower pixel corner, to the pixel
  // center.
  return vectorFromImagePlaneCoordinates(u_C.cast<float>() +
                                         Vector2f(0.5, 0.5));
}

Camera Camera::fromIntrinsicsMatrix(const Eigen::Matrix3f& mat, int width,
                                    int height) {
  const float fu = mat(0, 0);
  const float fv = mat(1, 1);
  const float cu = mat(0, 2);
  const float cv = mat(1, 2);
  return Camera(fu, fv, cu, cv, width, height);
}

}  // namespace nvblox
