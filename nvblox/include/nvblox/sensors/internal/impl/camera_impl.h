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

#include <cmath>
#include <iostream>

#include "nvblox/core/types.h"

namespace nvblox {

Camera::Camera(float fu, float fv, float cu, float cv, int width, int height)
    : fu_(fu), fv_(fv), cu_(cu), cv_(cv), width_(width), height_(height) {}

Camera::Camera(float fu, float fv, float cu, float cv, int width, int height,
               float k1, float k2, float k3, float p1, float p2)
    : fu_(fu),
      fv_(fv),
      cu_(cu),
      cv_(cv),
      width_(width),
      height_(height),
      distortion_params_(
          BrownConradyDistortionParams{RadialDistortionParams{k1, k2, k3},
                                       TangentialDistortionParams{p1, p2}}),
      has_distortion_(true) {}

Camera::Camera(float fu, float fv, int width, int height)
    : Camera(fu, fv, width / 2.0, height / 2.0, width, height) {}

bool Camera::project(const Vector3f& p_C, Vector2f* u_C, float min_depth,
                     bool check_viewport) const {
  // Projection to normalized coordinates
  if (!projectToNormalizedCoordinates(p_C, u_C, min_depth)) {
    return false;
  }

  // Apply distortion if present
  if (has_distortion_) {
    *u_C = applyDistortion(*u_C, distortion_params_);
  }

  // Apply intrinsics
  u_C->x() = u_C->x() * fu_ + cu_;
  u_C->y() = u_C->y() * fv_ + cv_;

  if (check_viewport && (u_C->x() > width_ || u_C->y() > height_ ||
                         u_C->x() < 0 || u_C->y() < 0)) {
    return false;
  }
  return true;
}

bool Camera::projectToNormalizedCoordinates(const Vector3f& p_C, Vector2f* u_C,
                                            const float min_depth) {
  if (p_C[2] >= min_depth) {
    u_C->x() = p_C[0] / p_C[2];
    u_C->y() = p_C[1] / p_C[2];
    return true;
  } else {
    return false;
  }
}

float Camera::getDepth(const Vector3f& p_C) const { return p_C.z(); }

Vector3f Camera::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                    const float depth) const {
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f Camera::unprojectFromPixelIndices(const Index2D& u_C,
                                           const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

Vector3f Camera::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  // NOTE(alexmillane): We allow u_C values up to the outer edges of pixels,
  // such that:
  // 0.0f < u_C[0] <= width
  // 0.0f < u_C[1] <= height

  // Convert to normalized coordinates
  Vector2f u_norm((u_C[0] - cu_) / fu_,  // NOLINT
                  (u_C[1] - cv_) / fv_);

  // Remove distortion if present
  if (has_distortion_) {
    u_norm = removeDistortion(u_norm, distortion_params_);
  }

  return Vector3f(u_norm[0], u_norm[1], 1.0f);
}

Vector3f Camera::vectorFromPixelIndices(const Index2D& u_C) const {
  // NOTE(alexmillane): The +0.5 here takes us from image plane indices, which
  // are equal to the coordinates of the lower pixel corner, to the pixel
  // center.
  return vectorFromImagePlaneCoordinates(u_C.cast<float>() +
                                         Vector2f(0.5, 0.5));
}

/// Define how this sensor interpolates on a depth image.
bool Camera::interpolateDepthImage(const DepthImageConstView depth_image,
                                   const Vector2f& u_px, const Vector3f&,
                                   const float, float* value_interpolated_ptr,
                                   Index2D* u_px_closest_ptr) {
  return interpolation::interpolate2DClosest<
      float, interpolation::checkers::PixelNotNan<float>>(
      depth_image, u_px, value_interpolated_ptr, u_px_closest_ptr);
}

Camera Camera::fromIntrinsicsMatrix(const Matrix3f& mat, int width,
                                    int height) {
  const float fu = mat(0, 0);
  const float fv = mat(1, 1);
  const float cu = mat(0, 2);
  const float cv = mat(1, 2);
  return Camera(fu, fv, cu, cv, width, height);
}

Camera Camera::fromIntrinsicsMatrixWithDistortion(const Matrix3f& mat,
                                                  int width, int height,
                                                  float k1, float k2, float k3,
                                                  float p1, float p2) {
  const float fu = mat(0, 0);
  const float fv = mat(1, 1);
  const float cu = mat(0, 2);
  const float cv = mat(1, 2);
  return Camera(fu, fv, cu, cv, width, height, k1, k2, k3, p1, p2);
}

bool operator==(const Camera& lhs, const Camera& rhs) {
  bool same_intrinsics = true;
  same_intrinsics &= std::abs(lhs.fu() - rhs.fu()) <= 0.1;
  same_intrinsics &= std::abs(lhs.fv() - rhs.fv()) <= 0.1;
  same_intrinsics &= std::abs(lhs.cu() - rhs.cu()) <= 0.1;
  same_intrinsics &= std::abs(lhs.cv() - rhs.cv()) <= 0.1;
  same_intrinsics &= lhs.width() == rhs.width();
  same_intrinsics &= lhs.height() == rhs.height();

  // Compare distortion parameters
  same_intrinsics &= lhs.distortion_params() == rhs.distortion_params();

  return same_intrinsics;
}

}  // namespace nvblox
