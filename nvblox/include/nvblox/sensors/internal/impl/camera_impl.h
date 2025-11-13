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
#include <optional>

#include "nvblox/core/types.h"

namespace nvblox {

Camera::Camera(
    float fu, float fv, float cu, float cv, int width, int height,
    std::optional<RadialTangentialDistortionParams> distortion_params)
    : fu_(fu),
      fv_(fv),
      cu_(cu),
      cv_(cv),
      width_(width),
      height_(height),
      distortion_params_(distortion_params) {}

bool Camera::project(const Vector3f& p_C, Vector2f* u_C, float min_depth,
                     bool check_viewport) const {
  // Check if all values are finite (not NaN or +/- infinity)
  if (!allFinite(p_C)) {
    return false;
  }

  // Projection to normalized coordinates
  if (!projectToNormalizedCoordinates(p_C, u_C, min_depth)) {
    return false;
  }

  // Apply distortion if present
  if (distortion_params_.has_value()) {
    *u_C = applyDistortion(*u_C, distortion_params_.value());
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
  NVBLOX_CHECK(min_depth > 0.f, "");
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
  if (distortion_params_.has_value()) {
    u_norm = removeDistortion(u_norm, distortion_params_.value());
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
  return interpolation::interpolate2DClosest<float>(
      depth_image, u_px, value_interpolated_ptr, u_px_closest_ptr);
}

Camera Camera::fromIntrinsicsMatrix(
    const Matrix3f& mat, int width, int height,
    std::optional<RadialTangentialDistortionParams> distortion_params) {
  const float fu = mat(0, 0);
  const float fv = mat(1, 1);
  const float cu = mat(0, 2);
  const float cv = mat(1, 2);
  return Camera(fu, fv, cu, cv, width, height, distortion_params);
}

bool operator==(const Camera& lhs, const Camera& rhs) {
  bool same_intrinsics = true;
  same_intrinsics &= std::abs(lhs.fu() - rhs.fu()) <= 0.1;
  same_intrinsics &= std::abs(lhs.fv() - rhs.fv()) <= 0.1;
  same_intrinsics &= std::abs(lhs.cu() - rhs.cu()) <= 0.1;
  same_intrinsics &= std::abs(lhs.cv() - rhs.cv()) <= 0.1;
  same_intrinsics &= lhs.width() == rhs.width();
  same_intrinsics &= lhs.height() == rhs.height();

  // Check that both models either have or do not have distortion parameters.
  const bool different_distortion_state =
      (lhs.distortion_params().has_value() !=
       rhs.distortion_params().has_value());
  same_intrinsics &= !different_distortion_state;

  // If both models have distortion, we check the distortion parameters as well.
  const bool both_have_distortion = lhs.distortion_params().has_value() &&
                                    rhs.distortion_params().has_value();
  if (both_have_distortion) {
    same_intrinsics &=
        lhs.distortion_params().value() == rhs.distortion_params().value();
  }
  return same_intrinsics;
}

}  // namespace nvblox
