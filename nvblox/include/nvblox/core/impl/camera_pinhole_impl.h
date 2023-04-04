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

CameraPinhole::CameraPinhole(const Matrix3f& K, const int width,
                             const int height)
    : K_(K), width_(width), height_(height), rectified_(false) {
  P_.setIdentity();
  R_rect_.setIdentity();
}

CameraPinhole::CameraPinhole(const Matrix3x4f& P, const Matrix3f& rect,
                             const int width, const int height)
    : P_(P), R_rect_(rect), width_(width), height_(height), rectified_(true) {
  K_.setIdentity();
}

bool CameraPinhole::project(const Eigen::Vector3f& p_C,
                            Eigen::Vector2f* u_C) const {
  // Point is behind the camera.
  if (p_C.z() <= 0.0f) {
    return false;
  }
  if (rectified_) {
    Vector3f p = P_.block<3, 3>(0, 0) * (R_rect_ * p_C) + P_.block<3, 1>(0, 3);
    *u_C = p.head<2>() / p(2);
    if (u_C->x() > width_ || u_C->y() > height_ || u_C->x() < 0 ||
        u_C->y() < 0) {
      return false;
    }
  } else {
    Vector3f p = K_ * p_C;
    *u_C = p.head<2>() / p(2);
    if (u_C->x() > width_ || u_C->y() > height_ || u_C->x() < 0 ||
        u_C->y() < 0) {
      return false;
    }
  }
  return true;
}

float CameraPinhole::getDepth(const Vector3f& p_C) const { return p_C.z(); }

Vector3f CameraPinhole::unprojectFromImagePlaneCoordinates(
    const Vector2f& u_C, const float depth) const {
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f CameraPinhole::unprojectFromPixelIndices(const Index2D& u_C,
                                                  const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

Vector3f CameraPinhole::vectorFromImagePlaneCoordinates(
    const Vector2f& u_C) const {
  // NOTE(alexmillane): We allow u_C values up to the outer edges of pixels,
  // such that:
  // 0.0f < u_C[0] <= width
  // 0.0f < u_C[1] <= height
  if (rectified_) {
    Matrix3f P_tmp = P_.block<3, 3>(0, 0);
    Vector3f x(u_C[0], u_C[1], 1.0f);
    Vector3f p = P_tmp.inverse() * x;
    p /= p(2);
    return p;
  } else {
    Vector3f x(u_C[0], u_C[1], 1.0f);
    Vector3f p = K_.inverse() * x;
    p /= p(2);
    return p;
  }
}

Vector3f CameraPinhole::vectorFromPixelIndices(const Index2D& u_C) const {
  // NOTE(alexmillane): The +0.5 here takes us from image plane indices, which
  // are equal to the coordinates of the lower pixel corner, to the pixel
  // center.
  return vectorFromImagePlaneCoordinates(u_C.cast<float>() +
                                         Vector2f(0.5, 0.5));
}

CameraPinhole CameraPinhole::fromIntrinsicsMatrix(const Eigen::Matrix3f& mat,
                                                  int width, int height) {
  return CameraPinhole(mat, width, height);
}

CameraPinhole CameraPinhole::fromIntrinsicsMatrix(const Matrix3x4f& P,
                                                  const Matrix3f& rect,
                                                  int width, int height) {
  return CameraPinhole(P, rect, width, height);
}

}  // namespace nvblox
