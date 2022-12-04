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

#include "nvblox/core/frustum.h"
#include "nvblox/core/types.h"

namespace nvblox {

/// Class that describes the parameters and FoV of a camera.
class CameraPinhole {
 public:
  __host__ __device__ inline CameraPinhole() = default;
  __host__ __device__ inline CameraPinhole(const Matrix3f& K, const int width,
                                           const int height);
  __host__ __device__ inline CameraPinhole(const Matrix3x4f& P,
                                           const Matrix3f& rect,
                                           const int width, const int height);

  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Vector2f* u_C) const;

  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

  // Back projection (image plane point to 3D point)
  __host__ __device__ inline Vector3f unprojectFromImagePlaneCoordinates(
      const Vector2f& u_C, const float depth) const;
  __host__ __device__ inline Vector3f unprojectFromPixelIndices(
      const Index2D& u_C, const float depth) const;

  /// Get the axis aligned bounding box of the view in the LAYER coordinate
  /// frame.
  __host__ AxisAlignedBoundingBox getViewAABB(const Transform& T_L_C,
                                              const float min_depth,
                                              const float max_depth) const;

  __host__ Frustum getViewFrustum(const Transform& T_L_C, const float min_depth,
                                  const float max_depth) const;

  /// Gets the view corners in the CAMERA coordinate frame.
  __host__ Eigen::Matrix<float, 8, 3> getViewCorners(
      const float min_depth, const float max_depth) const;

  // Returns an unnormalized ray direction in the camera frame corresponding to
  // the passed pixel.
  // Two functions, one for (floating point) image-plane coordinates, another
  // function for pixel indices.
  __host__ __device__ inline Vector3f vectorFromImagePlaneCoordinates(
      const Vector2f& u_C) const;
  __host__ __device__ inline Vector3f vectorFromPixelIndices(
      const Index2D& u_C) const;

  // Accessors
  __host__ __device__ inline Matrix3f K() const { return K_; }
  __host__ __device__ inline Matrix3x4f P() const { return P_; }
  __host__ __device__ inline Matrix3f Rect() const { return R_rect_; }
  __host__ __device__ inline int width() const { return width_; }
  __host__ __device__ inline int height() const { return height_; }
  __host__ __device__ inline int cols() const { return width_; }
  __host__ __device__ inline int rows() const { return height_; }
  __host__ __device__ inline bool isRectified() const { return rectified_; }

  // Factories
  inline static CameraPinhole fromIntrinsicsMatrix(const Matrix3f& mat,
                                                   int width, int height);

  inline static CameraPinhole fromIntrinsicsMatrix(const Matrix3x4f& P,
                                                   const Matrix3f& rect,
                                                   int width, int height);

 private:
  Matrix3f K_;
  /// NOTE(gogojjh):
  /// KITTI dataset: P_ = P_rect_00, we first transform the point into a camera
  /// P_(0:2, 3) = 0, P_(3, 3) = 1
  Matrix3x4f P_;
  /// KITTI dataset: R_rect_ = R_rect_0x
  Matrix3f R_rect_;

  int width_;
  int height_;
  bool rectified_;
};

// Stream Camera as text
std::ostream& operator<<(std::ostream& os, const CameraPinhole& camera);

}  // namespace nvblox

#include "nvblox/core/impl/camera_pinhole_impl.h"
