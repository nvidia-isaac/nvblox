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

#include "nvblox/core/types.h"

namespace nvblox {

class Frustum;
class BoundingPlane;

/// Class that describes the parameters and FoV of a camera.
class Camera {
 public:
  __host__ __device__ inline Camera() = default;
  __host__ __device__ inline Camera(float fu, float fv, float cu, float cv,
                                    int width, int height);
  __host__ __device__ inline Camera(float fu, float fv, int width, int height);

  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Vector2f* u_C) const;

  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

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
  __host__ __device__ inline float fu() const { return fu_; }
  __host__ __device__ inline float fv() const { return fv_; }
  __host__ __device__ inline float cu() const { return cu_; }
  __host__ __device__ inline float cv() const { return cv_; }
  __host__ __device__ inline int width() const { return width_; }
  __host__ __device__ inline int height() const { return height_; }
  __host__ __device__ inline int cols() const { return width_; }
  __host__ __device__ inline int rows() const { return height_; }

  // Factories
  inline static Camera fromIntrinsicsMatrix(const Eigen::Matrix3f& mat,
                                            int width, int height);

 private:
  float fu_;
  float fv_;
  float cu_;
  float cv_;

  int width_;
  int height_;
};

// Stream Camera as text
std::ostream& operator<<(std::ostream& os, const Camera& camera);

/// A bounding plane which has one "inside" direction and the other direction is
/// "outside." Quick tests for which side of the plane you are on.
class BoundingPlane {
 public:
  BoundingPlane() : normal_(Vector3f::Identity()), distance_(0.0f) {}

  void setFromPoints(const Vector3f& p1, const Vector3f& p2,
                     const Vector3f& p3);
  void setFromDistanceNormal(const Vector3f& normal, float distance);

  /// Is the point on the correct side of the bounding plane?
  bool isPointInside(const Vector3f& point) const;

  const Vector3f& normal() const { return normal_; }
  float distance() const { return distance_; }

 private:
  Vector3f normal_;
  float distance_;
};

/// Class that allows checking for whether objects are within the field of view
/// of a camera or not.
class Frustum {
 public:
  // Frustum must be initialized with a camera and min and max depth and pose.
  Frustum(const Camera& camera, const Transform& T_L_C, float min_depth,
          float max_depth);

  AxisAlignedBoundingBox getAABB() const { return aabb_; }

  bool isPointInView(const Vector3f& point) const;
  bool isAABBInView(const AxisAlignedBoundingBox& aabb) const;

 private:
  // Helper functions to do the actual computations.
  void computeBoundingPlanes(const Eigen::Matrix<float, 8, 3>& corners_C,
                             const Transform& T_L_C);

  /// Bounding planes containing around the frustum. Expressed in the layer
  /// coordinate frame.
  std::array<BoundingPlane, 6> bounding_planes_L_;

  /// Cached AABB of the
  AxisAlignedBoundingBox aabb_;
};

}  // namespace nvblox

#include "nvblox/core/impl/camera_impl.h"
