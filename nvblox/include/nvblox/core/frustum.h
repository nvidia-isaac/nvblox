#pragma once

#include "nvblox/core/types.h"

namespace nvblox {

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
  template <typename CameraType>
  Frustum(const CameraType& camera, const Transform& T_L_C, float min_depth,
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