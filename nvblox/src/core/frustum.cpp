#include "nvblox/core/frustum.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/camera_pinhole.h"

namespace nvblox {

template Frustum::Frustum(const Camera& camera, const Transform& T_L_C,
                          float min_depth, float max_depth);
template Frustum::Frustum(const CameraPinhole& camera, const Transform& T_L_C,
                          float min_depth, float max_depth);

// Frustum definitions.
template <typename CameraType>
Frustum::Frustum(const CameraType& camera, const Transform& T_L_C,
                 float min_depth, float max_depth) {
  Eigen::Matrix<float, 8, 3> corners_C =
      camera.getViewCorners(min_depth, max_depth);
  computeBoundingPlanes(corners_C, T_L_C);
}

void Frustum::computeBoundingPlanes(const Eigen::Matrix<float, 8, 3>& corners_C,
                                    const Transform& T_L_C) {
  // Transform the corners.
  const Eigen::Matrix<float, 8, 3> corners_L =
      (T_L_C * corners_C.transpose()).transpose();

  // Near plane first.
  bounding_planes_L_[0].setFromPoints(corners_L.row(0), corners_L.row(2),
                                      corners_L.row(1));
  // Far plane.
  bounding_planes_L_[1].setFromPoints(corners_L.row(4), corners_L.row(5),
                                      corners_L.row(6));
  // Left.
  bounding_planes_L_[2].setFromPoints(corners_L.row(3), corners_L.row(7),
                                      corners_L.row(6));
  // Right.
  bounding_planes_L_[3].setFromPoints(corners_L.row(0), corners_L.row(5),
                                      corners_L.row(4));
  // Top.
  bounding_planes_L_[4].setFromPoints(corners_L.row(3), corners_L.row(4),
                                      corners_L.row(7));
  // Bottom.
  bounding_planes_L_[5].setFromPoints(corners_L.row(2), corners_L.row(6),
                                      corners_L.row(5));

  // Calculate AABB.
  Vector3f aabb_min, aabb_max;

  aabb_min.setConstant(std::numeric_limits<double>::max());
  aabb_max.setConstant(std::numeric_limits<double>::lowest());

  for (int i = 0; i < corners_L.cols(); i++) {
    for (size_t j = 0; j < corners_L.rows(); j++) {
      aabb_min(i) = std::min(aabb_min(i), corners_L(j, i));
      aabb_max(i) = std::max(aabb_max(i), corners_L(j, i));
    }
  }

  aabb_ = AxisAlignedBoundingBox(aabb_min, aabb_max);
}

bool Frustum::isPointInView(const Vector3f& point) const {
  // Skip the AABB check, assume already been done.
  for (size_t i = 0; i < bounding_planes_L_.size(); i++) {
    if (!bounding_planes_L_[i].isPointInside(point)) {
      return false;
    }
  }
  return true;
}

bool Frustum::isAABBInView(const AxisAlignedBoundingBox& aabb) const {
  // If we're not even close, don't bother checking the planes.
  if (!aabb_.intersects(aabb)) {
    return false;
  }
  constexpr int kNumCorners = 8;

  // Check the center of the bounding box to see if it's within the AABB.
  // This covers a corner case where the given AABB is larger than the
  // frustum.
  if (isPointInView(aabb.center())) {
    return true;
  }

  // Iterate over all the corners of the bounding box and see if any are
  // within the view frustum.
  for (int i = 0; i < kNumCorners; i++) {
    if (isPointInView(
            aabb.corner(static_cast<Eigen::AlignedBox3f::CornerType>(i)))) {
      return true;
    }
  }
  return false;
}

// Bounding plane definitions.
void BoundingPlane::setFromPoints(const Vector3f& p1, const Vector3f& p2,
                                  const Vector3f& p3) {
  Vector3f p1p2 = p2 - p1;
  Vector3f p1p3 = p3 - p1;

  Vector3f cross = p1p2.cross(p1p3);
  normal_ = cross.normalized();
  distance_ = normal_.dot(p1);
}

void BoundingPlane::setFromDistanceNormal(const Vector3f& normal,
                                          float distance) {
  normal_ = normal;
  distance_ = distance;
}

bool BoundingPlane::isPointInside(const Vector3f& point) const {
  if (point.dot(normal_) >= distance_) {
    return true;
  }
  return false;
}
}  // namespace nvblox