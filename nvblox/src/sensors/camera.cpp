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
#include "nvblox/sensors/camera.h"

namespace nvblox {

std::ostream& operator<<(std::ostream& os, const Camera& camera) {
  os << "camera with intrinsics:\n\tfu: " << camera.fu() << "\n"
     << "\tfv: " << camera.fv() << "\n"
     << "\tcu: " << camera.cu() << "\n"
     << "\tcv: " << camera.cv() << "\n"
     << "\twidith: " << camera.width() << "\n"
     << "\theight: " << camera.height() << "\n";
  return os;
}

bool camerasAreEquivalent(const Camera& camera_1, const Camera& camera_2,
                          const Transform& T_L_C1, const Transform& T_L_C2) {
  // Check that the cameras have the same extrinsics
  bool same_extrinsics = true;
  const Transform T_C1_C2 = T_L_C1.inverse() * T_L_C2;
  const float distance_between_cameras_m = T_C1_C2.translation().norm();
  const float angular_between_cameras_rad =
      Eigen::AngleAxisf(T_C1_C2.rotation()).angle();
  same_extrinsics &= distance_between_cameras_m <= 0.001f;
  same_extrinsics &= std::abs(angular_between_cameras_rad) <= 0.01f;
  if (!same_extrinsics) {
    LOG(ERROR) << "The cameras do not have the same extrinsics.";
  }

  // Check that the cameras have the same intrinsics
  bool same_intrinsics = true;
  same_intrinsics &= std::abs(camera_1.fu() - camera_2.fu()) <= 0.1;
  same_intrinsics &= std::abs(camera_1.fv() - camera_2.fv()) <= 0.1;
  same_intrinsics &= std::abs(camera_1.cu() - camera_2.cu()) <= 0.1;
  same_intrinsics &= std::abs(camera_1.cv() - camera_2.cv()) <= 0.1;
  same_intrinsics &= camera_1.width() == camera_2.width();
  same_intrinsics &= camera_1.height() == camera_2.height();
  if (!same_intrinsics) {
    LOG(ERROR) << "The cameras do not have the same intrinsics.";
  }

  return same_extrinsics && same_intrinsics;
}

AxisAlignedBoundingBox Camera::getViewAABB(const Transform& T_L_C,
                                           const float min_depth,
                                           const float max_depth) const {
  // Get the bounding corners of this view.
  Eigen::Matrix<float, 8, 3> corners_C = getViewCorners(min_depth, max_depth);

  Vector3f aabb_min, aabb_max;
  aabb_min.setConstant(std::numeric_limits<float>::max());
  aabb_max.setConstant(std::numeric_limits<float>::lowest());

  // Transform it into the layer coordinate frame.
  for (int corner_idx = 0; corner_idx < corners_C.rows(); corner_idx++) {
    const Vector3f& corner_C = corners_C.row(corner_idx);
    Vector3f corner_L = T_L_C * corner_C;
    for (int i = 0; i < 3; i++) {
      aabb_min(i) = std::min(aabb_min(i), corner_L(i));
      aabb_max(i) = std::max(aabb_max(i), corner_L(i));
    }
  }

  return AxisAlignedBoundingBox(aabb_min, aabb_max);
}

Frustum Camera::getViewFrustum(const Transform& T_L_C, const float min_depth,
                               const float max_depth) const {
  return Frustum(*this, T_L_C, min_depth, max_depth);
}

Eigen::Matrix<float, 8, 3> Camera::getViewCorners(const float min_depth,
                                                  const float max_depth) const {
  // Rays through the corners of the image plane
  // Clockwise from the top left corner of the image.
  const Vector3f ray_0_C =
      vectorFromImagePlaneCoordinates(Vector2f(0.0f, 0.0f));  // NOLINT
  const Vector3f ray_1_C =
      vectorFromImagePlaneCoordinates(Vector2f(width_, 0.0f));  // NOLINT
  const Vector3f ray_2_C =
      vectorFromImagePlaneCoordinates(Vector2f(width_, height_));  // NOLINT
  const Vector3f ray_3_C =
      vectorFromImagePlaneCoordinates(Vector2f(0.0f, height_));  // NOLINT

  // True bounding box from the 3D points
  Eigen::Matrix<float, 8, 3> corners_C;
  corners_C.row(0) = min_depth * ray_2_C;
  corners_C.row(1) = min_depth * ray_1_C;
  corners_C.row(2) = min_depth * ray_0_C,
  corners_C.row(3) = min_depth * ray_3_C;
  corners_C.row(4) = max_depth * ray_2_C;
  corners_C.row(5) = max_depth * ray_1_C;
  corners_C.row(6) = max_depth * ray_0_C;
  corners_C.row(7) = max_depth * ray_3_C;
  return corners_C;
}

CameraViewport Camera::getNormalizedViewport(const float margin_pixels) const {
  // Get the normalized coordinates of the points defining the
  // viewport: (0, 0, width, height) with added margin. Normalized
  // coordinates are obtained by "undoing" the camera intrinsics,
  // i.e. multiplying a pixel coordinate with inverse(K). This is the
  // same operation we perform inside vectorFromImagePlaneCoordinates()
  const Eigen::Vector3f min_coord =
      vectorFromImagePlaneCoordinates({-margin_pixels, -margin_pixels});
  const Eigen::Vector3f max_coord = vectorFromImagePlaneCoordinates(
      {width() + margin_pixels, height() + margin_pixels});
  return CameraViewport(min_coord.head<2>(), max_coord.head<2>());
}

// Frustum definitions.
Frustum::Frustum(const Camera& camera, const Transform& T_L_C, float min_depth,
                 float max_depth) {
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
    for (int j = 0; j < corners_L.rows(); j++) {
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
