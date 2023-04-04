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
#include "nvblox/core/camera_pinhole.h"

namespace nvblox {

std::ostream& operator<<(std::ostream& os, const CameraPinhole& camera) {
  if (camera.isRectified()) {
    os << "camera with intrinsics:\n\t"
       << "\trectified: " << camera.isRectified() << "\n"
       << "\tP: " << camera.P() << "\n"
       << "\tRect: " << camera.Rect() << "\n"
       << "\twidith: " << camera.width() << "\n"
       << "\theight: " << camera.height() << "\n";
  } else {
    os << "camera with intrinsics:\n\t"
       << "\trectified: " << camera.isRectified() << "\n"
       << "\tK: " << camera.K() << "\n"
       << "\twidith: " << camera.width() << "\n"
       << "\theight: " << camera.height() << "\n";
  }
  return os;
}

AxisAlignedBoundingBox CameraPinhole::getViewAABB(const Transform& T_L_C,
                                                  const float min_depth,
                                                  const float max_depth) const {
  // Get the bounding corners of this view.
  Eigen::Matrix<float, 8, 3> corners_C = getViewCorners(min_depth, max_depth);

  Vector3f aabb_min, aabb_max;
  aabb_min.setConstant(std::numeric_limits<float>::max());
  aabb_max.setConstant(std::numeric_limits<float>::lowest());

  // Transform it into the layer coordinate frame.
  for (size_t i = 0; i < corners_C.rows(); i++) {
    const Vector3f& corner_C = corners_C.row(i);
    Vector3f corner_L = T_L_C * corner_C;
    for (int i = 0; i < 3; i++) {
      aabb_min(i) = std::min(aabb_min(i), corner_L(i));
      aabb_max(i) = std::max(aabb_max(i), corner_L(i));
    }
  }

  return AxisAlignedBoundingBox(aabb_min, aabb_max);
}

Frustum CameraPinhole::getViewFrustum(const Transform& T_L_C,
                                      const float min_depth,
                                      const float max_depth) const {
  return Frustum(*this, T_L_C, min_depth, max_depth);
}

Eigen::Matrix<float, 8, 3> CameraPinhole::getViewCorners(
    const float min_depth, const float max_depth) const {
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

}  // namespace nvblox
