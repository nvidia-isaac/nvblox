/*
Copyright 2024 NVIDIA CORPORATION

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
#include "nvblox/geometry/transforms.h"

#include <algorithm>
#include <cmath>

#include "nvblox/core/types.h"
#include "nvblox/geometry/plane.h"

namespace nvblox {

bool arePosesClose(const Transform& T_A_B1, const Transform& T_A_B2,
                   const float translation_tolerance_m,
                   const float angular_tolerance_deg) {
  // Check that the cameras have the same extrinsics
  const Transform T_B1_B2 = T_A_B1.inverse() * T_A_B2;
  if (T_B1_B2.translation().norm() > translation_tolerance_m) {
    return false;
  }
  const float angle_between_cameras_rad =
      Eigen::AngleAxisf(T_B1_B2.rotation()).angle();
  const float angle_between_cameras_deg =
      angle_between_cameras_rad * 180.0f / M_PI;
  if (std::abs(angle_between_cameras_deg) > angular_tolerance_deg) {
    return false;
  }
  return true;
}

Transform computeTransformToAlignPlaneToZ0(const Plane& ground_plane) {
  Vector3f plane_normal = ground_plane.normal();
  float plane_offset = ground_plane.offset();

  // If the plane normal points downward, negate both normal and offset
  if (plane_normal.z() < 0.0f) {
    plane_normal = -plane_normal;
    plane_offset = -plane_offset;
  }

  Vector3f plane_point = -plane_normal * plane_offset;

  Vector3f target_normal(0.0f, 0.0f, 1.0f);

  Transform T_plane_to_z0 = Transform::Identity();

  const float dot_product = plane_normal.dot(target_normal);
  const float alignment_threshold = 1e-6f;

  if (std::abs(dot_product - 1.0f) < alignment_threshold) {
    T_plane_to_z0.linear() = Eigen::Matrix3f::Identity();
  } else {
    // General case: compute rotation using Rodrigues' formula
    Vector3f rotation_axis = plane_normal.cross(target_normal).normalized();
    float angle = std::acos(std::clamp(dot_product, -1.0f, 1.0f));
    T_plane_to_z0.linear() = Eigen::AngleAxisf(angle, rotation_axis).matrix();
  }

  // Compute translation to move plane point to z=0 after rotation
  Vector3f rotated_plane_point = T_plane_to_z0.linear() * plane_point;
  T_plane_to_z0.translation() = Vector3f(0.0f, 0.0f, -rotated_plane_point.z());

  return T_plane_to_z0;
}

}  // namespace nvblox
