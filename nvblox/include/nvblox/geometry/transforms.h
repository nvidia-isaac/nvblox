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
#pragma once

#include "nvblox/core/types.h"
#include "nvblox/geometry/plane.h"

namespace nvblox {

/// Returns true if two poses are the same within some tolerance.
/// @param T_A_B1 Pose #1.
/// @param T_A_B2 Pose #2.
/// @param translation_tolerance_m Translation below which poses are close, in
/// meters.
/// @param angular_tolerance_deg Angle below which poses are close, in degrees.
/// @return True if poses pass both translation and rotation check.
bool arePosesClose(const Transform& T_A_B1, const Transform& T_A_B2,
                   const float translation_tolerance_m,
                   const float angular_tolerance_deg);

/// Computes a 3D transform (rotation and translation) that transforms
/// the given ground plane to the horizontal z=0 plane.
/// @param ground_plane The ground plane to align.
/// @return Transform from the original frame to the z=0 aligned frame.
Transform computeTransformToAlignPlaneToZ0(const Plane& ground_plane);

}  // namespace nvblox
