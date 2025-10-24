/*
Copyright 2025 NVIDIA CORPORATION

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

#include <optional>

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// A structure describing a view of the scene.
template <typename SensorType>
struct DepthObservationSpace {
  DepthObservationSpace() = delete;
  explicit DepthObservationSpace(
      Transform _T_L_C, SensorType _sensor,
      std::optional<DepthImageConstView> _depth_image = std::nullopt,
      std::optional<float> _max_view_distance_m = std::nullopt,
      std::optional<float> _truncation_distance_m = std::nullopt)
      : T_L_C(_T_L_C),
        sensor(_sensor),
        depth_image(_depth_image),
        max_view_distance_m(_max_view_distance_m),
        truncation_distance_m(_truncation_distance_m) {}

  ~DepthObservationSpace() = default;

  /// The pose of the sensor for view-based occlusion testing.
  Transform T_L_C;
  /// The intrinsics of the sensor for view-based occlusion testing.
  SensorType sensor;
  /// The depth image tested for valid depth during view-based occlusion
  /// testing.
  std::optional<DepthImageConstView> depth_image;
  /// The maximum depth at which a voxel is considered in view. If these are not
  /// provided the max distance is infinite.
  std::optional<float> max_view_distance_m;
  /// truncation_distance_m behind the depth measurement is considered occluded
  /// and will be decayed. If this is not provided, we do not do occlusion
  /// testing.
  std::optional<float> truncation_distance_m;
};

/// Storage for view data including sensor, pose, and owned depth image.
/// Used for storing the last view per sensor type.
template <typename SensorType>
struct PosedDepthImage {
  PosedDepthImage() = delete;
  explicit PosedDepthImage(Transform _T_L_C, SensorType _sensor,
                           DepthImage&& _depth_image)
      : T_L_C(_T_L_C), sensor(_sensor), depth_image(std::move(_depth_image)) {}

  /// Convert to DepthObservationSpace for use with integrators
  DepthObservationSpace<SensorType> toDepthObservationSpace(
      std::optional<float> max_view_distance_m = std::nullopt,
      std::optional<float> truncation_distance_m = std::nullopt) const {
    return DepthObservationSpace<SensorType>(
        T_L_C, sensor, depth_image, max_view_distance_m, truncation_distance_m);
  }

  Transform T_L_C;
  SensorType sensor;
  DepthImage depth_image;
};

}  // namespace nvblox
