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

#include <Eigen/Core>

#include "nvblox/core/color.h"

namespace nvblox {

struct TsdfVoxel {
  // Signed projective distance of the voxel from a surface.
  float distance = 0.0f;
  // How many observations/how confident we are in this observation.
  float weight = 0.0f;
  // ADD(jjiao): for the implementation of signed distance gradient, its
  // direction is from the surface toward the sensor // NOLINT
  Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
};

struct EsdfVoxel {
  // TODO(helen): optimize the memory layout here.
  // Cached squared distance towards the parent.
  float squared_distance_vox = 0.0f;
  // Direction towards the parent, *in units of voxels*.
  Eigen::Vector3i parent_direction = Eigen::Vector3i::Zero();
  // Whether this voxel is inside the surface or not.
  bool is_inside = false;
  // Whether this voxel has been observed.
  bool observed = false;
  // Whether this voxel is a "site": i.e., near the zero-crossing and is
  // eligible to be considered a parent.
  bool is_site = false;
};

struct ColorVoxel {
  Color color = Color::Gray();
  // How many observations/how confident we are in this observation.
  float weight = 0.0f;
};

}  // namespace nvblox
