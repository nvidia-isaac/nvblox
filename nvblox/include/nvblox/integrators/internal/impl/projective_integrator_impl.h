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

namespace nvblox {

template <typename VoxelType>
float ProjectiveIntegrator<VoxelType>::
    lidar_linear_interpolation_max_allowable_difference_vox() const {
  return lidar_linear_interpolation_max_allowable_difference_vox_;
}

template <typename VoxelType>
float ProjectiveIntegrator<VoxelType>::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox() const {
  return lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_;
}

template <typename VoxelType>
void ProjectiveIntegrator<VoxelType>::
    lidar_linear_interpolation_max_allowable_difference_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_linear_interpolation_max_allowable_difference_vox_ = value;
}

template <typename VoxelType>
void ProjectiveIntegrator<VoxelType>::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = value;
}

template <typename LayerType>
float ProjectiveIntegrator<LayerType>::truncation_distance_vox() const {
  return truncation_distance_vox_;
}

template <typename LayerType>
float ProjectiveIntegrator<LayerType>::max_integration_distance_m() const {
  return max_integration_distance_m_;
}

template <typename LayerType>
void ProjectiveIntegrator<LayerType>::truncation_distance_vox(
    float truncation_distance_vox) {
  CHECK_GT(truncation_distance_vox, 0.0f);
  truncation_distance_vox_ = truncation_distance_vox;
}

template <typename LayerType>
void ProjectiveIntegrator<LayerType>::max_integration_distance_m(
    float max_integration_distance_m) {
  CHECK_GT(max_integration_distance_m, 0.0f);
  max_integration_distance_m_ = max_integration_distance_m;
}

template <typename LayerType>
float ProjectiveIntegrator<LayerType>::get_truncation_distance_m(
    float voxel_size) const {
  return truncation_distance_vox_ * voxel_size;
}

template <typename LayerType>
const ViewCalculator& ProjectiveIntegrator<LayerType>::view_calculator() const {
  return view_calculator_;
}

/// Returns the object used to calculate the blocks in camera views.
template <typename LayerType>
ViewCalculator& ProjectiveIntegrator<LayerType>::view_calculator() {
  return view_calculator_;
}

}  // namespace nvblox
