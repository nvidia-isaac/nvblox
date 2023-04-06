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

#include "nvblox/core/indexing.h"

namespace nvblox {

__host__ __device__ inline int signum(float x) {
  return (x > 0.0f) ? 1 : ((x < 0.0f) ? -1 : 0);
}

inline RayCaster::RayCaster(const Vector3f& origin, const Vector3f& destination,
                            float scale)
    : scale_(scale) {
  Vector3f start_scaled = origin / scale_;
  Vector3f end_scaled = destination / scale_;

  current_index_ = getBlockIndexFromPositionInLayer(scale, origin);
  const Index3D end_index =
      getBlockIndexFromPositionInLayer(scale, destination);
  const Index3D diff_index = end_index - current_index_;

  current_step_ = 0;

  ray_length_in_steps_ = diff_index.cwiseAbs().sum();
  const Vector3f ray_scaled = end_scaled - start_scaled;

  ray_step_signs_ = Index3D(signum(ray_scaled.x()), signum(ray_scaled.y()),
                            signum(ray_scaled.z()));

  const Index3D corrected_step = ray_step_signs_.cwiseMax(0);

  const Vector3f start_scaled_shifted =
      start_scaled - current_index_.cast<float>();

  Vector3f distance_to_boundaries(corrected_step.cast<float>() -
                                  start_scaled_shifted);

  // NaNs are fine in the next 2 lines.
  t_to_next_boundary_ = distance_to_boundaries.cwiseQuotient(ray_scaled);

  // Distance to cross one grid cell along the ray in t.
  // Same as absolute inverse value of delta_coord.
  t_step_size_ = ray_step_signs_.cast<float>().cwiseQuotient(ray_scaled);
}

inline bool RayCaster::nextRayIndex(Index3D* ray_index) {
  if (current_step_++ > ray_length_in_steps_) {
    return false;
  }

  DCHECK(ray_index != nullptr);
  *ray_index = current_index_;

  int t_min_idx;
  t_to_next_boundary_.minCoeff(&t_min_idx);
  current_index_[t_min_idx] += ray_step_signs_[t_min_idx];
  t_to_next_boundary_[t_min_idx] += t_step_size_[t_min_idx];

  return true;
}

inline bool RayCaster::nextRayPositionScaled(Vector3f* ray_position) {
  Index3D ray_index;
  bool success = nextRayIndex(&ray_index);
  *ray_position = scale_ * ray_index.cast<float>();
  return success;
}

inline void RayCaster::getAllIndices(std::vector<Index3D>* indices) {
  indices->clear();
  Index3D next_index;
  while (nextRayIndex(&next_index)) {
    indices->push_back(next_index);
  }
}

}  // namespace nvblox