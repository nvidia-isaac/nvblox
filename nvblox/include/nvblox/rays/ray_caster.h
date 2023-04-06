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

#include "nvblox/core/types.h"

namespace nvblox {

/// Class for casting rays through a voxelized space from origin to destination.
class RayCaster {
 public:
  __host__ __device__ inline RayCaster(const Vector3f& origin,
                                       const Vector3f& destination,
                                       float scale = 1.0f);

  /// Returns the index, so in "unscaled" coordinates.
  __host__ __device__ inline bool nextRayIndex(Index3D* ray_index);
  /// Returns scaled coordinates. Just the above multiplied by the scale factor.
  __host__ __device__ inline bool nextRayPositionScaled(Vector3f* ray_position);

  /// Just raycasts over the whole thing and puts them in a vector for you.
  __host__ inline void getAllIndices(std::vector<Index3D>* indices);

 private:
  const float scale_ = 1.0f;

  uint32_t ray_length_in_steps_ = 0;
  uint32_t current_step_ = 0;

  Vector3f t_to_next_boundary_;
  Index3D current_index_;
  Index3D ray_step_signs_;
  Vector3f t_step_size_;
};

}  // namespace nvblox

#include "nvblox/rays/internal/impl/ray_caster_impl.h"
