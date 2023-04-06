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

#include "nvblox/core/log_odds.h"
#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

namespace nvblox {
namespace interpolation {

/// Single points
bool interpolateOnCPU(const Vector3f& p_L, const TsdfLayer& layer,
                      float* distance);
bool interpolateOnCPU(const Vector3f& p_L, const EsdfLayer& layer,
                      float* distance);
bool interpolateOnCPU(const Vector3f& p_L, const OccupancyLayer& layer,
                      float* distance);

/// Vectors of points
template <typename VoxelType>
void interpolateOnCPU(const std::vector<Vector3f>& points_L,
                      const VoxelBlockLayer<VoxelType>& layer,
                      std::vector<float>* distances_ptr,
                      std::vector<bool>* success_flags_ptr);

}  // namespace interpolation
}  // namespace nvblox

#include "nvblox/interpolation/internal/impl/interpolation_3d_impl.h"
