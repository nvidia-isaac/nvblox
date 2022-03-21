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
#include "nvblox/core/interpolation_3d.h"

#include <glog/logging.h>

namespace nvblox {
namespace interpolation {

bool interpolateOnCPU(const Vector3f& p_L, const TsdfLayer& layer,
                      float* distance_ptr) {
  CHECK_NOTNULL(distance_ptr);
  CHECK(layer.memory_type() == MemoryType::kUnified)
      << "For CPU-based interpolation, the layer must be CPU accessible (ie "
         "MemoryType::kUnified).";
  auto get_distance_lambda = [](const TsdfVoxel& voxel) -> float {
    return voxel.distance;
  };
  constexpr float kMinWeight = 1e-4;
  auto voxel_valid_lambda = [](const TsdfVoxel& voxel) -> bool {
    return voxel.weight > kMinWeight;
  };
  return internal::interpolateMemberOnCPU<TsdfVoxel>(
      p_L, layer, get_distance_lambda, voxel_valid_lambda, distance_ptr);
}

bool interpolateOnCPU(const Vector3f& p_L, const EsdfLayer& layer,
                      float* distance_ptr) {
  CHECK_NOTNULL(distance_ptr);
  CHECK(layer.memory_type() == MemoryType::kUnified)
      << "For CPU-based interpolation, the layer must be CPU accessible (ie "
         "MemoryType::kUnified).";
  auto get_distance_lambda = [](const EsdfVoxel& voxel) -> float {
    return std::sqrt(voxel.squared_distance_vox);
  };
  auto voxel_valid_lambda = [](const EsdfVoxel& voxel) -> bool {
    return voxel.observed;
  };
  return internal::interpolateMemberOnCPU<EsdfVoxel>(
      p_L, layer, get_distance_lambda, voxel_valid_lambda, distance_ptr);
}

}  // namespace interpolation
}  // namespace nvblox
