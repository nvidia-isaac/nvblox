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
#include "nvblox/interpolation/interpolation_3d.h"

#include "nvblox/utils/logging.h"

namespace nvblox {
namespace interpolation {

bool interpolateOnCPU(const Vector3f& p_L, const TsdfLayer& layer,
                      float* distance_ptr) {
  CHECK_NOTNULL(distance_ptr);
  CHECK(layer.memory_type() == MemoryType::kHost ||
        layer.memory_type() == MemoryType::kUnified)
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
  CHECK(layer.memory_type() == MemoryType::kHost ||
        layer.memory_type() == MemoryType::kUnified)
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

bool interpolateOnCPU(const Vector3f& p_L, const OccupancyLayer& layer,
                      float* probability_ptr) {
  CHECK_NOTNULL(probability_ptr);
  CHECK(layer.memory_type() == MemoryType::kHost ||
        layer.memory_type() == MemoryType::kUnified)
      << "For CPU-based interpolation, the layer must be CPU accessible (ie "
         "MemoryType::kUnified).";
  auto get_probability_lambda = [](const OccupancyVoxel& voxel) -> float {
    return probabilityFromLogOdds(voxel.log_odds);
  };
  auto voxel_valid_lambda = [](const OccupancyVoxel&) -> bool {
    return true;
  };
  return internal::interpolateMemberOnCPU<OccupancyVoxel>(
      p_L, layer, get_probability_lambda, voxel_valid_lambda, probability_ptr);
}

namespace internal {

Eigen::Matrix<float, 8, 1> getQVector3D(const Vector3f& p_offset_in_voxels_L) {
  // Units are in voxels
  CHECK((p_offset_in_voxels_L.array() >= 0.0f).all());
  CHECK((p_offset_in_voxels_L.array() <= 1.0f).all());

  constexpr float kEps = 1e-4;
  constexpr float kOnePlusEps = 1.0f + kEps;
  CHECK((p_offset_in_voxels_L.array() <= kOnePlusEps).all());
  // FROM PAPER (http://spie.org/samples/PM159.pdf)
  // clang-format off
  Eigen::Matrix<float, 8, 1> q_vector;
  q_vector <<
      1,
      p_offset_in_voxels_L[0],
      p_offset_in_voxels_L[1],
      p_offset_in_voxels_L[2],
      p_offset_in_voxels_L[0] * p_offset_in_voxels_L[1],
      p_offset_in_voxels_L[1] * p_offset_in_voxels_L[2],
      p_offset_in_voxels_L[2] * p_offset_in_voxels_L[0],
      p_offset_in_voxels_L[0] * p_offset_in_voxels_L[1] * p_offset_in_voxels_L[2];
  // clang-format on
  return q_vector;
}

}  // namespace internal
}  // namespace interpolation
}  // namespace nvblox
