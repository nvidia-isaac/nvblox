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
#include "nvblox/io/pointcloud_io.h"

namespace nvblox {
namespace io {

/// Specializations for the TSDF type.
template <>
bool outputVoxelLayerToPly(const TsdfLayer& layer,
                           const std::string& filename) {
  constexpr float kMinWeight = 0.1f;
  auto lambda = [&kMinWeight](const TsdfVoxel* voxel, float* distance) -> bool {
    *distance = voxel->distance;
    return voxel->weight > kMinWeight;
  };
  return outputVoxelLayerToPly<TsdfVoxel>(layer, filename, lambda);
}

/// Specialization for the occupancy type.
template <>
bool outputVoxelLayerToPly(const OccupancyLayer& layer,
                           const std::string& filename) {
  constexpr float kMinProbability = 0.5f;
  auto lambda = [&kMinProbability](const OccupancyVoxel* voxel,
                                   float* probability) -> bool {
    *probability = probabilityFromLogOdds(voxel->log_odds);
    return *probability > kMinProbability;
  };
  return outputVoxelLayerToPly<OccupancyVoxel>(layer, filename, lambda);
}

/// Specialization for the ESDF type.
template <>
bool outputVoxelLayerToPly(const EsdfLayer& layer,
                           const std::string& filename) {
  const float voxel_size = layer.voxel_size();
  auto lambda = [&voxel_size](const EsdfVoxel* voxel, float* distance) -> bool {
    *distance = voxel_size * std::sqrt(voxel->squared_distance_vox);
    if (voxel->is_inside) {
      *distance = -*distance;
    }
    return voxel->observed;
  };
  return outputVoxelLayerToPly<EsdfVoxel>(layer, filename, lambda);
}

}  // namespace io
}  // namespace nvblox
