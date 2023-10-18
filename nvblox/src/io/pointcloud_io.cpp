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
  constexpr float kMinWeight = 1e-4f;
  auto lambda = [&kMinWeight](const TsdfVoxel* voxel,
                              float* ply_intensity) -> bool {
    *ply_intensity = voxel->distance;
    return voxel->weight > kMinWeight;
  };
  return outputVoxelLayerToPly<TsdfVoxel>(layer, filename, lambda);
}

/// Specialization for the freespace type.
template <>
bool outputVoxelLayerToPly(const FreespaceLayer& layer,
                           const std::string& filename) {
  auto lambda = [](const FreespaceVoxel* voxel, float* ply_intensity) -> bool {
    // Show freespace voxels with an intensity of 1
    *ply_intensity = voxel->is_high_confidence_freespace;
    return true;
  };
  return outputVoxelLayerToPly<FreespaceVoxel>(layer, filename, lambda);
}

/// Specialization for the occupancy type.
template <>
bool outputVoxelLayerToPly(const OccupancyLayer& layer,
                           const std::string& filename) {
  constexpr float kMinProbability = 0.5f;
  auto lambda = [&kMinProbability](const OccupancyVoxel* voxel,
                                   float* ply_intensity) -> bool {
    const float probability = probabilityFromLogOdds(voxel->log_odds);
    *ply_intensity = probability;
    return probability > kMinProbability;
  };
  return outputVoxelLayerToPly<OccupancyVoxel>(layer, filename, lambda);
}

/// Specialization for the ESDF type.
template <>
bool outputVoxelLayerToPly(const EsdfLayer& layer,
                           const std::string& filename) {
  const float voxel_size = layer.voxel_size();
  auto lambda = [&voxel_size](const EsdfVoxel* voxel,
                              float* ply_intensity) -> bool {
    *ply_intensity = voxel_size * std::sqrt(voxel->squared_distance_vox);
    if (voxel->is_inside) {
      *ply_intensity = -*ply_intensity;
    }
    return voxel->observed;
  };
  return outputVoxelLayerToPly<EsdfVoxel>(layer, filename, lambda);
}

bool outputPointMatrixToPly(const Eigen::Matrix3Xf& pointcloud,
                            const std::string& filename) {
  Eigen::VectorXf intensities(pointcloud.cols());
  intensities.setZero();
  return outputPointMatrixToPly(pointcloud, intensities, filename);
}

bool outputPointMatrixToPly(const Eigen::Matrix3Xf& pointcloud,
                            const Eigen::VectorXf& intensities,
                            const std::string& filename) {
  CHECK(pointcloud.cols() == intensities.size());

  // Create a ply writer object.
  io::PlyWriter writer(filename);

  // Write the points in the matrix to a vector
  std::vector<Vector3f> points(pointcloud.cols());
  Eigen::Matrix3Xf::Map(points.data()->data(), 3, pointcloud.cols()) =
      pointcloud;

  // Write the intensities to a vector
  std::vector<float> intensities_vec(intensities.data(),
                                     intensities.data() + intensities.size());

  // Add the pointcloud to the ply writer.
  writer.setPoints(&points);
  writer.setIntensities(&intensities_vec);

  // Write out the ply.
  return writer.write();
}

}  // namespace io
}  // namespace nvblox
