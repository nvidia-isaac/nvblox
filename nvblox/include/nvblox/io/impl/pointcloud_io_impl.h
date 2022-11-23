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

#include "nvblox/core/accessors.h"
#include "nvblox/io/ply_writer.h"

namespace nvblox {
namespace io {

/// NOTE(jjiao): This function outputs TSDF and ESDF map to a file
/// Outputs a voxel layer as a pointcloud with the lambda function deciding the
/// intensity.
template <typename VoxelType>
bool outputVoxelLayerToPly(
    const VoxelBlockLayer<VoxelType>& layer, const std::string& filename,
    std::function<bool(const VoxelType* voxel, float* intensity)> lambda) {
  // Create a ply writer object.
  io::PlyWriter writer(filename);

  // Combine all the voxels in the mesh into a pointcloud.
  std::vector<Vector3f> points;
  std::vector<float> intensities;  // NOTE(jjiao): record the distance

  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;
  const float block_size = layer.block_size();
  const float voxel_size = layer.voxel_size();

  auto new_lambda = [&points, &intensities, &block_size, &voxel_size, &lambda](
                        const Index3D& block_index, const Index3D& voxel_index,
                        const VoxelType* voxel) {
    float intensity = 0.0f;
    if (lambda(voxel, &intensity)) {
      points.push_back(getCenterPostionFromBlockIndexAndVoxelIndex(
          block_size, block_index, voxel_index));
      intensities.push_back(intensity);
    }
  };

  // Call above lambda on every voxel in the layer.
  callFunctionOnAllVoxels<VoxelType>(layer, new_lambda);

  // Add the pointcloud to the ply writer.
  writer.setPoints(&points);
  writer.setIntensities(&intensities);

  // Write out the ply.
  return writer.write();
}

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

/// NOTE(jjiao): This function outputs obstance information to a file
template <typename VoxelType>
bool outputObstacleToPly(
    const VoxelBlockLayer<VoxelType>& layer, const std::string& filename,
    std::function<bool(const VoxelType* voxel, float* intensity)> lambda) {
  // Create a ply writer object.
  io::PlyWriter writer(filename);

  // Combine all the voxels in the mesh into a pointcloud.
  std::vector<Vector3f> points;
  std::vector<float> intensities;

  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;
  const float block_size = layer.block_size();
  const float voxel_size = layer.voxel_size();

  auto new_lambda = [&points, &intensities, &block_size, &voxel_size, &lambda](
                        const Index3D& block_index, const Index3D& voxel_index,
                        const VoxelType* voxel) {
    float intensity = 0.0f;
    if (lambda(voxel, &intensity)) {
      points.push_back(getCenterPostionFromBlockIndexAndVoxelIndex(
          block_size, block_index, voxel_index));
      intensities.push_back(intensity);
    }
  };

  // Call above lambda on every voxel in the layer.
  callFunctionOnAllVoxels<VoxelType>(layer, new_lambda);

  // Add the obstacle pointcloud to the ply writer.
  const float OBSTACLE_DISTANCE_TH = 0.02f;
  std::vector<Vector3f> obs_points;
  obs_points.reserve(points.size() / 10);
  for (size_t i = 0; i < points.size(); i++) {
    if (abs(intensities[i]) <= OBSTACLE_DISTANCE_TH) {
      obs_points.push_back(points[i]);
    }
  }
  writer.setPoints(&obs_points);

  // Write out the ply.
  return writer.write();
}

/// Specialization for the obstacle information.
// template <>
bool outputObstacleToPly(const EsdfLayer& layer, const std::string& filename) {
  const float voxel_size = layer.voxel_size();
  auto lambda = [&voxel_size](const EsdfVoxel* voxel, float* distance) -> bool {
    *distance = voxel_size * std::sqrt(voxel->squared_distance_vox);
    if (voxel->is_inside) {
      *distance = -*distance;
    }
    return voxel->observed;
  };
  return outputObstacleToPly<EsdfVoxel>(layer, filename, lambda);
}

}  // namespace io
}  // namespace nvblox