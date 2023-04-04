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

#include <string>

#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/voxels.h"

namespace nvblox {
namespace io {

/// Outputs a voxel layer as a pointcloud with the lambda function deciding the
/// intensity.
/// The lambda outputs a boolean, saying whether that voxel should be
/// visualized, and an intensity which will be written to the pointcloud.
template <typename VoxelType>
bool outputVoxelLayerToPly(
    const VoxelBlockLayer<VoxelType>& layer, const std::string& filename,
    std::function<bool(const VoxelType* voxel, float* intensity)> lambda);

/// Without specifying a lambda, this outputs the distance as intensity.
template <typename VoxelType>
bool outputVoxelLayerToPly(const VoxelBlockLayer<VoxelType>& layer,
                           const std::string& filename);

/// Without specifying a lambda, this outputs the point with smaller distance.
template <typename VoxelType>
bool outputObstacleToPly(
    const VoxelBlockLayer<VoxelType>& layer, const std::string& filename,
    std::function<bool(const VoxelType* voxel, float* intensity)> lambda);

/// Specializations for the TSDF type.
template <>
bool outputVoxelLayerToPly(const TsdfLayer& layer, const std::string& filename);

/// Specialization for the ESDF type.
template <>
bool outputVoxelLayerToPly(const EsdfLayer& layer, const std::string& filename);

/// Specialization for the obstacle information.
// template <>
bool outputObstacleToPly(const EsdfLayer& layer, const std::string& filename);

}  // namespace io
}  // namespace nvblox

#include "nvblox/io/impl/pointcloud_io_impl.h"