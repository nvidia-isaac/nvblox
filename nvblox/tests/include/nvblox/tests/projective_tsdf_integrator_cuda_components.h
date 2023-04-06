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
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace test_utils {

// A Nx2 matrix representing the results of projecting all voxels in a block
// into the image plane
using BlockProjectionResult =
    Eigen::Matrix<float,
                  VoxelBlock<bool>::kVoxelsPerSide *
                      VoxelBlock<bool>::kVoxelsPerSide *
                      VoxelBlock<bool>::kVoxelsPerSide,
                  2>;

Eigen::Matrix3Xf transformPointsOnGPU(const Transform& T_A_B,
                                      const Eigen::Matrix3Xf& vecs_A);

std::vector<BlockProjectionResult> projectBlocksOnGPU(
    const std::vector<Index3D>& block_indices, const Camera& camera,
    const Transform& T_C_L, TsdfLayer* distance_layer_ptr);

Eigen::VectorXf interpolatePointsOnGPU(const DepthImage& depth_frame,
                                       const Eigen::MatrixX2f& u_px_vec);

void setVoxelBlockOnGPU(TsdfLayer* layer);

}  // namespace test_utils
}  // namespace nvblox
