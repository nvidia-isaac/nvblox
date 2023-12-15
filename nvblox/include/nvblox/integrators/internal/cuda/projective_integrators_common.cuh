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
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// Camera projection of a voxel onto the image plane.
/// Projects the center of the voxel associated with this GPU block/thread into
/// the image plane. Internally uses threadIdx and blockIdx to select the
/// appropriate voxel to project.
/// @param block_indices_device_ptr A vector containing a list of block indices
/// to be projected.
/// @param camera A the camera (intrinsics) model.
/// @param T_L_C The pose of the camera. Supplied as a Transform mapping
/// points in the camera frame (C) to the layer frame (L).
/// @param block_size The size of a VoxelBlock
/// @param max_depth The maximum depth at which we consider projection sucessful.
/// @param[out] u_px_ptr A pointer to the (floating point) image plane
/// coordinates (u,v) of the voxel center projected on the image plane.
/// @param[out] u_depth_ptr A pointer to the depth of the voxel center.
/// @return A flag indicating if the voxel projected within the image plane
/// bounds, and under the max depth.
__device__ inline bool projectThreadVoxel(
    const Index3D* block_indices_device_ptr, const Camera& camera,
    const Transform& T_C_L, const float block_size, const float max_depth,
    Eigen::Vector2f* u_px_ptr, float* u_depth_ptr);

}  // namespace nvblox

#include "nvblox/integrators/internal/cuda/impl/projective_integrators_common_impl.cuh"
