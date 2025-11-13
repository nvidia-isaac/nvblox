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

#include <cuda/std/tuple>

namespace nvblox {

/// Camera projection of a voxel onto the image plane.
/// Projects the center of the voxel associated with this GPU block/thread into
/// the image plane. Internally uses threadIdx and blockIdx to select the
/// appropriate voxel to project.
/// @param block_idx 3D block index of the voxel to be projected.
/// @param voxel_idx 3D voxel index of the voxel to be projected.
/// @param camera A the camera (intrinsics) model.
/// @param T_L_C The pose of the camera. Supplied as a Transform mapping
/// points in the camera frame (C) to the layer frame (L).
/// @param block_size The size of a VoxelBlock
/// @param max_depth The maximum depth at which we consider projection
/// sucessful.
/// @param[out] u_px_ptr A pointer to the (floating point) image plane
/// coordinates (u,v) of the voxel center projected on the image plane.
/// @param[out] u_depth_ptr A pointer to the depth of the voxel center.
/// @param[out] p_voxel_center_C_ptr A pointer to center of the voxel in camera
/// frame.
/// @return A flag indicating if the voxel projected within the image plane
/// bounds, and under the max depth.
template <typename SensorType>
__device__ inline bool projectThreadVoxel(
    const Index3D& block_idx, const Index3D& voxel_idx,
    const SensorType& camera, const Transform& T_C_L, const float block_size,
    const float max_depth, Eigen::Vector2f* u_px_ptr, float* u_depth_ptr,
    Vector3f* p_voxel_center_C_ptr);

/// Get Block and Voxel indices from ThreadIdx and BlockIdx
/// block_indices_device_ptr[blockIdx.x]:
///                 - The index of the block we're working on (blockIdx.y/z
///                   should be zero)
/// threadIdx.x/y/z - The indices of the voxel within the block (we
///                   expect the threadBlockDims == voxelBlockDims)
/// @param block_indices_device_ptr A list of block pointers.
/// @param[out] block_idx The block index for this thread.
/// @param[out] voxel_idx The voxel index for this thread.
/// @return A pair of block_idx and voxel_idx.
__device__ inline void voxelAndBlockIndexFromCudaThreadIndex(
    const Index3D* block_indices_device_ptr, Index3D* const block_idx,
    Index3D* voxel_idx);

/// Returns true if a voxel is in view of the camera, is not occluded, is not
/// out of max range, and has a valid depth measurment.
/// @param block_idx The block index of the voxel in question.
/// @param voxel_idx The voxel index of the voxel in question.
/// @param sensor The intrinsics of the sensor viewing the voxel.
/// @param depth_image The depth image.
/// @param T_C_L The pose of the viewing camera.
/// @param voxel_size The side-length of a voxel.
/// @param block_size The side-length of a block.
/// @param max_integration_distance The maximum distance at which we consider a
/// voxel in view.
/// @param truncation_distance_m The distance behind a surface, after which we
/// consider a voxel out-of-view.
/// @return True if the voxel is in view.
template <typename SensorType>
__device__ inline bool doesVoxelHaveDepthMeasurement(
    const Index3D& block_idx, const Index3D& voxel_idx,
    const SensorType& sensor, const DepthImageConstView depth_image,
    const Transform T_C_L, const float voxel_size, const float block_size,
    const float max_integration_distance, const float truncation_distance_m);

}  // namespace nvblox

#include "nvblox/integrators/internal/cuda/impl/projective_integrators_common_impl.cuh"
