/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include <vector>

#include "nvblox/integrators/internal/projective_integrator.h"

#include "nvblox/core/cuda_stream.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

/*****************************************************************************
 * File internal helper functions
 ******************************************************************************/

namespace {

std::pair<int, dim3> getLaunchSizes(int num_voxel_blocks) {
  // We call all kernels in this file with:
  // - One threadBlock per VoxelBlock
  // - NxNxN threads where N is the block side-length in voxels.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_voxel_blocks;
  return {num_thread_blocks, kThreadsPerBlock};
}

}  // namespace

/*****************************************************************************
 * Kernels
 ******************************************************************************/

// CAMERA
template <typename VoxelType, typename UpdateFunctor>
__global__ void integrateBlocksKernel(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const float* image, int rows, int cols, const Transform T_C_L,
    const float block_size, const float max_integration_distance,
    UpdateFunctor* op, VoxelBlock<VoxelType>** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return;
  }

  // Interpolate on the image plane
  float image_value;
  if (!interpolation::interpolate2DClosest<
          float, interpolation::checkers::FloatPixelGreaterThanZero>(
          image, u_px, rows, cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  VoxelType* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  (*op)(image_value, voxel_depth_m, voxel_ptr);
}

// LIDAR
template <typename VoxelType, typename UpdateFunctor>
__global__ void integrateBlocksKernel(
    const Index3D* block_indices_device_ptr, const Lidar lidar,
    const float* image, int rows, int cols, const Transform T_C_L,
    const float block_size, const float max_integration_distance,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    UpdateFunctor* op, VoxelBlock<VoxelType>** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, lidar, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return;
  }

  // Interpolate on the image plane
  float image_value;
  if (!interpolation::interpolateLidarImage(
          lidar, p_voxel_center_C, image, u_px, rows, cols,
          linear_interpolation_max_allowable_difference_m,
          nearest_interpolation_max_allowable_squared_dist_to_ray_m,
          &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  VoxelType* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  (*op)(image_value, voxel_depth_m, voxel_ptr);
}

// COLOR
template <typename UpdateFunctor>
__global__ void integrateBlocksKernel(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Color* color_image, const int color_rows, const int color_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float max_integration_distance, const int depth_subsample_factor,
    UpdateFunctor* op, ColorBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel<Camera>(block_indices_device_ptr, camera, T_C_L,
                                  block_size, max_integration_distance, &u_px,
                                  &voxel_depth_m, &p_voxel_center_C)) {
    return;
  }

  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DLinear<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > op->truncation_distance_m_) {
    return;
  }

  Color image_value;
  if (!interpolation::interpolate2DLinear<
          Color, interpolation::checkers::ColorPixelAlphaGreaterThanZero>(
          color_image, u_px, color_rows, color_cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  ColorVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  (*op)(surface_depth_m, voxel_depth_m, image_value, voxel_ptr);
}

/*****************************************************************************
 * Public interfaces
 ******************************************************************************/

// Camera
template <typename VoxelType>
template <typename UpdateFunctor>
void ProjectiveIntegrator<VoxelType>::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    UpdateFunctor* op, VoxelBlockLayer<VoxelType>* layer,
    std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate<Camera, UpdateFunctor>(
      depth_frame, ColorImage(MemoryType::kDevice), T_L_C, camera, op, layer,
      updated_blocks);
}

// Lidar
template <typename VoxelType>
template <typename UpdateFunctor>
void ProjectiveIntegrator<VoxelType>::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    UpdateFunctor* op, VoxelBlockLayer<VoxelType>* layer,
    std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate<Lidar, UpdateFunctor>(
      depth_frame, ColorImage(MemoryType::kDevice), T_L_C, lidar, op, layer,
      updated_blocks);
}

/*****************************************************************************
 * Templated, common integrate frame function
 * This function is shared between
 * - Camera/Lidar
 * - Occupancy/TSDF
 * - BUT color is in it's own file, because we haven't unified it yet.
 ******************************************************************************/

template <typename VoxelType>
template <typename SensorType, typename UpdateFunctor>
void ProjectiveIntegrator<VoxelType>::integrateFrameTemplate(
    const DepthImage& depth_frame, const ColorImage& color_frame,
    const Transform& T_L_C, const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer_ptr,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(layer_ptr);
  CHECK_NOTNULL(op);
  using BlockType = VoxelBlock<VoxelType>;
  if (!integrator_name_initialized_) {
    integrator_name_ = getIntegratorName();
  }

  timing::Timer integration_timer(integrator_name_ + "/integrate");

  // Identify blocks we can (potentially) see
  timing::Timer blocks_in_view_timer(integrator_name_ +
                                     "/integrate/get_blocks_in_view");
  const float max_integration_distance_behind_surface_m =
      truncation_distance_vox_ * layer_ptr->voxel_size();
  const std::vector<Index3D> block_indices =
      view_calculator_.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, sensor, layer_ptr->block_size(),
          max_integration_distance_behind_surface_m,
          max_integration_distance_m_);
  blocks_in_view_timer.Stop();

  // Return if we don't see anything
  if (block_indices.empty()) {
    return;
  }

  // Allocate blocks (CPU)
  timing::Timer allocate_blocks_timer(integrator_name_ +
                                      "/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, layer_ptr, *cuda_stream_);
  allocate_blocks_timer.Stop();

  // Move blocks to GPU for update
  timing::Timer transfer_blocks_timer(integrator_name_ +
                                      "/integrate/transfer_blocks");
  transferBlockPointersToDevice<BlockType>(block_indices, *cuda_stream_,
                                           layer_ptr, &block_ptrs_host_,
                                           &block_ptrs_device_);
  transferBlocksIndicesToDevice(block_indices, *cuda_stream_,
                                &block_indices_host_, &block_indices_device_);
  transfer_blocks_timer.Stop();

  // Update identified blocks
  timing::Timer update_blocks_timer(integrator_name_ +
                                    "/integrate/update_blocks");
  const Transform T_C_L = T_L_C.inverse();
  integrateBlocks(depth_frame, color_frame, T_C_L, sensor, op, layer_ptr);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

/*****************************************************************************
 * Integrate block functions
 *
 * These small functions call the kernels for the specifc sensor type
 ******************************************************************************/

// Camera
template <typename VoxelType>
template <typename UpdateFunctor>
void ProjectiveIntegrator<VoxelType>::integrateBlocks(
    const DepthImage& depth_frame, const ColorImage&, /*unused*/
    const Transform& T_C_L, const Camera& camera, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer_ptr) {
  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<<<num_thread_blocks, num_threads, 0,
                          *cuda_stream_>>>(
      block_indices_device_.data(),  // NOLINT
      camera,                        // NOLINT
      depth_frame.dataConstPtr(),    // NOLINT
      depth_frame.rows(),            // NOLINT
      depth_frame.cols(),            // NOLINT
      T_C_L,                         // NOLINT
      layer_ptr->block_size(),       // NOLINT
      max_integration_distance_m_,   // NOLINT
      op,                            // NOLINT
      block_ptrs_device_.data());    // NOLINT
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

// Lidar
template <typename VoxelType>
template <typename UpdateFunctor>
void ProjectiveIntegrator<VoxelType>::integrateBlocks(
    const DepthImage& depth_frame, const ColorImage&, /*unused*/
    const Transform& T_C_L, const Lidar& lidar, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer_ptr) {
  // Metric params - LiDAR specific
  const float voxel_size = layer_ptr->voxel_size();
  const float linear_interpolation_max_allowable_difference_m =
      lidar_linear_interpolation_max_allowable_difference_vox_ * voxel_size;
  const float nearest_interpolation_max_allowable_squared_dist_to_ray_m =
      std::pow(lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ *
                   voxel_size,
               2);

  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<<<num_thread_blocks, num_threads, 0,
                          *cuda_stream_>>>(
      block_indices_device_.data(),                               // NOLINT
      lidar,                                                      // NOLINT
      depth_frame.dataConstPtr(),                                 // NOLINT
      depth_frame.rows(),                                         // NOLINT
      depth_frame.cols(),                                         // NOLINT
      T_C_L,                                                      // NOLINT
      layer_ptr->block_size(),                                    // NOLINT
      max_integration_distance_m_,                                // NOLINT
      linear_interpolation_max_allowable_difference_m,            // NOLINT
      nearest_interpolation_max_allowable_squared_dist_to_ray_m,  // NOLINT
      op,                                                         // NOLINT
      block_ptrs_device_.data());                                 // NOLINT
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

// Specialization for color integration which use both depth and color
// to update a color voxel layer. The color version of
// integrateBlocksKernel is called within.
template <>
template <typename UpdateFunctor>
void ProjectiveIntegrator<ColorVoxel>::integrateBlocks(
    const DepthImage& depth_frame, const ColorImage& color_frame,
    const Transform& T_C_L, const Camera& camera, UpdateFunctor* op,
    VoxelBlockLayer<ColorVoxel>* layer_ptr) {
  // Let the kernel know that we've subsampled - Color specific
  const int depth_subsampling_factor = color_frame.rows() / depth_frame.rows();

  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<<<num_thread_blocks, num_threads, 0,
                          *cuda_stream_>>>(
      block_indices_device_.data(),  // NOLINT
      camera,                        // NOLINT
      color_frame.dataConstPtr(),    // NOLINT
      color_frame.rows(),            // NOLINT
      color_frame.cols(),            // NOLINT
      depth_frame.dataConstPtr(),    // NOLINT
      depth_frame.rows(),            // NOLINT
      depth_frame.cols(),            // NOLINT
      T_C_L,                         // NOLINT
      layer_ptr->block_size(),       // NOLINT
      max_integration_distance_m_,   // NOLINT
      depth_subsampling_factor,      // NOLINT
      op,                            // NOLINT
      block_ptrs_device_.data());    // NOLINT

  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

__device__ inline void setUnobservedVoxel(const TsdfVoxel& voxel_value,
                                          TsdfVoxel* voxel_ptr) {
  constexpr float kMinObservedWeight = 0.001;
  if (voxel_ptr->weight < kMinObservedWeight) {
    *voxel_ptr = voxel_value;
  }
}

__device__ inline void setUnobservedVoxel(const OccupancyVoxel& voxel_value,
                                          OccupancyVoxel* voxel_ptr) {
  constexpr float kEps = 1e-4;
  constexpr float kLogOddsUnobserved = 0;
  if (fabsf(voxel_ptr->log_odds - kLogOddsUnobserved) < kEps) {
    *voxel_ptr = voxel_value;
  }
}

// Call with:
// - One threadBlock per VoxelBlock
// - 8x8x8 threads per threadBlock
template <typename VoxelType>
__global__ void setUnobservedVoxelsKernel(const VoxelType voxel_value,
                                          VoxelBlock<VoxelType>** block_ptrs) {
  // Get the voxel addressed by this thread.
  VoxelBlock<VoxelType>* block = block_ptrs[blockIdx.x];
  VoxelType* block_voxel =
      &block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  // Call for the voxel type.
  setUnobservedVoxel(voxel_value, block_voxel);
}

template <typename VoxelType>
void ProjectiveIntegrator<VoxelType>::markUnobservedFreeInsideRadiusTemplate(
    const Vector3f& center, float radius, VoxelBlockLayer<VoxelType>* layer,
    std::vector<Index3D>* updated_blocks_ptr) {
  CHECK_NOTNULL(layer);
  CHECK_GT(radius, 0.0f);
  // First get blocks in AABB
  const Vector3f min = center.array() - radius;
  const Vector3f max = center.array() + radius;
  const AxisAlignedBoundingBox aabb(min, max);
  const std::vector<Index3D> blocks_touched_by_aabb =
      getBlockIndicesTouchedByBoundingBox(layer->block_size(), aabb);
  // Narrow to radius
  const std::vector<Index3D> blocks_inside_radius = getBlocksWithinRadius(
      blocks_touched_by_aabb, layer->block_size(), center, radius);
  // Allocate (if they're not already);
  std::for_each(blocks_inside_radius.begin(), blocks_inside_radius.end(),
                [layer, this](const Index3D& idx) {
                  layer->allocateBlockAtIndexAsync(idx, *cuda_stream_);
                });

  // VoxelBlock<VoxelType> pointers to GPU
  const std::vector<VoxelBlock<VoxelType>*> block_ptrs_host =
      getBlockPtrsFromIndices(blocks_inside_radius, layer);
  device_vector<VoxelBlock<VoxelType>*> block_ptrs_device;
  block_ptrs_device.copyFromAsync(block_ptrs_host, *cuda_stream_);

  // The value given to "observed" voxels
  VoxelType slightly_observed_voxel;
  if constexpr (std::is_same<TsdfVoxel, VoxelType>::value) {
    constexpr float kSlightlyObservedVoxelWeight = 0.1;
    slightly_observed_voxel.distance =
        get_truncation_distance_m(layer->voxel_size());
    slightly_observed_voxel.weight = kSlightlyObservedVoxelWeight;
  } else if (std::is_same<OccupancyVoxel, VoxelType>::value) {
    constexpr float kSlightlyObservedVoxelLogOdds = -2e-4;
    slightly_observed_voxel.log_odds = kSlightlyObservedVoxelLogOdds;
  }

  // Kernel launch
  const int num_thread_blocks = block_ptrs_device.size();
  constexpr int kVoxelsPerSide = VoxelBlock<VoxelType>::kVoxelsPerSide;
  const dim3 num_threads_per_block(kVoxelsPerSide, kVoxelsPerSide,
                                   kVoxelsPerSide);
  setUnobservedVoxelsKernel<<<num_thread_blocks, num_threads_per_block, 0,
                              *cuda_stream_>>>(slightly_observed_voxel,
                                               block_ptrs_device.data());
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Return blocks affected
  if (updated_blocks_ptr != nullptr) {
    *updated_blocks_ptr = blocks_inside_radius;
  }
}

}  // namespace nvblox
