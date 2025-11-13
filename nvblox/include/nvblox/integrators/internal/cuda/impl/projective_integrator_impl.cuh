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
#include "nvblox/sensors/sensor.h"
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
 * Kernels.
 * Not that in order to optimize resource usage, we use __launch_bounds__ to
 * specify an upper limit of threads.
 ******************************************************************************/

// Depth integration
template <typename VoxelType, typename UpdateFunctor, typename SensorType>
__global__ void __launch_bounds__(kMaxNumThreadsPerBlock<VoxelType>())
    integrateBlocksKernel(const Index3D* block_indices_device_ptr,
                          const SensorType sensor,
                          const MaskedDepthImageConstView image,
                          const Transform T_C_L, const float voxel_size,
                          const float block_size,
                          const float max_integration_distance,
                          UpdateFunctor* op,
                          VoxelBlock<VoxelType>** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  Index3D block_idx, voxel_idx;
  voxelAndBlockIndexFromCudaThreadIndex(block_indices_device_ptr, &block_idx,
                                        &voxel_idx);
  if (!projectThreadVoxel(block_idx, voxel_idx, sensor, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return;
  }

  // Interpolate on the image plane
  float image_value;
  Index2D pix_pos;

  if (!sensor.interpolateDepthImage(image, u_px, p_voxel_center_C, voxel_size,
                                    &image_value, &pix_pos)) {
    return;
  }

  // Handle invalid depth values
  if (!interpolation::checkers::PixelIsValidDepth::check(image_value)) {
    // Set all invalid depth values to zero.
    // Will not be integrated in Occupancy/TSDF layers.
    // The ray along this pixel can be decayed by invalid_depth_decay_factor in
    // the TSDF layer.
    image_value = 0.0f;
  }

  // Note that isMasked is always true if there is no mask attached to the
  // incoming image
  const bool is_active = image.isMasked(pix_pos.y(), pix_pos.x());

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  VoxelType* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  (*op)(image_value, voxel_depth_m, is_active, voxel_ptr);
}

// Appearance integration
template <typename UpdateFunctor, typename VoxelType, typename SensorType>
__global__ void __launch_bounds__(kMaxNumThreadsPerBlock<VoxelType>())
    integrateBlocksKernel(
        const Index3D* block_indices_device_ptr, const SensorType sensor,
        MaskedImageView<const typename VoxelType::ArrayType> appearance_image,
        DepthImageConstView depth_image, const Transform T_C_L,
        const float voxel_size, const float block_size,
        const float max_integration_distance, const int depth_subsample_factor,
        UpdateFunctor* op, VoxelBlock<VoxelType>** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  Index3D block_idx, voxel_idx;
  voxelAndBlockIndexFromCudaThreadIndex(block_indices_device_ptr, &block_idx,
                                        &voxel_idx);
  if (!projectThreadVoxel(block_idx, voxel_idx, sensor, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return;
  }

  // Interpolate depth. We use the method specified by the sensor
  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!sensor.interpolateDepthImage(depth_image, u_px_depth, p_voxel_center_C,
                                    voxel_size, &surface_depth_m)) {
    return;
  }

  // Skip appearance integration for invalid depth values
  if (!interpolation::checkers::PixelIsValidDepth::check(surface_depth_m)) {
    // For appearance integration, we only allow valid depth values to get the
    // distance to the surface.
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > op->truncation_distance_m_) {
    return;
  }

  // Interpolate in the appearance image. Here we always use linear
  // interpolation
  typename VoxelType::ArrayType image_value;
  if (!interpolation::interpolate2DLinear<typename VoxelType::ArrayType>(
          appearance_image, u_px, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  VoxelType* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Note that isMasked is always true if there is no mask attached to the
  // incoming image
  const bool is_active = appearance_image.isMasked(u_px.y(), u_px.x());

  // Update the voxel using the update rule for this layer type
  (*op)(surface_depth_m, voxel_depth_m, is_active, image_value, voxel_ptr);
}

/*****************************************************************************
 * Public interfaces
 ******************************************************************************/

// Depth integration
template <typename VoxelType>
template <typename UpdateFunctor, typename SensorType>
void ProjectiveIntegrator<VoxelType>::integrateFrame(
    const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,
    const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer, std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate<SensorType, UpdateFunctor>(
      depth_frame, MaskedColorImageConstView(), T_L_C, sensor, op, layer,
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
    const MaskedDepthImageConstView& depth_frame,
    const MaskedColorImageConstView& color_frame, const Transform& T_L_C,
    const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer_ptr,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(layer_ptr);
  CHECK_NOTNULL(op);

  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

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
  transferBlockPointersToDeviceAsync<BlockType>(
      block_indices, layer_ptr, &block_ptrs_host_, &block_ptrs_device_,
      *cuda_stream_);
  transferBlockIndicesToDeviceAsync(block_indices, &block_indices_host_,
                                    &block_indices_device_, *cuda_stream_);
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

// Depth integration
template <typename VoxelType>
template <typename UpdateFunctor, typename SensorType>
void ProjectiveIntegrator<VoxelType>::integrateBlocks(
    const MaskedDepthImageConstView& depth_frame,
    const MaskedColorImageConstView&, /*unused*/
    const Transform& T_C_L, const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<VoxelType>* layer_ptr) {
  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<<<num_thread_blocks, num_threads, 0,
                          *cuda_stream_>>>(
      block_indices_device_.data(),  // NOLINT
      sensor,                        // NOLINT
      depth_frame,                   // NOLINT
      T_C_L,                         // NOLINT
      layer_ptr->voxel_size(),       // NOLINT
      layer_ptr->block_size(),       // NOLINT
      max_integration_distance_m_,   // NOLINT
      op,                            // NOLINT
      block_ptrs_device_.data());    // NOLINT
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

// Specialization for color integration which use both depth and color
// to update a color voxel layer. The color version of
// integrateBlocksKernel is called within.
template <>
template <typename UpdateFunctor, typename SensorType>
void ProjectiveIntegrator<ColorVoxel>::integrateBlocks(
    const MaskedDepthImageConstView& depth_frame,
    const MaskedColorImageConstView& color_frame, const Transform& T_C_L,
    const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<ColorVoxel>* layer_ptr) {
  // Let the kernel know that we've subsampled - Color specific
  const int depth_subsampling_factor = color_frame.rows() / depth_frame.rows();

  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<UpdateFunctor, ColorVoxel>
      <<<num_thread_blocks, num_threads, 0,
         *cuda_stream_>>>(block_indices_device_.data(),  // NOLINT
                          sensor,                        // NOLINT
                          color_frame,                   // NOLINT
                          depth_frame,                   // NOLINT
                          T_C_L,                         // NOLINT
                          layer_ptr->voxel_size(),       // NOLINT
                          layer_ptr->block_size(),       // NOLINT
                          max_integration_distance_m_,   // NOLINT
                          depth_subsampling_factor,      // NOLINT
                          op,                            // NOLINT
                          block_ptrs_device_.data());    // NOLINT

  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

// Specialization for feature integration.
template <>
template <typename UpdateFunctor, typename SensorType>
void ProjectiveIntegrator<FeatureVoxel>::integrateBlocks<UpdateFunctor,
                                                         FeatureImage>(
    const MaskedDepthImageConstView& depth_frame,
    const MaskedFeatureImageConstView& feature_frame, const Transform& T_C_L,
    const SensorType& sensor, UpdateFunctor* op,
    VoxelBlockLayer<FeatureVoxel>* layer_ptr) {
  // Let the kernel know that we've subsampled - Feature specific
  const int depth_subsampling_factor =
      feature_frame.rows() / depth_frame.rows();

  // Kernel
  const auto [num_thread_blocks, num_threads] =
      getLaunchSizes(block_indices_device_.size());
  integrateBlocksKernel<UpdateFunctor, FeatureVoxel>
      <<<num_thread_blocks, num_threads, 0,
         *cuda_stream_>>>(block_indices_device_.data(),  // NOLINT
                          sensor,                        // NOLINT
                          feature_frame,                 // NOLINT
                          depth_frame,                   // NOLINT
                          T_C_L,                         // NOLINT
                          layer_ptr->voxel_size(),       // NOLINT
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
