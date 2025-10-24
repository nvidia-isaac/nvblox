/*
Copyright 2025 NVIDIA CORPORATION

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

#include <cuda_runtime.h>
#include "nvblox/core/types.h"
#include "nvblox/integrators/depth_observation_space.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/voxels.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

// Return true if the voxel is free, i.e. if these three critera are satisfied:
// * The corresponding TSDF voxel is active
// * Voxel must be initialized
// * Voxel have not been recently occupied
__device__ inline bool isVoxelFree(
    const FreespaceVoxel& freespace_voxel, const TsdfVoxel& tsdf_voxel,
    Time current_time_ms, Time min_duration_since_occupied_for_freespace_ms) {
  return tsdf_voxel.weight > 1e-6 &&
         (freespace_voxel.last_occupied_timestamp_ms != Time(0)) &&
         freespace_voxel.last_occupied_timestamp_ms <=
             current_time_ms - min_duration_since_occupied_for_freespace_ms;
}
// Return true if all voxels in a neighborhood are free.
__device__ inline bool isVoxelNeighborhoodFree(
    const Index3D& center_voxel_index, const VoxelBlock<uint8_t>& is_free_block,
    Time current_time_ms, Time min_duration_since_occupied_for_freespace_ms,
    const int neighborhood_size = 3) {
  bool neighborhood_is_free = true;

  NVBLOX_CHECK(neighborhood_size % 2 != 0, "Need odd neighborhood size");

  const int padding = (neighborhood_size - 1) / 2;

  // Go over all blocks in the neighborhood.
  for (int u = -padding; u <= padding; ++u) {
    for (int v = -padding; v <= padding; ++v) {
      for (int w = -padding; w <= padding; ++w) {
        const int x = center_voxel_index.x() + u;
        const int y = center_voxel_index.y() + v;
        const int z = center_voxel_index.z() + w;

        // Skip center voxel.
        if (x == center_voxel_index.x() && y == center_voxel_index.y() &&
            z == center_voxel_index.z()) {
          continue;
        }
        // Skip voxels out-of-bounds.
        if (x < 0 || x >= FreespaceBlock::kVoxelsPerSide || y < 0 ||
            y >= FreespaceBlock::kVoxelsPerSide || z < 0 ||
            z >= FreespaceBlock::kVoxelsPerSide) {
          continue;
        }

        // Check if free
        neighborhood_is_free &= is_free_block.voxels[x][y][z];
      }
    }
  }

  return neighborhood_is_free;
}

// Kernel for freespace update
// Expected launch parameters:
//   num_blocks: Number of voxelblocks in the freespace layer
//   num_threads_per_block: dim3(a, a, a) where a = voxels_per_side +
//   2*PaddingSize
// @tparam PaddingSize Number of padded voxels appended to each side of a block
// to  allow for lookup of neighboring blocks.
//  Should be set to "1" if check_neighborhood is used. Set to "0" otherwise for
//  improved performance
// NOTE(alexmillane): We faced an issue where in debug mode, this kernel blew
// the device register limits, and crashed on launch. We're therefore using
// __launch_bounds__ to inform the compiler of the maximum number of threads
// that it will be launched with such that it can respect the register limit in
// the worst-case thread number.
template <typename SensorType>
__global__ void __launch_bounds__(kMaxNumThreadsPerBlock<FreespaceVoxel>())
    updateFreespaceLayerKernel(
        const TsdfBlock** tsdf_blocks_to_update,
        const Index3D* block_indices_to_update, int num_block_indices_to_update,
        float voxel_size, float max_tsdf_distance_for_occupancy_m,
        Time max_unobserved_to_keep_consecutive_occupancy_ms,
        Time min_duration_since_occupied_for_freespace_ms,
        Time min_consecutive_occupancy_duration_for_reset_ms,
        bool check_neighborhood, bool initialize_to_high_confidence_freespace,
        Time last_update_time_ms, Time current_update_time_ms,
        const bool do_viewpoint_exclusion, const SensorType sensor,
        const Transform T_C_L, DepthImageConstView depth_image,
        float max_view_distance_m, float truncation_distance_m,
        float block_size_m, FreespaceBlock** freespace_blocks_to_update) {
  // This kernel implements the freespace update as described in the
  // dynablox paper (https://ieeexplore.ieee.org/document/10218983).
  //
  // It consist of the following steps:
  // - Initialization of freespace voxels if seen for the first time.
  // - Update the consecutive_occupancy_duration_ms field
  // - Update the last_occupied_timestamp_ms field
  // - Check if the voxel (and all its neighbors if check_neighborhood=true)
  //   is/are free
  // - Update the is_high_confidence_freespace field
  // Every ThreadBlock works on one VoxelBlock (blockIdx.y/z should be zero)
  //
  // Limitation: When performing neighborhood checks, only voxels in the current
  // block is taken into account

  NVBLOX_CHECK(blockIdx.x < num_block_indices_to_update, "Out of bounds");

  // Get block pointers in global mem
  const Index3D block_index = block_indices_to_update[blockIdx.x];
  const Index3D voxel_index(threadIdx.x, threadIdx.y, threadIdx.z);

  /// Initialize shared mem
  __shared__ VoxelBlock<uint8_t> is_free_block_sh;
  is_free_block_sh.voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()] =
      0;
  __syncthreads();

  FreespaceVoxel* freespace_voxel_ptr =
      &freespace_blocks_to_update[blockIdx.x]
           ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  FreespaceVoxel freespace_voxel = *freespace_voxel_ptr;

  const TsdfVoxel& tsdf_voxel =
      tsdf_blocks_to_update[blockIdx.x]
          ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

  // Only do the updates if we're in-view.
  bool update_voxel = true;
  if (do_viewpoint_exclusion) {
    NVBLOX_CHECK(depth_image.dataConstPtr() != nullptr, "");
    const bool in_view = doesVoxelHaveDepthMeasurement(
        block_index, voxel_index, sensor, depth_image, T_C_L,
        voxelSize<FreespaceBlock>(block_size_m), block_size_m,
        max_view_distance_m, truncation_distance_m);
    // If not in view, don't run updates.
    if (!in_view) {
      update_voxel = false;
    }
  }

  // Initialization of freespace
  bool initialize_freespace_voxel =
      freespace_voxel.last_occupied_timestamp_ms == Time(0);
  if (initialize_freespace_voxel) {
    // All voxels are initialized to being occupied
    freespace_voxel.last_occupied_timestamp_ms = current_update_time_ms;
    freespace_voxel.consecutive_occupancy_duration_ms = Time(0);
    freespace_voxel.is_high_confidence_freespace =
        initialize_to_high_confidence_freespace;
  }

  // NOTE(alexmillane): We don't want to run the rest of this function for
  // voxels out-of-view. However because of the remaining syncthreads we cannot
  // exit early for some threads, so we put the remaining operations in
  // conditionals.
  bool is_free;
  if (update_voxel && !initialize_freespace_voxel) {
    // Update consecutive occupancy duration
    // Note: We use the last_occupied_timestamp_ms from the last update here
    // to start counting the consecutive_occupancy_duration_ms from 0 ms when
    // a voxel was seen occupied. Dynablox Eq. (9)
    if (current_update_time_ms - freespace_voxel.last_occupied_timestamp_ms <=
        max_unobserved_to_keep_consecutive_occupancy_ms) {
      // Voxel was occupied lately
      freespace_voxel.consecutive_occupancy_duration_ms +=
          current_update_time_ms - last_update_time_ms;
    } else {
      // We haven't seen the voxel occupied for some time
      freespace_voxel.consecutive_occupancy_duration_ms = Time(0);
    }

    // Update the last occupied timestamp
    // Dynablox Eq. (8)
    if (tsdf_voxel.distance <= max_tsdf_distance_for_occupancy_m) {
      // We are close to a surface, let's assume the voxel is occupied
      freespace_voxel.last_occupied_timestamp_ms = current_update_time_ms;
    }

    // Check if the voxel is free and cache the result
    is_free = isVoxelFree(freespace_voxel, tsdf_voxel, current_update_time_ms,
                          min_duration_since_occupied_for_freespace_ms);
    is_free_block_sh.voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()] =
        is_free;
  }

  // Synchronize here because the last_occupied_timestamp_ms field of the
  // neighboring voxels could have been updated during this kernel. This is
  // strictly only necessary if check_neighborhood=true, but syncing inside an
  // if-statement is not recommended since it might lead to a deadlock if
  // threads are diverging.
  __syncthreads();

  // NOTE: See comment above about the reason for "update_voxel".
  if (update_voxel && !initialize_freespace_voxel) {
    // Check if neighbors are free as well
    // Dynablox Eq. (10) (neighborhood)
    if (check_neighborhood && is_free) {
      is_free &= isVoxelNeighborhoodFree(
          voxel_index, is_free_block_sh, current_update_time_ms,
          min_duration_since_occupied_for_freespace_ms);
    }

    // Update high confidence freespace
    // Dynablox Eq. (12)
    if (freespace_voxel.consecutive_occupancy_duration_ms >=
        min_consecutive_occupancy_duration_for_reset_ms) {
      // There was consecutive occupancy for some time: reset freespace
      freespace_voxel.is_high_confidence_freespace = false;
    } else {
      // Otherwise high confidence freespace is set if the voxel is free
      // and kept if it was high confidence before
      // Dynablox Eq. (11)
      freespace_voxel.is_high_confidence_freespace =
          freespace_voxel.is_high_confidence_freespace || is_free;
    }
  }

  // Copy back to global mem if the voxel was updated or initialized
  if (update_voxel || initialize_freespace_voxel) {
    *freespace_voxel_ptr = freespace_voxel;
  }
}

// This function just:
// - Returns a bool indicating if viewpoint exclusion should be run, and
// - breaks apart the viewpoint into types which can be passed to the kernel,
// and
// - if not requested returns default values.
template <typename SensorType>
auto get_viewpoint_or_defaults(
    const std::optional<DepthObservationSpace<SensorType>>& maybe_view) {
  bool do_viewpoint_exclusion = false;
  float truncation_distance_m = 0.F;
  float max_view_distance_m = 0.F;
  DepthImageConstView depth_image;
  SensorType sensor;
  Transform T_L_C;
  if (maybe_view.has_value() && maybe_view.value().depth_image.has_value()) {
    do_viewpoint_exclusion = true;
    const auto& view = maybe_view.value();
    T_L_C = view.T_L_C;
    sensor = view.sensor;
    max_view_distance_m =
        view.max_view_distance_m.value_or(std::numeric_limits<float>::max());
    truncation_distance_m =
        view.truncation_distance_m.value_or(std::numeric_limits<float>::max());
    depth_image = view.depth_image.value();
  } else if (maybe_view.has_value()) {
    LOG(WARNING) << "We only support viewpoint exclusion with a depth image.";
  }

  // Post-condition. Just make sure everything is valid
  if (do_viewpoint_exclusion) {
    CHECK_NOTNULL(depth_image.dataConstPtr());
    CHECK_GT(depth_image.rows(), 0);
    CHECK_GT(depth_image.cols(), 0);
    CHECK_GT(truncation_distance_m, 0);
    CHECK_GT(max_view_distance_m, 0);
  }
  return std::make_tuple(do_viewpoint_exclusion, T_L_C, sensor, depth_image,
                         max_view_distance_m, truncation_distance_m);
}

template <typename SensorType>
void FreespaceIntegrator::launchKernel(
    Time update_time_ms,
    const std::optional<DepthObservationSpace<SensorType>>& maybe_view,
    FreespaceLayer* freespace_layer_ptr) {
  const dim3 kThreadsPerBlock(TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide);
  const int num_thread_blocks = block_indices_to_update_device_.size();

  // Break-up the optional into parts for kernel, if viewpoint exclusion
  // requested (otherwise the first returned bool will be false).
  auto [do_viewpoint_exclusion, T_L_C, sensor, depth_image, max_view_distance_m,
        truncation_distance_m] = get_viewpoint_or_defaults(maybe_view);

  updateFreespaceLayerKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                               *cuda_stream_>>>(
      tsdf_blocks_to_update_device_.data(),                     // NOLINT
      block_indices_to_update_device_.data(),                   // NOLINT
      block_indices_to_update_device_.size(),                   // NOLINT
      freespace_layer_ptr->voxel_size(),                        // NOLINT
      max_tsdf_distance_for_occupancy_m_,                       // NOLINT
      max_unobserved_to_keep_consecutive_occupancy_ms_,         // NOLINT
      min_duration_since_occupied_for_freespace_ms_,            // NOLINT
      min_consecutive_occupancy_duration_for_reset_ms_,         // NOLINT
      check_neighborhood_,                                      // NOLINT
      initialize_to_high_confidence_freespace_,                 // NOLINT
      last_update_time_ms_,                                     // NOLINT
      update_time_ms,                                           // NOLINT
      do_viewpoint_exclusion,                                   // NOLINT
      sensor,                                                   // NOLINT
      T_L_C.inverse(),                                          // NOLINT
      depth_image, max_view_distance_m, truncation_distance_m,  // NOLINT
      freespace_layer_ptr->block_size(),                        // NOLINT
      freespace_blocks_to_update_device_.data());               // NOLINT
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename SensorType>
void FreespaceIntegrator::updateFreespaceLayer(
    const std::vector<Index3D>& block_indices_to_update, Time update_time_ms,
    const TsdfLayer& tsdf_layer,
    const std::optional<DepthObservationSpace<SensorType>>& view,
    FreespaceLayer* freespace_layer_ptr) {
  timing::Timer integration_timer("freespace/integrate");

  // Check inputs
  CHECK_NOTNULL(freespace_layer_ptr);
  CHECK(freespace_layer_ptr->voxel_size() - tsdf_layer.voxel_size() < 1e-4)
      << "Voxel size of tsdf and freespace layer must be equal.";
  if (block_indices_to_update.empty()) {
    return;
  }
  const size_t num_block_to_update = block_indices_to_update.size();

  // Allocate missing blocks
  timing::Timer allocate_timer("freespace/integrate/allocate");
  freespace_layer_ptr->allocateBlocksAtIndices(block_indices_to_update,
                                               *cuda_stream_);
  allocate_timer.Stop();

  // Expand the buffers when needed
  if (num_block_to_update > block_indices_to_update_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_block_to_update);
    block_indices_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    freespace_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    tsdf_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
  }

  timing::Timer transfer_timer("freespace/integrate/transfer_blocks");

  // Transfer block indices
  transferBlockIndicesToDeviceAsync(
      block_indices_to_update, &block_indices_to_update_host_,
      &block_indices_to_update_device_, *cuda_stream_);

  // Transfer freespace block pointers
  transferBlockPointersToDeviceAsync(
      block_indices_to_update, freespace_layer_ptr,
      &freespace_blocks_to_update_host_, &freespace_blocks_to_update_device_,
      *cuda_stream_);

  // Transfer tsdf block pointers
  transferBlockPointersToDeviceAsync<TsdfBlock>(
      block_indices_to_update, tsdf_layer, &tsdf_blocks_to_update_host_,
      &tsdf_blocks_to_update_device_, *cuda_stream_);
  transfer_timer.Stop();

  timing::Timer update_timer("freespace/integrate/update_blocks");
  launchKernel(update_time_ms, view, freespace_layer_ptr);

  cuda_stream_->synchronize();
  update_timer.Stop();

  last_update_time_ms_ = update_time_ms;
}

}  // namespace nvblox
