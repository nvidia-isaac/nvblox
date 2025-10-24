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

#include "nvblox/integrators/internal/cuda/impl/decayer_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {

struct OccupancyDecayFunctor {
  __host__ __device__ OccupancyDecayFunctor(float free_space_decay_log_odds,
                                            float occupied_space_decay_log_odds,
                                            float decay_to_log_odds)
      : free_space_decay_log_odds_(free_space_decay_log_odds),
        occupied_space_decay_log_odds_(occupied_space_decay_log_odds),
        decay_to_log_odds_(decay_to_log_odds) {}
  __host__ __device__ ~OccupancyDecayFunctor() = default;

  /// Return true if the passed voxel is fully decayed
  /// @param voxel_ptr The voxel to check
  /// @return True if fully decayed
  __device__ bool isFullyDecayed(OccupancyVoxel* voxel_ptr) const {
    // Check if the next decay step would pass the threshold value in either
    // direction.
    const float log_odds = voxel_ptr->log_odds;
    if (log_odds >= decay_to_log_odds_) {
      return log_odds + occupied_space_decay_log_odds_ < decay_to_log_odds_;
    } else {
      return log_odds + free_space_decay_log_odds_ >= decay_to_log_odds_;
    }
  }

  /// Decays a single Occupancy voxel.
  /// @param voxel_ptr voxel to decay
  __device__ void operator()(OccupancyVoxel* voxel_ptr) const {
    // If fully decayed, set to decay-to probability
    if (isFullyDecayed(voxel_ptr)) {
      voxel_ptr->log_odds = decay_to_log_odds_;
      return;
    }

    // Else decay
    if (voxel_ptr->log_odds >= 0) {
      voxel_ptr->log_odds += occupied_space_decay_log_odds_;
    } else {
      voxel_ptr->log_odds += free_space_decay_log_odds_;
    }
  }

 protected:
  // Params
  float free_space_decay_log_odds_;
  float occupied_space_decay_log_odds_;
  float decay_to_log_odds_;
};

template <typename SensorType>
std::vector<Index3D> OccupancyDecayIntegrator::decay(
    OccupancyLayer* layer_ptr,
    const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
    const std::optional<DepthObservationSpace<SensorType>>&
        view_exclusion_options,
    const CudaStream& cuda_stream) {
  // Build the functor which decays a single voxel.
  OccupancyDecayFunctor voxel_decayer(free_space_decay_log_odds_,
                                      occupied_space_decay_log_odds_,
                                      decay_to_log_odds_);
  // Run it on all voxels
  return decayer_.decay(layer_ptr, voxel_decayer, deallocate_decayed_blocks_,
                        block_exclusion_options, view_exclusion_options,
                        cuda_stream);
}

}  // namespace nvblox
