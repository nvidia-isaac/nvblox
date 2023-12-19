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
#include <nvblox/integrators/occupancy_decay_integrator.h>

#include "nvblox/integrators/internal/cuda/impl/decayer_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"

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
  /// @return True if the voxel is fully decayed
  __device__ void operator()(OccupancyVoxel* voxel_ptr) const {
    // If fully decayed, set to decay-to probility
    if (isFullyDecayed(voxel_ptr)) {
      // This voxel decayed to zero log odds (0.5 occupancy probability).
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

std::vector<Index3D> OccupancyDecayIntegrator::decay(
    OccupancyLayer* layer_ptr, const CudaStream cuda_stream) {
  return decay(layer_ptr, {}, {}, cuda_stream);
}

std::vector<Index3D> OccupancyDecayIntegrator::decay(
    OccupancyLayer* layer_ptr,
    const DecayBlockExclusionOptions& block_exclusion_options,
    const CudaStream cuda_stream) {
  return decay(layer_ptr, block_exclusion_options, {}, cuda_stream);
}

std::vector<Index3D> OccupancyDecayIntegrator::decay(
    OccupancyLayer* layer_ptr,
    const DecayViewExclusionOptions& view_exclusion_options,
    const CudaStream cuda_stream) {
  return decay(layer_ptr, {}, view_exclusion_options, cuda_stream);
}

std::vector<Index3D> OccupancyDecayIntegrator::decay(
    OccupancyLayer* layer_ptr,
    const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
    const std::optional<DecayViewExclusionOptions>& view_exclusion_options,
    const CudaStream cuda_stream) {
  // Build the functor which decays a single voxel.
  OccupancyDecayFunctor voxel_decayer(free_space_decay_log_odds_,
                                      occupied_space_decay_log_odds_,
                                      decay_to_log_odds_);
  // Run it on all voxels
  return decayer_.decay(layer_ptr, voxel_decayer, deallocate_decayed_blocks_,
                        block_exclusion_options, view_exclusion_options,
                        cuda_stream);
}

OccupancyDecayIntegrator::OccupancyDecayIntegrator(DecayMode decay_mode)
    : DecayIntegratorBase(decay_mode) {
  if (decay_mode == DecayMode::kDecayToDeallocate) {
    decay_to_probability(kDefaultProbabilityDeallocate);
  } else if (decay_mode == DecayMode::kDecayToFree) {
    // NOTE(alexmillane): When we decay to free we decay to a probability
    // slightly lower than 0.5 (see default value). Note that if you want blocks
    // to be free in the ESDF, this will have to be less than the occupied
    // threshold in the ESDF integrator (which is 0.5 by default).
    decay_to_probability(kDefaultProbabilityFree);
  } else {
    LOG(FATAL) << "Decay mode not implemented";
  }
}

float OccupancyDecayIntegrator::free_region_decay_probability() const {
  return probabilityFromLogOdds(free_space_decay_log_odds_);
}
void OccupancyDecayIntegrator::free_region_decay_probability(float value) {
  CHECK(value > 0.5f && value <= 1.f)
      << "The free_region_decay_probability must be in [0.5, "
         "1.0] for the free region to decay towards 0.5 occupancy probability.";
  free_space_decay_log_odds_ = logOddsFromProbability(value);
}

float OccupancyDecayIntegrator::occupied_region_decay_probability() const {
  return probabilityFromLogOdds(occupied_space_decay_log_odds_);
}
void OccupancyDecayIntegrator::occupied_region_decay_probability(float value) {
  CHECK(value >= 0.f && value < 0.5f)
      << "The occupied_region_decay_probability must be in [0.0, "
         "0.5] for the occupied region to decay towards 0.5 occupancy "
         "probability.";
  occupied_space_decay_log_odds_ = logOddsFromProbability(value);
}

float OccupancyDecayIntegrator::decay_to_probability() const {
  return probabilityFromLogOdds(decay_to_log_odds_);
}

void OccupancyDecayIntegrator::decay_to_probability(float value) {
  CHECK(value >= 0.f && value <= 1.0f)
      << "The decay-to probility needs to be a valid probability (ie lying "
         "between [0.0, 1.0].)";
  decay_to_log_odds_ = logOddsFromProbability(value);
}

parameters::ParameterTreeNode OccupancyDecayIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "occupancy_decay_integrator" : name_remap;
  return ParameterTreeNode(
      name, {ParameterTreeNode("free_space_decay_log_odds:",
                               free_space_decay_log_odds_),
             ParameterTreeNode("occupied_space_decay_log_odds:",
                               occupied_space_decay_log_odds_),
             ParameterTreeNode("decay_to_log_odds_:", decay_to_log_odds_),
             DecayIntegratorBase::getParameterTree()});
}

}  // namespace nvblox
