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

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

__device__ float inline voxelIsDecayed(float log_odds,
                                       float free_space_decay_log_odds,
                                       float occupied_space_decay_log_odds,
                                       float decay_to_log_odds) {
  // Check if the next decay step would pass the threshold value in either
  // direction.
  if (log_odds >= decay_to_log_odds) {
    return log_odds + occupied_space_decay_log_odds < decay_to_log_odds;
  } else {
    return log_odds + free_space_decay_log_odds >= decay_to_log_odds;
  }
}

__global__ void decayProbabilityKernel(OccupancyBlock** block_ptrs,
                                       float free_space_decay_log_odds,
                                       float occupied_space_decay_log_odds,
                                       float decay_to_log_odds,
                                       bool* is_block_fully_decayed) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_decayed[blockIdx.x] = true;
  }
  __syncthreads();

  OccupancyVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  if (voxelIsDecayed(voxel_ptr->log_odds, free_space_decay_log_odds,
                     occupied_space_decay_log_odds, decay_to_log_odds)) {
    // This voxel decayed to zero log odds (0.5 occupancy probability).
    voxel_ptr->log_odds = decay_to_log_odds;
    return;
  } else {
    // If one voxel in a block is not decayed, the block is not fully decayed.
    // NOTE: There could be more than one thread writing this value, but because
    // all of them write false it is no issue.
    is_block_fully_decayed[blockIdx.x] = false;
  }

  if (voxel_ptr->log_odds >= 0) {
    voxel_ptr->log_odds += occupied_space_decay_log_odds;
  } else {
    voxel_ptr->log_odds += free_space_decay_log_odds;
  }
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

void OccupancyDecayIntegrator::decayImplementationAsync(
    OccupancyLayer*, const CudaStream cuda_stream) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = allocated_block_ptrs_host_.size();
  decayProbabilityKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                           cuda_stream>>>(
      allocated_block_ptrs_device_.data(),  // NOLINT
      free_space_decay_log_odds_,           // NOLINT
      occupied_space_decay_log_odds_,       // NOLINT
      decay_to_log_odds_,                   // NOLINT
      block_fully_decayed_device_.data());  // NOLINT
  checkCudaErrors(cudaPeekAtLastError());
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
  return ParameterTreeNode(name,
                           {ParameterTreeNode("free_space_decay_log_odds:",
                                              free_space_decay_log_odds_),
                            ParameterTreeNode("occupied_space_decay_log_odds:",
                                              occupied_space_decay_log_odds_),
                            DecayIntegratorBase::getParameterTree()});
}

}  // namespace nvblox
