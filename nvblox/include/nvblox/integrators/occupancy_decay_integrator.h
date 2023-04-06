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

#include "nvblox/core/log_odds.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// The OccupancyDecayIntegrator class can be used to decay (approach 0.5
/// occupancy probability) an occupancy layer.
/// This is useful for dynamic scenarios, where object might move/disappear
/// when not constantly observed.
class OccupancyDecayIntegrator {
 public:
  OccupancyDecayIntegrator();
  ~OccupancyDecayIntegrator();

  /// Decay the occupancy grid and deallocate fully decayed blocks (if
  /// deallocate_decayed_blocks_ is true);
  void decay(OccupancyLayer* layer_ptr);

  /// A parameter getter
  /// The flag that controls if fully decayed block should be deallocated or
  /// not.
  /// @returns the deallocate_decayed_blocks flag
  bool deallocate_decayed_blocks();

  /// A parameter setter
  /// See deallocate_decayed_blocks().
  /// @param deallocate_decayed_blocks the new flag.
  void deallocate_decayed_blocks(bool deallocate_decayed_blocks);

  /// A parameter getter
  /// The decay probability that is applied to the free region on decay.
  /// @returns the free region decay probability
  float free_region_decay_probability();

  /// A parameter setter
  /// See free_region_decay_probability().
  /// @param value the new free region decay probability.
  void free_region_decay_probability(float value);

  /// A parameter getter
  /// The decay probability that is applied to the occupied region on decay.
  /// @returns the occupied region decay probability
  float occupied_region_decay_probability();

  /// A parameter setter
  /// See occupied_region_decay_probability().
  /// @param value the new occupied region decay probability.
  void occupied_region_decay_probability(float value);

 private:
  // Member functions called on the decay step
  void decayProbability(OccupancyLayer* layer_ptr);
  void deallocateFullyDecayedBlocks(OccupancyLayer* layer_ptr);

  // Parameter for the decay step
  bool deallocate_decayed_blocks_ = true;
  float free_space_decay_log_odds_ = logOddsFromProbability(0.55);
  float occupied_space_decay_log_odds_ = logOddsFromProbability(0.4);

  // Internal buffers
  host_vector<OccupancyBlock*> allocated_block_ptrs_host_;
  device_vector<OccupancyBlock*> allocated_block_ptrs_device_;
  device_vector<bool> block_fully_decayed_device_;
  host_vector<bool> block_fully_decayed_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox
