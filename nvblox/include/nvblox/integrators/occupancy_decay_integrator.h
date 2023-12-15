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

#include <memory>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/log_odds.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/integrators/internal/decay_integrator.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// The OccupancyDecayIntegrator class can be used to decay (approach 0.5
/// occupancy probability) an occupancy layer.
class OccupancyDecayIntegrator : public DecayIntegratorBase<OccupancyLayer> {
 public:
  static constexpr float kDefaultFreeRegionDecayProbability = 0.55;
  static constexpr float kDefaultOccupiedRegionDecayProbability = 0.4;
  static constexpr float kDefaultProbabilityDeallocate = 0.5;
  static constexpr float kDefaultProbabilityFree = 0.49;


  explicit OccupancyDecayIntegrator(DecayMode decay_mode = kDefaultDecayMode);
  virtual ~OccupancyDecayIntegrator() = default;

  OccupancyDecayIntegrator(const OccupancyDecayIntegrator&) = delete;
  OccupancyDecayIntegrator& operator=(const OccupancyDecayIntegrator&) const =
      delete;
  OccupancyDecayIntegrator(OccupancyDecayIntegrator&&) = delete;
  OccupancyDecayIntegrator& operator=(const OccupancyDecayIntegrator&&) const =
      delete;

  /// A parameter getter
  /// The decay probability that is applied to the free region on decay.
  /// @returns the free region decay probability
  float free_region_decay_probability() const;

  /// A parameter setter
  /// See free_region_decay_probability().
  /// @param value the new free region decay probability.
  void free_region_decay_probability(float value);

  /// A parameter getter
  /// The decay probability that is applied to the occupied region on decay.
  /// @returns the occupied region decay probability
  float occupied_region_decay_probability() const;

  /// A parameter setter
  /// See occupied_region_decay_probability().
  /// @param value the new occupied region decay probability.
  void occupied_region_decay_probability(float value);

  /// A parameter getter
  /// @return The probability that the integrator decays each voxel to.
  float decay_to_probability() const;

  /// A parameter setter
  /// @param value The probability value that the integrator decays each voxel
  /// to.
  void decay_to_probability(float value);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  // Member functions called on the decay step
  void decayImplementationAsync(OccupancyLayer* layer_ptr,
                                const CudaStream cuda_stream) override;

  // Parameter for the decay step (these control the rate of decay)
  float free_space_decay_log_odds_ =
      logOddsFromProbability(kDefaultFreeRegionDecayProbability);
  float occupied_space_decay_log_odds_ =
      logOddsFromProbability(kDefaultOccupiedRegionDecayProbability);

  // Decay-to point. The probability value that we decay to. For example, we
  // could decay to 0.5 probability (unknown), or something like ~0.4 (slightly
  // free).
  float decay_to_log_odds_ =
      logOddsFromProbability(kDefaultProbabilityDeallocate);
};

}  // namespace nvblox
