/*
Copyright 2022-2024 NVIDIA CORPORATION

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
#include <nvblox/integrators/internal/cuda/impl/occupancy_decay_integrator_impl.cuh>

#include "nvblox/integrators/internal/cuda/impl/decayer_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

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

void OccupancyDecayIntegrator::decay_to_free(bool decay_to_free) {
  if (decay_to_free) {
    // NOTE(alexmillane): When we decay to free we decay to a probability
    // slightly lower than 0.5 (see default value). Note that if you want blocks
    // to be free in the ESDF, this will have to be less than the occupied
    // threshold in the ESDF integrator (which is 0.5 by default).
    decay_to_probability(kDefaultProbabilityFree);
  } else {
    // NOTE(remos): When we do not decay to free we decay to 0.5 which means
    // unkown occupancy.
    decay_to_probability(kDefaultProbabilityUnknown);
  }
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
