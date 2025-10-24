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
#include <nvblox/integrators/tsdf_decay_integrator.h>
#include <nvblox/integrators/internal/cuda/impl/tsdf_decay_integrator_impl.cuh>

#include "nvblox/integrators/internal/cuda/impl/decayer_impl.cuh"

namespace nvblox {

float TsdfDecayIntegrator::decay_factor() const { return decay_factor_; }

void TsdfDecayIntegrator::decay_factor(const float value) {
  CHECK_GT(value, 0.0);
  CHECK_LT(value, 1.0);
  decay_factor_ = value;
}

float TsdfDecayIntegrator::decayed_weight_threshold() const {
  return decayed_weight_threshold_;
};

void TsdfDecayIntegrator::decayed_weight_threshold(
    const float decayed_weight_threshold) {
  CHECK_GE(decayed_weight_threshold, 0.f);
  decayed_weight_threshold_ = decayed_weight_threshold;
};

bool TsdfDecayIntegrator::set_free_distance_on_decayed() const {
  return set_free_distance_on_decayed_;
}

void TsdfDecayIntegrator::set_free_distance_on_decayed(
    const bool set_free_distance_on_decayed) {
  set_free_distance_on_decayed_ = set_free_distance_on_decayed;
}

float TsdfDecayIntegrator::free_distance_vox() const {
  return free_distance_vox_;
}

void TsdfDecayIntegrator::free_distance_vox(const float free_distance_vox) {
  CHECK_GT(free_distance_vox, 0.f);
  free_distance_vox_ = free_distance_vox;
}

parameters::ParameterTreeNode TsdfDecayIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "tsdf_decay_integrator" : name_remap;
  return ParameterTreeNode(
      name,
      {ParameterTreeNode("decay_factor:", decay_factor_),
       ParameterTreeNode("decayed_weight_theshold:", decayed_weight_threshold_),
       ParameterTreeNode("set_free_distance_on_decayed:",
                         set_free_distance_on_decayed_),
       ParameterTreeNode("free_distance_vox:", free_distance_vox_),
       DecayIntegratorBase::getParameterTree()});
}

}  // namespace nvblox
