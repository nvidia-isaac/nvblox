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
#include <nvblox/integrators/projective_occupancy_integrator.h>
#include <nvblox/integrators/internal/cuda/impl/projective_occupancy_integrator_impl.cuh>

#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/occupancy_integrator_params.h"

namespace nvblox {

ProjectiveOccupancyIntegrator::ProjectiveOccupancyIntegrator()
    : ProjectiveOccupancyIntegrator(std::make_shared<CudaStreamOwning>()) {}

ProjectiveOccupancyIntegrator::ProjectiveOccupancyIntegrator(
    const std::shared_ptr<CudaStream> cuda_stream)
    : ProjectiveIntegrator<OccupancyVoxel>(cuda_stream),
      cuda_stream_(cuda_stream) {
  update_functor_host_ptr_ =
      make_unified<UpdateOccupancyVoxelFunctor>(MemoryType::kHost);
}

ProjectiveOccupancyIntegrator::~ProjectiveOccupancyIntegrator() {
  // NOTE(alexmillane): We can't default this in the header file because to the
  // unified_ptr to a forward declared type. The type has to be defined where
  // the destructor is.
}

void ProjectiveOccupancyIntegrator::setFunctorParameters(
    const float voxel_size) {
  update_functor_host_ptr_->free_region_log_odds_ = free_region_log_odds_;
  update_functor_host_ptr_->occupied_region_log_odds_ =
      occupied_region_log_odds_;
  update_functor_host_ptr_->unobserved_region_log_odds_ =
      unobserved_region_log_odds_;
  update_functor_host_ptr_->occupied_region_half_width_m_ =
      occupied_region_half_width_m_;

  // Make sure all blocks that are considered
  // occupied by the sensor model are updated.
  if (get_truncation_distance_m(voxel_size) <
      update_functor_host_ptr_->occupied_region_half_width_m_) {
    const float new_truncation_distance_vox =
        update_functor_host_ptr_->occupied_region_half_width_m_ / voxel_size;
    LOG(WARNING)
        << "Truncation distance of the occupancy integrator is smaller than "
           "the occupied_region_half_width_m of the sensor model."
           "\nIncreasing truncation distance to "
        << new_truncation_distance_vox << " voxels.";
    truncation_distance_vox(new_truncation_distance_vox);
  }
}

float ProjectiveOccupancyIntegrator::free_region_occupancy_probability() const {
  return probabilityFromLogOdds(free_region_log_odds_);
}

void ProjectiveOccupancyIntegrator::free_region_occupancy_probability(
    float value) {
  CHECK(value >= 0.f && value <= 1.f) << "Probability must be in [0, 1].";
  free_region_log_odds_ = logOddsFromProbability(value);
}

float ProjectiveOccupancyIntegrator::occupied_region_occupancy_probability()
    const {
  return probabilityFromLogOdds(occupied_region_log_odds_);
}

void ProjectiveOccupancyIntegrator::occupied_region_occupancy_probability(
    float value) {
  CHECK(value >= 0.f && value <= 1.f) << "Probability must be in [0, 1].";
  occupied_region_log_odds_ = logOddsFromProbability(value);
}

float ProjectiveOccupancyIntegrator::unobserved_region_occupancy_probability()
    const {
  return probabilityFromLogOdds(unobserved_region_log_odds_);
}

void ProjectiveOccupancyIntegrator::unobserved_region_occupancy_probability(
    float value) {
  CHECK(value >= 0.f && value <= 1.f) << "Probability must be in [0, 1].";
  unobserved_region_log_odds_ = logOddsFromProbability(value);
}

float ProjectiveOccupancyIntegrator::occupied_region_half_width_m() const {
  return occupied_region_half_width_m_;
}

void ProjectiveOccupancyIntegrator::occupied_region_half_width_m(
    float occupied_region_half_width_m) {
  occupied_region_half_width_m_ = occupied_region_half_width_m;
}

std::string ProjectiveOccupancyIntegrator::getIntegratorName() const {
  return "occupancy";
}

void ProjectiveOccupancyIntegrator::markUnobservedFreeInsideRadius(
    const Vector3f& center, float radius, OccupancyLayer* layer,
    std::vector<Index3D>* updated_blocks_ptr) {
  markUnobservedFreeInsideRadiusTemplate(center, radius, layer,
                                         updated_blocks_ptr);
}

parameters::ParameterTreeNode ProjectiveOccupancyIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "projective_occupancy_integrator" : name_remap;
  return ParameterTreeNode(
      name,
      {
          ParameterTreeNode("free_region_log_odds:", free_region_log_odds_),
          ParameterTreeNode("occupied_region_log_odds:",
                            occupied_region_log_odds_),
          ParameterTreeNode("unobserved_region_log_odds:",
                            unobserved_region_log_odds_),
          ParameterTreeNode("occupied_region_half_width_m:",
                            occupied_region_half_width_m_),
          ProjectiveIntegrator<OccupancyVoxel>::getParameterTree(),
      });
}

}  // namespace nvblox
