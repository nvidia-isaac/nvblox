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
#include <nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh>

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

struct UpdateOccupancyVoxelFunctor {
  UpdateOccupancyVoxelFunctor() {}

  __device__ bool operator()(const float surface_depth_measured,
                             const float voxel_depth_m,
                             OccupancyVoxel* voxel_ptr) {
    // Get the update summand depending on the measured depth
    float log_odds_update;
    if (voxel_depth_m <
        surface_depth_measured - occupied_region_half_width_m_) {
      log_odds_update = free_region_log_odds_;
    } else if (voxel_depth_m <=
               surface_depth_measured + occupied_region_half_width_m_) {
      log_odds_update = occupied_region_log_odds_;
    } else {
      log_odds_update = unobserved_region_log_odds_;
    }

    // Update and clip
    float updated_log_odds = voxel_ptr->log_odds + log_odds_update;
    voxel_ptr->log_odds =
        fmax(kMinLogOdds_, fmin(updated_log_odds, kMaxLogOdds_));

    return true;
  }

  // Sensor model parameters
  float free_region_log_odds_ = logOddsFromProbability(0.3);
  float occupied_region_log_odds_ = logOddsFromProbability(0.7);
  float unobserved_region_log_odds_ = logOddsFromProbability(0.5);
  float occupied_region_half_width_m_ = 0.1;

  // Min and max values for clipping
  const float kMaxLogOdds_ = logOddsFromProbability(0.99);
  const float kMinLogOdds_ = logOddsFromProbability(0.01);
};

ProjectiveOccupancyIntegrator::ProjectiveOccupancyIntegrator()
    : ProjectiveIntegrator<OccupancyVoxel>() {
  update_functor_host_ptr_ =
      make_unified<UpdateOccupancyVoxelFunctor>(MemoryType::kHost);
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveOccupancyIntegrator::~ProjectiveOccupancyIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveOccupancyIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    OccupancyLayer* layer, std::vector<Index3D>* updated_blocks) {
  setFunctorParameters(layer->voxel_size());
  ProjectiveIntegrator<OccupancyVoxel>::integrateFrame(
      depth_frame, T_L_C, camera,
      update_functor_host_ptr_.clone(MemoryType::kDevice).get(), layer,
      updated_blocks);
}

void ProjectiveOccupancyIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    OccupancyLayer* layer, std::vector<Index3D>* updated_blocks) {
  setFunctorParameters(layer->voxel_size());
  ProjectiveIntegrator<OccupancyVoxel>::integrateFrame(
      depth_frame, T_L_C, lidar,
      update_functor_host_ptr_.clone(MemoryType::kDevice).get(), layer,
      updated_blocks);
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

}  // namespace nvblox
