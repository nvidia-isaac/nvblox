/*
Copyright 2023 NVIDIA CORPORATION

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
#include <nvblox/integrators/freespace_integrator.h>
#include <nvblox/integrators/internal/cuda/impl/freespace_integrator_impl.cuh>

#include "nvblox/core/internal/cuda/device_function_utils.cuh"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"

namespace nvblox {

static_assert(TsdfBlock::kVoxelsPerSide == FreespaceBlock::kVoxelsPerSide,
              "Need same block dimensions for tsdf and freespace blocks");

FreespaceIntegrator::FreespaceIntegrator()
    : FreespaceIntegrator(std::make_shared<CudaStreamOwning>()) {}

FreespaceIntegrator::FreespaceIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

float FreespaceIntegrator::max_tsdf_distance_for_occupancy_m() const {
  return max_tsdf_distance_for_occupancy_m_;
}

void FreespaceIntegrator::max_tsdf_distance_for_occupancy_m(float value) {
  max_tsdf_distance_for_occupancy_m_ = value;
}

Time FreespaceIntegrator::max_unobserved_to_keep_consecutive_occupancy_ms()
    const {
  return max_unobserved_to_keep_consecutive_occupancy_ms_;
}

void FreespaceIntegrator::max_unobserved_to_keep_consecutive_occupancy_ms(
    Time value) {
  max_unobserved_to_keep_consecutive_occupancy_ms_ = value;
}

Time FreespaceIntegrator::min_duration_since_occupied_for_freespace_ms() const {
  return min_duration_since_occupied_for_freespace_ms_;
}

void FreespaceIntegrator::min_duration_since_occupied_for_freespace_ms(
    Time value) {
  min_duration_since_occupied_for_freespace_ms_ = value;
}

Time FreespaceIntegrator::min_consecutive_occupancy_duration_for_reset_ms()
    const {
  return min_consecutive_occupancy_duration_for_reset_ms_;
}

void FreespaceIntegrator::min_consecutive_occupancy_duration_for_reset_ms(
    Time value) {
  min_consecutive_occupancy_duration_for_reset_ms_ = value;
}

bool FreespaceIntegrator::check_neighborhood() const {
  return check_neighborhood_;
}

void FreespaceIntegrator::check_neighborhood(bool value) {
  check_neighborhood_ = value;
}

bool FreespaceIntegrator::initialize_to_high_confidence_freespace() const {
  return initialize_to_high_confidence_freespace_;
}

void FreespaceIntegrator::initialize_to_high_confidence_freespace(bool value) {
  initialize_to_high_confidence_freespace_ = value;
}

parameters::ParameterTreeNode FreespaceIntegrator::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "freespace_integrator" : name_remap;
  std::function<std::string(const Time&)> time_to_string = [](const Time& t) {
    return std::to_string(static_cast<int64_t>(t));
  };
  using parameters::ParameterTreeNode;
  return ParameterTreeNode(
      name,
      {
          ParameterTreeNode("max_tsdf_distance_for_occupancy_m:",
                            max_tsdf_distance_for_occupancy_m_),
          ParameterTreeNode("max_unobserved_to_keep_consecutive_occupancy_ms:",
                            max_unobserved_to_keep_consecutive_occupancy_ms_,
                            time_to_string),
          ParameterTreeNode("min_duration_since_occupied_for_freespace_ms:",
                            min_duration_since_occupied_for_freespace_ms_,
                            time_to_string),
          ParameterTreeNode("min_consecutive_occupancy_duration_for_reset_ms:",
                            min_consecutive_occupancy_duration_for_reset_ms_,
                            time_to_string),
          ParameterTreeNode("check_neighborhood:", check_neighborhood_),
          ParameterTreeNode("initialize_to_high_confidence_freespace:",
                            initialize_to_high_confidence_freespace_),
      });
}

}  // namespace nvblox
