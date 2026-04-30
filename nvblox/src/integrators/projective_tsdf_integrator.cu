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
#include <nvblox/integrators/projective_tsdf_integrator.h>
#include <nvblox/integrators/internal/cuda/impl/projective_tsdf_integrator_impl.cuh>

#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/projective_integrator_params.h"
#include "nvblox/integrators/weighting_function.h"

namespace nvblox {

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator()
    : ProjectiveTsdfIntegrator(std::make_shared<CudaStreamOwning>()) {}

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : ProjectiveIntegrator<TsdfVoxel>(cuda_stream) {
  update_functor_host_ptr_ =
      make_unified<UpdateTsdfVoxelFunctor>(MemoryType::kHost);
}

ProjectiveTsdfIntegrator::~ProjectiveTsdfIntegrator() {
  // NOTE(alexmillane): We can't default this in the header file because to the
  // unified_ptr to a forward declared type. The type has to be defined where
  // the destructor is.
}

unified_ptr<UpdateTsdfVoxelFunctor>
ProjectiveTsdfIntegrator::getTsdfUpdateFunctorOnDevice(float voxel_size) {
  // Set the update function params
  // NOTE(alex.millane): We do this with every frame integration to avoid
  // bug-prone logic for detecting when params have changed etc.
  update_functor_host_ptr_->max_weight_ = max_weight();
  update_functor_host_ptr_->truncation_distance_m_ =
      get_truncation_distance_m(voxel_size);
  update_functor_host_ptr_->invalid_depth_decay_factor_ =
      invalid_depth_decay_factor();
  update_functor_host_ptr_->weighting_function_ =
      WeightingFunction(weighting_function_type_);
  update_functor_host_ptr_->dynamic_discrepancy_threshold_m_ =
      dynamic_discrepancy_threshold_m();
  update_functor_host_ptr_->dynamic_discrepancy_min_weight_ =
      dynamic_discrepancy_min_weight();
  // Transfer to the device
  return update_functor_host_ptr_.cloneAsync(MemoryType::kDevice,
                                             *cuda_stream_);
}

float ProjectiveTsdfIntegrator::max_weight() const { return max_weight_; }

void ProjectiveTsdfIntegrator::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

WeightingFunctionType ProjectiveTsdfIntegrator::weighting_function_type()
    const {
  return weighting_function_type_;
}

void ProjectiveTsdfIntegrator::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

float ProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m() const {
  return marked_unobserved_voxels_distance_m_;
}

void ProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m(
    float marked_unobserved_voxels_distance_m) {
  marked_unobserved_voxels_distance_m_ = marked_unobserved_voxels_distance_m;
}

float ProjectiveTsdfIntegrator::marked_unobserved_voxels_weight() const {
  return marked_unobserved_voxels_weight_;
}

void ProjectiveTsdfIntegrator::marked_unobserved_voxels_weight(
    float marked_unobserved_voxels_weight) {
  marked_unobserved_voxels_weight_ = marked_unobserved_voxels_weight;
}

float ProjectiveTsdfIntegrator::invalid_depth_decay_factor() const {
  return invalid_depth_decay_factor_;
}

void ProjectiveTsdfIntegrator::invalid_depth_decay_factor(
    float invalid_depth_decay_factor) {
  invalid_depth_decay_factor_ = invalid_depth_decay_factor;
}

float ProjectiveTsdfIntegrator::dynamic_discrepancy_threshold_m() const {
  return dynamic_discrepancy_threshold_m_;
}

void ProjectiveTsdfIntegrator::dynamic_discrepancy_threshold_m(
    float dynamic_discrepancy_threshold_m) {
  dynamic_discrepancy_threshold_m_ = dynamic_discrepancy_threshold_m;
}

float ProjectiveTsdfIntegrator::dynamic_discrepancy_min_weight() const {
  return dynamic_discrepancy_min_weight_;
}

void ProjectiveTsdfIntegrator::dynamic_discrepancy_min_weight(
    float dynamic_discrepancy_min_weight) {
  CHECK_GE(dynamic_discrepancy_min_weight, 0.0f);
  dynamic_discrepancy_min_weight_ = dynamic_discrepancy_min_weight;
}

std::string ProjectiveTsdfIntegrator::getIntegratorName() const {
  return "tsdf";
}

void ProjectiveTsdfIntegrator::markUnobservedFreeInsideRadius(
    const Vector3f& center, float radius, TsdfLayer* layer,
    std::vector<Index3D>* updated_blocks_ptr) {
  markUnobservedFreeInsideRadiusTemplate(center, radius, layer,
                                         updated_blocks_ptr);
}

parameters::ParameterTreeNode ProjectiveTsdfIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "projective_tsdf_integrator" : name_remap;
  // NOTE(alexmillane): Wrapping our weighting function to_string version in the
  // std::function for passing to the parameter tree node constructor because it
  // seems to have trouble with template deduction.
  std::function<std::string(const WeightingFunctionType&)>
      weighting_function_to_string =
          [](const WeightingFunctionType& w) { return to_string(w); };

  return ParameterTreeNode(
      name,
      {ParameterTreeNode("max_weight:", max_weight_),
       ParameterTreeNode("marked_unobserved_voxels_distance_m:",
                         marked_unobserved_voxels_distance_m_),
       ParameterTreeNode("marked_unobserved_voxels_weight:",
                         marked_unobserved_voxels_weight_),
       ParameterTreeNode("weighting_function_type:", weighting_function_type_,
                         weighting_function_to_string),
       ParameterTreeNode("invalid_depth_decay_factor:",
                         invalid_depth_decay_factor_),
       ParameterTreeNode("dynamic_discrepancy_threshold_m:",
                         dynamic_discrepancy_threshold_m_),
       ParameterTreeNode("dynamic_discrepancy_min_weight:",
                         dynamic_discrepancy_min_weight_),
       ProjectiveIntegrator<TsdfVoxel>::getParameterTree()});
}

}  // namespace nvblox
