/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <nvblox_torch/py_mapper_params.h>

namespace pynvblox {

nvblox::WeightingFunctionType weighting_function_type_from_string(
    const std::string& fn_string) {
  if (fn_string == "kConstantWeight") {
    return nvblox::WeightingFunctionType::kConstantWeight;
  } else if (fn_string == "kConstantDropoffWeight") {
    return nvblox::WeightingFunctionType::kConstantDropoffWeight;
  } else if (fn_string == "kInverseSquareWeight") {
    return nvblox::WeightingFunctionType::kInverseSquareWeight;
  } else if (fn_string == "kInverseSquareDropoffWeight") {
    return nvblox::WeightingFunctionType::kInverseSquareDropoffWeight;
  } else if ("kInverseSquareTsdfDistancePenalty") {
    return nvblox::WeightingFunctionType::kInverseSquareTsdfDistancePenalty;
  } else if (fn_string == "kLinearWithMax") {
    return nvblox::WeightingFunctionType::kLinearWithMax;
  } else {
    NVBLOX_ABORT(std::string("Unrecognized weighting function type: ") +
                 fn_string);
    return nvblox::WeightingFunctionType::kConstantWeight;
  }
}

nvblox::WorkspaceBoundsType workspace_bounds_type_from_string(
    const std::string& fn_string) {
  if (fn_string == "kUnbounded") {
    return nvblox::WorkspaceBoundsType::kUnbounded;
  } else if (fn_string == "kHeightBounds") {
    return nvblox::WorkspaceBoundsType::kHeightBounds;
  } else if (fn_string == "kBoundingBox") {
    return nvblox::WorkspaceBoundsType::kBoundingBox;
  } else {
    NVBLOX_ABORT(std::string("Unrecognized workspace bound type: ") +
                 fn_string);
    return nvblox::WorkspaceBoundsType::kUnbounded;
  }
}

/*****************************
 * PROJECTIVE INTEGRATOR PARAMS
 ******************************/

double ProjectiveIntegratorParams::
    get_projective_integrator_max_integration_distance_m() const {
  return static_cast<double>(
      params_->projective_integrator_max_integration_distance_m);
}
void ProjectiveIntegratorParams::
    set_projective_integrator_max_integration_distance_m(double value) {
  params_->projective_integrator_max_integration_distance_m =
      static_cast<float>(value);
}

double ProjectiveIntegratorParams::
    get_lidar_projective_integrator_max_integration_distance_m() const {
  return static_cast<double>(
      params_->lidar_projective_integrator_max_integration_distance_m);
}
void ProjectiveIntegratorParams::
    set_lidar_projective_integrator_max_integration_distance_m(double value) {
  params_->lidar_projective_integrator_max_integration_distance_m =
      static_cast<float>(value);
}

double
ProjectiveIntegratorParams::get_projective_integrator_truncation_distance_vox()
    const {
  return static_cast<double>(
      params_->projective_integrator_truncation_distance_vox);
}
void ProjectiveIntegratorParams::
    set_projective_integrator_truncation_distance_vox(double value) {
  params_->projective_integrator_truncation_distance_vox =
      static_cast<float>(value);
}

std::string
ProjectiveIntegratorParams::get_projective_integrator_weighting_mode() const {
  return nvblox::to_string(params_->projective_integrator_weighting_mode);
}

void ProjectiveIntegratorParams::set_projective_integrator_weighting_mode(
    std::string value) {
  params_->projective_integrator_weighting_mode =
      weighting_function_type_from_string(value);
}

double ProjectiveIntegratorParams::get_projective_integrator_max_weight()
    const {
  return static_cast<double>(params_->projective_integrator_max_weight);
}
void ProjectiveIntegratorParams::set_projective_integrator_max_weight(
    double value) {
  params_->projective_integrator_max_weight = static_cast<float>(value);
}

double ProjectiveIntegratorParams::
    get_projective_tsdf_integrator_invalid_depth_decay_factor() const {
  return static_cast<double>(
      params_->projective_tsdf_integrator_invalid_depth_decay_factor);
}
void ProjectiveIntegratorParams::
    set_projective_tsdf_integrator_invalid_depth_decay_factor(double value) {
  params_->projective_tsdf_integrator_invalid_depth_decay_factor =
      static_cast<float>(value);
}

double ProjectiveIntegratorParams::
    get_projective_appearance_integrator_measurement_weight() const {
  return static_cast<double>(
      params_->projective_appearance_integrator_measurement_weight);
}
void ProjectiveIntegratorParams::
    set_projective_appearance_integrator_measurement_weight(double value) {
  params_->projective_appearance_integrator_measurement_weight =
      static_cast<float>(value);
}

/*****************************
 * MESH INTEGRATOR PARAMS
 ******************************/

double MeshIntegratorParams::get_mesh_integrator_min_weight() const {
  return static_cast<double>(params_->mesh_integrator_min_weight);
}

void MeshIntegratorParams::set_mesh_integrator_min_weight(double value) const {
  params_->mesh_integrator_min_weight = static_cast<float>(value);
}

bool MeshIntegratorParams::get_mesh_integrator_weld_vertices() const {
  return params_->mesh_integrator_weld_vertices;
}

void MeshIntegratorParams::set_mesh_integrator_weld_vertices(bool value) {
  params_->mesh_integrator_weld_vertices = value;
}

/*****************************
 * DECAY INTEGRATOR BASE PARAMS
 ******************************/

bool DecayIntegratorBaseParams::get_decay_integrator_deallocate_decayed_blocks()
    const {
  return params_->decay_integrator_deallocate_decayed_blocks;
}

void DecayIntegratorBaseParams::set_decay_integrator_deallocate_decayed_blocks(
    bool value) const {
  params_->decay_integrator_deallocate_decayed_blocks = value;
}

/*****************************
 * TSDF DECAY INTEGRATOR PARAMS
 ******************************/

double TsdfDecayIntegratorParams::get_tsdf_decay_factor() const {
  return static_cast<double>(params_->tsdf_decay_factor);
}

void TsdfDecayIntegratorParams::set_tsdf_decay_factor(double value) const {
  params_->tsdf_decay_factor = static_cast<float>(value);
}

double TsdfDecayIntegratorParams::get_tsdf_decayed_weight_threshold() const {
  return static_cast<double>(params_->tsdf_decayed_weight_threshold);
}

void TsdfDecayIntegratorParams::set_tsdf_decayed_weight_threshold(
    double value) const {
  params_->tsdf_decayed_weight_threshold = static_cast<float>(value);
}

bool TsdfDecayIntegratorParams::get_tsdf_set_free_distance_on_decayed() const {
  return params_->tsdf_set_free_distance_on_decayed;
}

void TsdfDecayIntegratorParams::set_tsdf_set_free_distance_on_decayed(
    bool value) const {
  params_->tsdf_set_free_distance_on_decayed = value;
}

double TsdfDecayIntegratorParams::get_tsdf_decayed_free_distance_vox() const {
  return static_cast<double>(params_->tsdf_decayed_free_distance_vox);
}

void TsdfDecayIntegratorParams::set_tsdf_decayed_free_distance_vox(
    double value) const {
  params_->tsdf_decayed_free_distance_vox = static_cast<float>(value);
}

/**********************************
 * OCCUPANCY DECAY INTEGRATOR PARAMS
 **********************************/

double OccupancyDecayIntegratorParams::get_free_region_decay_probability()
    const {
  return static_cast<double>(params_->free_region_decay_probability);
}

void OccupancyDecayIntegratorParams::set_free_region_decay_probability(
    double value) const {
  params_->free_region_decay_probability = static_cast<float>(value);
}

double OccupancyDecayIntegratorParams::get_occupied_region_decay_probability()
    const {
  return static_cast<double>(params_->occupied_region_decay_probability);
}

void OccupancyDecayIntegratorParams::set_occupied_region_decay_probability(
    double value) const {
  params_->occupied_region_decay_probability = static_cast<float>(value);
}

bool OccupancyDecayIntegratorParams::get_occupancy_decay_to_free() const {
  return params_->occupancy_decay_to_free;
}

void OccupancyDecayIntegratorParams::set_occupancy_decay_to_free(
    bool value) const {
  params_->occupancy_decay_to_free = value;
}

/*****************************
 * ESDF INTEGRATOR PARAMS
 ******************************/

double EsdfIntegratorParams::get_esdf_integrator_max_distance_m() const {
  return static_cast<double>(params_->esdf_integrator_max_distance_m);
}
void EsdfIntegratorParams::set_esdf_integrator_max_distance_m(
    double value) const {
  params_->esdf_integrator_max_distance_m = static_cast<float>(value);
}

double EsdfIntegratorParams::get_esdf_integrator_min_weight() const {
  return static_cast<double>(params_->esdf_integrator_min_weight);
}
void EsdfIntegratorParams::set_esdf_integrator_min_weight(double value) const {
  params_->esdf_integrator_min_weight = static_cast<float>(value);
}

double EsdfIntegratorParams::get_esdf_integrator_max_site_distance_vox() const {
  return static_cast<double>(params_->esdf_integrator_max_site_distance_vox);
}
void EsdfIntegratorParams::set_esdf_integrator_max_site_distance_vox(
    double value) const {
  params_->esdf_integrator_max_site_distance_vox = static_cast<float>(value);
}

double EsdfIntegratorParams::get_esdf_slice_min_height() const {
  return static_cast<double>(params_->esdf_slice_min_height);
}
void EsdfIntegratorParams::set_esdf_slice_min_height(double value) const {
  params_->esdf_slice_min_height = static_cast<float>(value);
}

double EsdfIntegratorParams::get_esdf_slice_max_height() const {
  return static_cast<double>(params_->esdf_slice_max_height);
}
void EsdfIntegratorParams::set_esdf_slice_max_height(double value) const {
  params_->esdf_slice_max_height = static_cast<float>(value);
}

double EsdfIntegratorParams::get_esdf_slice_height() const {
  return static_cast<double>(params_->esdf_slice_height);
}
void EsdfIntegratorParams::set_esdf_slice_height(double value) const {
  params_->esdf_slice_height = static_cast<float>(value);
}

double EsdfIntegratorParams::get_slice_height_above_plane_m() const {
  return static_cast<double>(params_->slice_height_above_plane_m);
}
void EsdfIntegratorParams::set_slice_height_above_plane_m(double value) const {
  params_->slice_height_above_plane_m = static_cast<float>(value);
}

double EsdfIntegratorParams::get_slice_height_thickness_m() const {
  return static_cast<double>(params_->slice_height_thickness_m);
}
void EsdfIntegratorParams::set_slice_height_thickness_m(double value) const {
  params_->slice_height_thickness_m = static_cast<float>(value);
}

/*****************************
 * VIEW CALCULATOR PARAMS
 ******************************/

int64_t ViewCalculatorParams::get_raycast_subsampling_factor() const {
  return static_cast<int64_t>(params_->raycast_subsampling_factor);
}
void ViewCalculatorParams::set_raycast_subsampling_factor(int64_t value) const {
  params_->raycast_subsampling_factor = static_cast<int>(value);
}

std::string ViewCalculatorParams::get_workspace_bounds_type() const {
  return nvblox::to_string(params_->workspace_bounds_type);
}
void ViewCalculatorParams::set_workspace_bounds_type(
    const std::string& value) const {
  params_->workspace_bounds_type = workspace_bounds_type_from_string(value);
}

double ViewCalculatorParams::get_workspace_bounds_min_height_m() const {
  return params_->workspace_bounds_min_height_m;
}
void ViewCalculatorParams::set_workspace_bounds_min_height_m(
    double value) const {
  params_->workspace_bounds_min_height_m = static_cast<float>(value);
}

double ViewCalculatorParams::get_workspace_bounds_max_height_m() const {
  return params_->workspace_bounds_max_height_m;
}
void ViewCalculatorParams::set_workspace_bounds_max_height_m(
    double value) const {
  params_->workspace_bounds_max_height_m = static_cast<float>(value);
}

double ViewCalculatorParams::get_workspace_bounds_min_corner_x_m() const {
  return params_->workspace_bounds_min_corner_x_m;
}
void ViewCalculatorParams::set_workspace_bounds_min_corner_x_m(
    double value) const {
  params_->workspace_bounds_min_corner_x_m = static_cast<float>(value);
}

double ViewCalculatorParams::get_workspace_bounds_max_corner_x_m() const {
  return params_->workspace_bounds_max_corner_x_m;
}
void ViewCalculatorParams::set_workspace_bounds_max_corner_x_m(
    double value) const {
  params_->workspace_bounds_max_corner_x_m = static_cast<float>(value);
}

double ViewCalculatorParams::get_workspace_bounds_min_corner_y_m() const {
  return params_->workspace_bounds_min_corner_y_m;
}
void ViewCalculatorParams::set_workspace_bounds_min_corner_y_m(
    double value) const {
  params_->workspace_bounds_min_corner_y_m = static_cast<float>(value);
}

double ViewCalculatorParams::get_workspace_bounds_max_corner_y_m() const {
  return params_->workspace_bounds_max_corner_y_m;
}
void ViewCalculatorParams::set_workspace_bounds_max_corner_y_m(
    double value) const {
  params_->workspace_bounds_max_corner_y_m = static_cast<float>(value);
}

/*****************************
 * BLOCK MEMORY POOL PARAMS
 ******************************/

int64_t BlockMemoryPoolParams::get_num_preallocated_blocks() const {
  return params_->num_preallocated_blocks;
}
void BlockMemoryPoolParams::set_num_preallocated_blocks(int64_t value) const {
  params_->num_preallocated_blocks = value;
}

double BlockMemoryPoolParams::get_expansion_factor() const {
  return params_->expansion_factor;
}
void BlockMemoryPoolParams::set_expansion_factor(double value) const {
  params_->expansion_factor = static_cast<float>(value);
}

/*****************************
 * MAPPER PARAMS
 ******************************/

c10::intrusive_ptr<ProjectiveIntegratorParams>
MapperParams::get_projective_integrator_params() const {
  return c10::make_intrusive<ProjectiveIntegratorParams>(
      params_->projective_integrator_params);
}

void MapperParams::set_projective_integrator_params(
    c10::intrusive_ptr<ProjectiveIntegratorParams> params) {
  params_->projective_integrator_params = *params->params_;
}

c10::intrusive_ptr<MeshIntegratorParams>
MapperParams::get_mesh_integrator_params() const {
  return c10::make_intrusive<MeshIntegratorParams>(
      params_->mesh_integrator_params);
}

void MapperParams::set_mesh_integrator_params(
    c10::intrusive_ptr<MeshIntegratorParams> params) {
  params_->mesh_integrator_params = *params->params_;
}

c10::intrusive_ptr<DecayIntegratorBaseParams>
MapperParams::get_decay_integrator_base_params() const {
  return c10::make_intrusive<DecayIntegratorBaseParams>(
      params_->decay_integrator_base_params);
}

void MapperParams::set_decay_integrator_base_params(
    c10::intrusive_ptr<DecayIntegratorBaseParams> params) {
  params_->decay_integrator_base_params = *params->params_;
}

c10::intrusive_ptr<TsdfDecayIntegratorParams>
MapperParams::get_tsdf_decay_integrator_params() const {
  return c10::make_intrusive<TsdfDecayIntegratorParams>(
      params_->tsdf_decay_integrator_params);
}

void MapperParams::set_tsdf_decay_integrator_params(
    c10::intrusive_ptr<TsdfDecayIntegratorParams> params) {
  params_->tsdf_decay_integrator_params = *params->params_;
}

c10::intrusive_ptr<OccupancyDecayIntegratorParams>
MapperParams::get_occupancy_decay_integrator_params() const {
  return c10::make_intrusive<OccupancyDecayIntegratorParams>(
      params_->occupancy_decay_integrator_params);
}

void MapperParams::set_occupancy_decay_integrator_params(
    c10::intrusive_ptr<OccupancyDecayIntegratorParams> params) {
  params_->occupancy_decay_integrator_params = *params->params_;
}

c10::intrusive_ptr<EsdfIntegratorParams>
MapperParams::get_esdf_integrator_params() const {
  return c10::make_intrusive<EsdfIntegratorParams>(
      params_->esdf_integrator_params);
}

void MapperParams::set_esdf_integrator_params(
    c10::intrusive_ptr<EsdfIntegratorParams> params) {
  params_->esdf_integrator_params = *params->params_;
}

c10::intrusive_ptr<ViewCalculatorParams>
MapperParams::get_view_calculator_params() const {
  return c10::make_intrusive<ViewCalculatorParams>(
      params_->view_calculator_params);
}

void MapperParams::set_view_calculator_params(
    c10::intrusive_ptr<ViewCalculatorParams> params) {
  params_->view_calculator_params = *params->params_;
}

c10::intrusive_ptr<BlockMemoryPoolParams>
MapperParams::get_block_memory_pool_params() const {
  return c10::make_intrusive<BlockMemoryPoolParams>(*block_memory_pool_params_);
}

void MapperParams::set_block_memory_pool_params(
    c10::intrusive_ptr<BlockMemoryPoolParams> params) {
  block_memory_pool_params_ = params->params_;
}

}  // namespace pynvblox
