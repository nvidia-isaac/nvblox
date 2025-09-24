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
#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <nvblox/mapper/mapper_params.h>

namespace pynvblox {

// NOTE(alexmillane, 2024.11.14): The following sub-parameter structs are
// currently unwrapped. If you need them wrapped, ask alex. Unwrapped
// sub-parameter classes:
// - OccupancyIntegratorParams occupancy_integrator_params;
// - OccupancyDecayIntegratorParams occupancy_decay_integrator_params;
// - FreespaceIntegratorParams freespace_integrator_params;

struct ProjectiveIntegratorParams : torch::CustomClassHolder {
  // Constructor
  ProjectiveIntegratorParams()
      : params_(std::make_shared<nvblox::ProjectiveIntegratorParams>()) {}
  ProjectiveIntegratorParams(const nvblox::ProjectiveIntegratorParams params)
      : params_(std::make_shared<nvblox::ProjectiveIntegratorParams>(params)) {}

  double get_projective_integrator_max_integration_distance_m() const;
  void set_projective_integrator_max_integration_distance_m(double value);

  double get_lidar_projective_integrator_max_integration_distance_m() const;
  void set_lidar_projective_integrator_max_integration_distance_m(double value);

  double get_projective_integrator_truncation_distance_vox() const;
  void set_projective_integrator_truncation_distance_vox(double value);

  std::string get_projective_integrator_weighting_mode() const;
  void set_projective_integrator_weighting_mode(std::string value);

  double get_projective_integrator_max_weight() const;
  void set_projective_integrator_max_weight(double value);

  double get_projective_tsdf_integrator_invalid_depth_decay_factor() const;
  void set_projective_tsdf_integrator_invalid_depth_decay_factor(double value);

  double get_projective_appearance_integrator_measurement_weight() const;
  void set_projective_appearance_integrator_measurement_weight(double value);

  std::shared_ptr<nvblox::ProjectiveIntegratorParams> params_;
};

struct MeshIntegratorParams : torch::CustomClassHolder {
  // Constructor
  MeshIntegratorParams()
      : params_(std::make_shared<nvblox::MeshIntegratorParams>()) {}
  MeshIntegratorParams(const nvblox::MeshIntegratorParams params)
      : params_(std::make_shared<nvblox::MeshIntegratorParams>(params)) {}

  double get_mesh_integrator_min_weight() const;
  void set_mesh_integrator_min_weight(double value) const;

  bool get_mesh_integrator_weld_vertices() const;
  void set_mesh_integrator_weld_vertices(bool value);

  std::shared_ptr<nvblox::MeshIntegratorParams> params_;
};

struct DecayIntegratorBaseParams : torch::CustomClassHolder {
  // Constructor
  DecayIntegratorBaseParams()
      : params_(std::make_shared<nvblox::DecayIntegratorBaseParams>()) {}
  DecayIntegratorBaseParams(const nvblox::DecayIntegratorBaseParams params)
      : params_(std::make_shared<nvblox::DecayIntegratorBaseParams>(params)) {}

  bool get_decay_integrator_deallocate_decayed_blocks() const;
  void set_decay_integrator_deallocate_decayed_blocks(bool value) const;

  std::shared_ptr<nvblox::DecayIntegratorBaseParams> params_;
};

struct TsdfDecayIntegratorParams : torch::CustomClassHolder {
  // Constructor
  TsdfDecayIntegratorParams()
      : params_(std::make_shared<nvblox::TsdfDecayIntegratorParams>()) {}
  TsdfDecayIntegratorParams(const nvblox::TsdfDecayIntegratorParams params)
      : params_(std::make_shared<nvblox::TsdfDecayIntegratorParams>(params)) {}

  double get_tsdf_decay_factor() const;
  void set_tsdf_decay_factor(double value) const;

  double get_tsdf_decayed_weight_threshold() const;
  void set_tsdf_decayed_weight_threshold(double value) const;

  bool get_tsdf_set_free_distance_on_decayed() const;
  void set_tsdf_set_free_distance_on_decayed(bool value) const;

  double get_tsdf_decayed_free_distance_vox() const;
  void set_tsdf_decayed_free_distance_vox(double value) const;

  std::shared_ptr<nvblox::TsdfDecayIntegratorParams> params_;
};

struct OccupancyDecayIntegratorParams : torch::CustomClassHolder {
  // Constructor
  OccupancyDecayIntegratorParams()
      : params_(std::make_shared<nvblox::OccupancyDecayIntegratorParams>()) {}
  OccupancyDecayIntegratorParams(
      const nvblox::OccupancyDecayIntegratorParams params)
      : params_(
            std::make_shared<nvblox::OccupancyDecayIntegratorParams>(params)) {}

  double get_free_region_decay_probability() const;
  void set_free_region_decay_probability(double value) const;

  double get_occupied_region_decay_probability() const;
  void set_occupied_region_decay_probability(double value) const;

  bool get_occupancy_decay_to_free() const;
  void set_occupancy_decay_to_free(bool value) const;

  std::shared_ptr<nvblox::OccupancyDecayIntegratorParams> params_;
};

struct EsdfIntegratorParams : torch::CustomClassHolder {
  // Constructor
  EsdfIntegratorParams()
      : params_(std::make_shared<nvblox::EsdfIntegratorParams>()) {}
  EsdfIntegratorParams(const nvblox::EsdfIntegratorParams params)
      : params_(std::make_shared<nvblox::EsdfIntegratorParams>(params)) {}

  double get_esdf_integrator_max_distance_m() const;
  void set_esdf_integrator_max_distance_m(double value) const;

  double get_esdf_integrator_min_weight() const;
  void set_esdf_integrator_min_weight(double value) const;

  double get_esdf_integrator_max_site_distance_vox() const;
  void set_esdf_integrator_max_site_distance_vox(double value) const;

  double get_esdf_slice_min_height() const;
  void set_esdf_slice_min_height(double value) const;

  double get_esdf_slice_max_height() const;
  void set_esdf_slice_max_height(double value) const;

  double get_esdf_slice_height() const;
  void set_esdf_slice_height(double value) const;

  double get_slice_height_above_plane_m() const;
  void set_slice_height_above_plane_m(double value) const;

  double get_slice_height_thickness_m() const;
  void set_slice_height_thickness_m(double value) const;

  std::shared_ptr<nvblox::EsdfIntegratorParams> params_;
};

struct ViewCalculatorParams : torch::CustomClassHolder {
  // Constructor
  ViewCalculatorParams()
      : params_(std::make_shared<nvblox::ViewCalculatorParams>()) {}
  ViewCalculatorParams(const nvblox::ViewCalculatorParams params)
      : params_(std::make_shared<nvblox::ViewCalculatorParams>(params)) {}

  int64_t get_raycast_subsampling_factor() const;
  void set_raycast_subsampling_factor(int64_t value) const;

  std::string get_workspace_bounds_type() const;
  void set_workspace_bounds_type(const std::string& value) const;

  double get_workspace_bounds_min_height_m() const;
  void set_workspace_bounds_min_height_m(double value) const;

  double get_workspace_bounds_max_height_m() const;
  void set_workspace_bounds_max_height_m(double value) const;

  double get_workspace_bounds_min_corner_x_m() const;
  void set_workspace_bounds_min_corner_x_m(double value) const;

  double get_workspace_bounds_max_corner_x_m() const;
  void set_workspace_bounds_max_corner_x_m(double value) const;

  double get_workspace_bounds_min_corner_y_m() const;
  void set_workspace_bounds_min_corner_y_m(double value) const;

  double get_workspace_bounds_max_corner_y_m() const;
  void set_workspace_bounds_max_corner_y_m(double value) const;

  std::shared_ptr<nvblox::ViewCalculatorParams> params_;
};

struct BlockMemoryPoolParams : torch::CustomClassHolder {
  BlockMemoryPoolParams()
      : params_(std::make_shared<nvblox::BlockMemoryPoolParams>()) {}
  BlockMemoryPoolParams(const nvblox::BlockMemoryPoolParams params)
      : params_(std::make_shared<nvblox::BlockMemoryPoolParams>(params)) {}

  int64_t get_num_preallocated_blocks() const;
  void set_num_preallocated_blocks(int64_t value) const;

  double get_expansion_factor() const;
  void set_expansion_factor(double value) const;

  std::shared_ptr<nvblox::BlockMemoryPoolParams> params_;
};

struct MapperParams : torch::CustomClassHolder {
  // Constructor
  MapperParams()
      : params_(std::make_shared<nvblox::MapperParams>()),
        block_memory_pool_params_(
            std::make_shared<nvblox::BlockMemoryPoolParams>()) {}

  c10::intrusive_ptr<ProjectiveIntegratorParams>
  get_projective_integrator_params() const;
  void set_projective_integrator_params(
      c10::intrusive_ptr<ProjectiveIntegratorParams> params);

  c10::intrusive_ptr<MeshIntegratorParams> get_mesh_integrator_params() const;
  void set_mesh_integrator_params(
      c10::intrusive_ptr<MeshIntegratorParams> params);

  c10::intrusive_ptr<DecayIntegratorBaseParams>
  get_decay_integrator_base_params() const;
  void set_decay_integrator_base_params(
      c10::intrusive_ptr<DecayIntegratorBaseParams> params);

  c10::intrusive_ptr<TsdfDecayIntegratorParams>
  get_tsdf_decay_integrator_params() const;
  void set_tsdf_decay_integrator_params(
      c10::intrusive_ptr<TsdfDecayIntegratorParams> params);

  c10::intrusive_ptr<OccupancyDecayIntegratorParams>
  get_occupancy_decay_integrator_params() const;
  void set_occupancy_decay_integrator_params(
      c10::intrusive_ptr<OccupancyDecayIntegratorParams> params);

  c10::intrusive_ptr<EsdfIntegratorParams> get_esdf_integrator_params() const;
  void set_esdf_integrator_params(
      c10::intrusive_ptr<EsdfIntegratorParams> params);

  c10::intrusive_ptr<ViewCalculatorParams> get_view_calculator_params() const;
  void set_view_calculator_params(
      c10::intrusive_ptr<ViewCalculatorParams> params);

  c10::intrusive_ptr<BlockMemoryPoolParams> get_block_memory_pool_params()
      const;
  void set_block_memory_pool_params(
      c10::intrusive_ptr<BlockMemoryPoolParams> params);

  std::shared_ptr<nvblox::MapperParams> params_;

  // TODO(dtingdahl) Remove when Block memory params becomes part of
  // MapperParams in the core lib.
  std::shared_ptr<nvblox::BlockMemoryPoolParams> block_memory_pool_params_;
};

}  // namespace pynvblox
