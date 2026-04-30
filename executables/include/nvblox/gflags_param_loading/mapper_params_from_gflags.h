/*
Copyright 2023-2024 NVIDIA CORPORATION

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

#include <gflags/gflags.h>

#include "nvblox/mapper/mapper_params.h"
#include "nvblox/mapper/multi_mapper_params.h"

namespace nvblox {

// <<<<<<<<<<<<<<<<<<<<<<<<<< DEFINE THE PARAMS >>>>>>>>>>>>>>>>>>>>>>>>>>

// ======= MAPPER =======
DEFINE_bool(do_depth_preprocessing, kDoDepthPrepocessingParamDesc.default_value,
            kDoDepthPrepocessingParamDesc.help_string);

DEFINE_int32(depth_preprocessing_num_dilations,
             kDepthPreprocessingNumDilationsParamDesc.default_value,
             kDepthPreprocessingNumDilationsParamDesc.help_string);

DEFINE_double(esdf_slice_min_height, kEsdfSliceMinHeightParamDesc.default_value,
              kEsdfSliceMinHeightParamDesc.help_string);

DEFINE_double(esdf_slice_max_height, kEsdfSliceMaxHeightParamDesc.default_value,
              kEsdfSliceMaxHeightParamDesc.help_string);

DEFINE_double(esdf_slice_height, kEsdfSliceHeightParamDesc.default_value,
              kEsdfSliceHeightParamDesc.help_string);

DEFINE_double(slice_height_above_plane_m,
              kSliceHeightAbovePlaneMParamDesc.default_value,
              kSliceHeightAbovePlaneMParamDesc.help_string);

DEFINE_double(slice_height_thickness_m,
              kSliceHeightThicknessMParamDesc.default_value,
              kSliceHeightThicknessMParamDesc.help_string);

// ======= MULTI-MAPPER =======
DEFINE_int32(connected_mask_component_size_threshold,
             kConnectedMaskComponentSizeThresholdParamDesc.default_value,
             kConnectedMaskComponentSizeThresholdParamDesc.help_string);

DEFINE_bool(remove_small_connected_components,
            kRemoveSmallConnectedComponentsParamDesc.default_value,
            kRemoveSmallConnectedComponentsParamDesc.help_string);

DEFINE_double(ground_points_candidates_min_z_m,
              kGroundPointsCandidatesMinZMDesc.default_value,
              kGroundPointsCandidatesMinZMDesc.help_string);

DEFINE_double(ground_points_candidates_max_z_m,
              kGroundPointsCandidatesMaxZMDesc.default_value,
              kGroundPointsCandidatesMaxZMDesc.help_string);

DEFINE_double(ransac_distance_threshold_m,
              kRansacDistanceThresholdMDesc.default_value,
              kRansacDistanceThresholdMDesc.help_string);

DEFINE_int32(num_ransac_iterations, kNumRansacIterationsDesc.default_value,
             kNumRansacIterationsDesc.help_string);

DEFINE_int32(ransac_type, static_cast<int>(kRansacTypeDesc.default_value),
             kRansacTypeDesc.help_string);

DEFINE_bool(experimental_use_ground_plane_estimation,
            kExperimentalUseGroundPlaneEstimationDesc.default_value,
            kExperimentalUseGroundPlaneEstimationDesc.help_string);

// ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY) =======
DEFINE_double(
    projective_integrator_max_integration_distance_m,
    kProjectiveIntegratorMaxIntegrationDistanceMParamDesc.default_value,
    kProjectiveIntegratorMaxIntegrationDistanceMParamDesc.help_string);

DEFINE_double(projective_integrator_truncation_distance_vox,
              kProjectiveIntegratorTruncationDistanceVoxParamDesc.default_value,
              kProjectiveIntegratorTruncationDistanceVoxParamDesc.help_string);

DEFINE_double(projective_integrator_max_weight,
              kProjectiveIntegratorMaxWeightParamDesc.default_value,
              kProjectiveIntegratorMaxWeightParamDesc.help_string);

DEFINE_double(projective_tsdf_integrator_invalid_depth_decay_factor,
              kProjectiveTsdfIntegratorInvalidDepthDecayFactor.default_value,
              kProjectiveTsdfIntegratorInvalidDepthDecayFactor.help_string);

DEFINE_int32(
    projective_integrator_weighting_mode,
    static_cast<int>(kProjectiveIntegratorWeightingModeParamDesc.default_value),
    kProjectiveIntegratorWeightingModeParamDesc.help_string);

// ======= OCCUPANCY INTEGRATOR =======
DEFINE_double(free_region_occupancy_probability,
              kFreeRegionOccupancyProbabilityParamDesc.default_value,
              kFreeRegionOccupancyProbabilityParamDesc.help_string);

DEFINE_double(occupied_region_occupancy_probability,
              kOccupiedRegionOccupancyProbabilityParamDesc.default_value,
              kOccupiedRegionOccupancyProbabilityParamDesc.help_string);

DEFINE_double(unobserved_region_occupancy_probability,
              kUnobservedRegionOccupancyProbabilityParamDesc.default_value,
              kUnobservedRegionOccupancyProbabilityParamDesc.help_string);

DEFINE_double(occupied_region_half_width_m,
              kOccupiedRegionHalfWidthMParamDesc.default_value,
              kOccupiedRegionHalfWidthMParamDesc.help_string);

// ======= VIEW CALCULATOR =======
DEFINE_int32(raycast_subsampling_factor,
             kRaycastSubsamplingFactorDesc.default_value,
             kRaycastSubsamplingFactorDesc.help_string);

DEFINE_int32(workspace_bounds_type,
             static_cast<int>(kWorkspaceBoundsTypeDesc.default_value),
             kWorkspaceBoundsTypeDesc.help_string);

DEFINE_double(workspace_bounds_min_height_m,
              kWorkspaceBoundsMinHeightDesc.default_value,
              kWorkspaceBoundsMinHeightDesc.help_string);

DEFINE_double(workspace_bounds_max_height_m,
              kWorkspaceBoundsMaxHeightDesc.default_value,
              kWorkspaceBoundsMaxHeightDesc.help_string);

DEFINE_double(workspace_bounds_min_corner_x_m,
              kWorkspaceBoundsMinCornerXDesc.default_value,
              kWorkspaceBoundsMinCornerXDesc.help_string);

DEFINE_double(workspace_bounds_max_corner_x_m,
              kWorkspaceBoundsMaxCornerXDesc.default_value,
              kWorkspaceBoundsMaxCornerXDesc.help_string);

DEFINE_double(workspace_bounds_min_corner_y_m,
              kWorkspaceBoundsMinCornerYDesc.default_value,
              kWorkspaceBoundsMinCornerYDesc.help_string);

DEFINE_double(workspace_bounds_max_corner_y_m,
              kWorkspaceBoundsMaxCornerYDesc.default_value,
              kWorkspaceBoundsMaxCornerYDesc.help_string);

// ======= ESDF INTEGRATOR =======
DEFINE_double(esdf_integrator_max_distance_m,
              kEsdfIntegratorMaxDistanceMParamDesc.default_value,
              kEsdfIntegratorMaxDistanceMParamDesc.help_string);

DEFINE_double(esdf_integrator_min_weight,
              kEsdfIntegratorMinWeightParamDesc.default_value,
              kEsdfIntegratorMinWeightParamDesc.help_string);

DEFINE_double(esdf_integrator_max_site_distance_vox,
              kEsdfIntegratorMaxSiteDistanceVoxParamDesc.default_value,
              kEsdfIntegratorMaxSiteDistanceVoxParamDesc.help_string);

// ======= MESH INTEGRATOR =======
DEFINE_double(mesh_integrator_min_weight,
              kMeshIntegratorMinWeightParamDesc.default_value,
              kMeshIntegratorMinWeightParamDesc.help_string);

DEFINE_bool(mesh_integrator_weld_vertices,
            kMeshIntegratorWeldVerticesParamDesc.default_value,
            kMeshIntegratorWeldVerticesParamDesc.help_string);

// ======= DECAY INTEGRATOR (TSDF/OCCUPANCY)=======
DEFINE_bool(decay_integrator_deallocate_decayed_blocks,
            kDecayIntegratorDeallocateDecayedBlocks.default_value,
            kDecayIntegratorDeallocateDecayedBlocks.help_string);

// ======= TSDF DECAY INTEGRATOR =======
DEFINE_double(tsdf_decay_factor, kTsdfDecayFactorParamDesc.default_value,
              kTsdfDecayFactorParamDesc.help_string);

DEFINE_double(tsdf_decayed_weight_threshold,
              kTsdfDecayedWeightThresholdDesc.default_value,
              kTsdfDecayedWeightThresholdDesc.help_string);

DEFINE_bool(tsdf_set_free_distance_on_decayed,
            kTsdfSetFreeDistanceOnDecayedDesc.default_value,
            kTsdfSetFreeDistanceOnDecayedDesc.help_string);

DEFINE_double(tsdf_decayed_free_distance_vox,
              kTsdfDecayedFreeDistanceVoxDesc.default_value,
              kTsdfDecayedFreeDistanceVoxDesc.help_string);

// ======= OCCUPANCY DECAY INTEGRATOR =======
DEFINE_double(free_region_decay_probability,
              kFreeRegionDecayProbabilityParamDesc.default_value,
              kFreeRegionDecayProbabilityParamDesc.help_string);

DEFINE_double(occupied_region_decay_probability,
              kOccupiedRegionDecayProbabilityParamDesc.default_value,
              kOccupiedRegionDecayProbabilityParamDesc.help_string);

DEFINE_bool(occupancy_decay_to_free,
            kOccupancyDecayToFreeParamDesc.default_value,
            kOccupancyDecayToFreeParamDesc.help_string);

// ======= FREESPACE INTEGRATOR =======
DEFINE_double(max_tsdf_distance_for_occupancy_m,
              kMaxTsdfDistanceForOccupancyMParamDesc.default_value,
              kMaxTsdfDistanceForOccupancyMParamDesc.help_string);

DEFINE_int64(
    max_unobserved_to_keep_consecutive_occupancy_ms,
    static_cast<int64_t>(
        kMaxUnobservedToKeepConsecutiveOccupancyMsParamDesc.default_value),
    kMaxUnobservedToKeepConsecutiveOccupancyMsParamDesc.help_string);

DEFINE_int64(
    min_duration_since_occupied_for_freespace_ms,
    static_cast<int64_t>(
        kMinDurationSinceOccupiedForFreespaceMsParamDesc.default_value),
    kMinDurationSinceOccupiedForFreespaceMsParamDesc.help_string);

DEFINE_int64(
    min_consecutive_occupancy_duration_for_reset_ms,
    static_cast<int64_t>(
        kMinConsecutiveOccupancyDurationForResetMsParamDesc.default_value),
    kMinConsecutiveOccupancyDurationForResetMsParamDesc.help_string);

DEFINE_bool(check_neighborhood, kCheckNeighborhoodParamDesc.default_value,
            kCheckNeighborhoodParamDesc.help_string);

DEFINE_bool(initialize_to_high_confidence_freespace,
            kInitializeToHighConfidenceFreespaceParamDesc.default_value,
            kInitializeToHighConfidenceFreespaceParamDesc.help_string);

// <<<<<<<<<<<<<<<<<<<<<<<<<< GET THE PARAMS >>>>>>>>>>>>>>>>>>>>>>>>>>

inline MultiMapperParams get_multi_mapper_params_from_gflags() {
  MultiMapperParams params;
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "connected_mask_component_size_threshold")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "connected_mask_component_size_threshold = "
              << FLAGS_connected_mask_component_size_threshold;
    params.connected_mask_component_size_threshold =
        FLAGS_connected_mask_component_size_threshold;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("remove_small_connected_components")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "remove_small_connected_components = "
              << FLAGS_remove_small_connected_components;
    params.remove_small_connected_components =
        FLAGS_remove_small_connected_components;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("ground_points_candidates_min_z_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "ground_points_candidates_min_z_m = "
              << FLAGS_ground_points_candidates_min_z_m;
    params.ground_plane_estimator_params.ground_points_candidates_min_z_m =
        FLAGS_ground_points_candidates_min_z_m;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("ground_points_candidates_max_z_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "ground_points_candidates_max_z_m = "
              << FLAGS_ground_points_candidates_max_z_m;
    params.ground_plane_estimator_params.ground_points_candidates_max_z_m =
        FLAGS_ground_points_candidates_max_z_m;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("ransac_distance_threshold_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: ransac_distance_threshold_m = "
              << FLAGS_ransac_distance_threshold_m;
    params.ransac_plane_fitter_params.ransac_distance_threshold_m =
        FLAGS_ransac_distance_threshold_m;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("num_ransac_iterations")
           .is_default) {
    LOG(INFO) << "Command line parameter found: num_ransac_iterations = "
              << FLAGS_num_ransac_iterations;
    params.ransac_plane_fitter_params.num_ransac_iterations =
        FLAGS_num_ransac_iterations;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("ransac_type").is_default) {
    LOG(INFO) << "Command line parameter found: ransac_type"
              << FLAGS_ransac_type;
    params.ransac_plane_fitter_params.ransac_type =
        static_cast<RansacType>(FLAGS_ransac_type);
  }

  if (!gflags::GetCommandLineFlagInfoOrDie(
           "experimental_use_ground_plane_estimation")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "experimental_use_ground_plane_estimation = "
              << FLAGS_experimental_use_ground_plane_estimation;
    params.experimental_use_ground_plane_estimation =
        FLAGS_experimental_use_ground_plane_estimation;
  }

  return params;
}

inline MapperParams get_mapper_params_from_gflags() {
  MapperParams params;
  // ======= MAPPER =======
  // depth preprocessing
  if (!gflags::GetCommandLineFlagInfoOrDie("do_depth_preprocessing")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "do_depth_preprocessing = "
              << FLAGS_do_depth_preprocessing;
    params.do_depth_preprocessing = FLAGS_do_depth_preprocessing;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("depth_preprocessing_num_dilations")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "depth_preprocessing_num_dilations = "
              << FLAGS_depth_preprocessing_num_dilations;
    params.depth_preprocessing_num_dilations =
        FLAGS_depth_preprocessing_num_dilations;
  }
  // 2D esdf slice
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_slice_min_height")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_slice_min_height = "
              << FLAGS_esdf_slice_min_height;
    params.esdf_integrator_params.esdf_slice_min_height =
        static_cast<float>(FLAGS_esdf_slice_min_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_slice_max_height")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_slice_max_height = "
              << FLAGS_esdf_slice_max_height;
    params.esdf_integrator_params.esdf_slice_max_height =
        static_cast<float>(FLAGS_esdf_slice_max_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_slice_height").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_slice_height = "
              << FLAGS_esdf_slice_height;
    params.esdf_integrator_params.esdf_slice_height =
        static_cast<float>(FLAGS_esdf_slice_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("slice_height_above_plane_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: slice_height_above_plane_m = "
              << FLAGS_slice_height_above_plane_m;
    params.esdf_integrator_params.slice_height_above_plane_m =
        static_cast<float>(FLAGS_slice_height_above_plane_m);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("slice_height_thickness_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: slice_height_thickness_m = "
              << FLAGS_slice_height_thickness_m;
    params.esdf_integrator_params.slice_height_thickness_m =
        static_cast<float>(FLAGS_slice_height_thickness_m);
  }

  // ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY) =======
  // max integration distance
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_max_integration_distance_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_max_integration_distance_m= "
              << FLAGS_projective_integrator_max_integration_distance_m;
    params.projective_integrator_params
        .projective_integrator_max_integration_distance_m =
        FLAGS_projective_integrator_max_integration_distance_m;
  }
  // truncation distance
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_truncation_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_truncation_distance_vox = "
              << FLAGS_projective_integrator_truncation_distance_vox;
    params.projective_integrator_params
        .projective_integrator_truncation_distance_vox =
        FLAGS_projective_integrator_truncation_distance_vox;
  }
  // weighting
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_weighting_mode")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: projective_integrator_weighting_mode "
        << FLAGS_projective_integrator_weighting_mode;
    params.projective_integrator_params.projective_integrator_weighting_mode =
        static_cast<WeightingFunctionType>(
            FLAGS_projective_integrator_weighting_mode);
  }
  // max weight
  if (!gflags::GetCommandLineFlagInfoOrDie("projective_integrator_max_weight")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: projective_integrator_max_weight = "
        << FLAGS_projective_integrator_max_weight;
    params.projective_integrator_params.projective_integrator_max_weight =
        FLAGS_projective_integrator_max_weight;
  }

  // Invalid depth decay
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_tsdf_integrator_invalid_depth_decay_factor")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_tsdf_integrator_invalid_depth_decay_factor = "
              << FLAGS_projective_tsdf_integrator_invalid_depth_decay_factor;
    params.projective_integrator_params
        .projective_tsdf_integrator_invalid_depth_decay_factor =
        FLAGS_projective_tsdf_integrator_invalid_depth_decay_factor;
  }

  // ======= OCCUPANCY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("free_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "free_region_occupancy_probability = "
              << FLAGS_free_region_occupancy_probability;
    params.occupancy_integrator_params.free_region_occupancy_probability =
        FLAGS_free_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "occupied_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "occupied_region_occupancy_probability = "
              << FLAGS_occupied_region_occupancy_probability;
    params.occupancy_integrator_params.occupied_region_occupancy_probability =
        FLAGS_occupied_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "unobserved_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "unobserved_region_occupancy_probability = "
              << FLAGS_unobserved_region_occupancy_probability;
    params.occupancy_integrator_params.unobserved_region_occupancy_probability =
        FLAGS_unobserved_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupied_region_half_width_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: occupied_region_half_width_m = "
              << FLAGS_occupied_region_half_width_m;
    params.occupancy_integrator_params.occupied_region_half_width_m =
        FLAGS_occupied_region_half_width_m;
  }

  // ======= VIEW CALCULATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("raycast_subsampling_factor")
           .is_default) {
    LOG(INFO) << "Command line parameter found: raycast_subsampling_factor = "
              << FLAGS_raycast_subsampling_factor;
    params.view_calculator_params.raycast_subsampling_factor =
        FLAGS_raycast_subsampling_factor;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_type")
           .is_default) {
    LOG(INFO) << "Command line parameter found: workspace_bounds_type = "
              << FLAGS_workspace_bounds_type;
    params.view_calculator_params.workspace_bounds_type =
        static_cast<WorkspaceBoundsType>(FLAGS_workspace_bounds_type);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_min_height_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_min_height_m = "
        << FLAGS_workspace_bounds_min_height_m;
    params.view_calculator_params.workspace_bounds_min_height_m =
        FLAGS_workspace_bounds_min_height_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_max_height_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_max_height_m = "
        << FLAGS_workspace_bounds_max_height_m;
    params.view_calculator_params.workspace_bounds_max_height_m =
        FLAGS_workspace_bounds_max_height_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_min_corner_x_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_min_corner_x_m = "
        << FLAGS_workspace_bounds_min_corner_x_m;
    params.view_calculator_params.workspace_bounds_min_corner_x_m =
        FLAGS_workspace_bounds_min_corner_x_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_max_corner_x_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_max_corner_x_m = "
        << FLAGS_workspace_bounds_max_corner_x_m;
    params.view_calculator_params.workspace_bounds_max_corner_x_m =
        FLAGS_workspace_bounds_max_corner_x_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_min_corner_y_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_min_corner_y_m = "
        << FLAGS_workspace_bounds_min_corner_y_m;
    params.view_calculator_params.workspace_bounds_min_corner_y_m =
        FLAGS_workspace_bounds_min_corner_y_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("workspace_bounds_max_corner_y_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: workspace_bounds_max_corner_y_m = "
        << FLAGS_workspace_bounds_max_corner_y_m;
    params.view_calculator_params.workspace_bounds_max_corner_y_m =
        FLAGS_workspace_bounds_max_corner_y_m;
  }

  // ======= ESDF INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_integrator_min_weight = "
              << FLAGS_esdf_integrator_min_weight;
    params.esdf_integrator_params.esdf_integrator_min_weight =
        FLAGS_esdf_integrator_min_weight;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "esdf_integrator_max_site_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "esdf_integrator_max_site_distance_vox = "
              << FLAGS_esdf_integrator_max_site_distance_vox;
    params.esdf_integrator_params.esdf_integrator_max_site_distance_vox =
        FLAGS_esdf_integrator_max_site_distance_vox;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_max_distance_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: esdf_integrator_max_distance_m = "
        << FLAGS_esdf_integrator_max_distance_m;
    params.esdf_integrator_params.esdf_integrator_max_distance_m =
        FLAGS_esdf_integrator_max_distance_m;
  }

  // ======= MESH INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: mesh_integrator_min_weight = "
              << FLAGS_mesh_integrator_min_weight;
    params.mesh_integrator_params.mesh_integrator_min_weight =
        FLAGS_mesh_integrator_min_weight;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_weld_vertices")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: mesh_integrator_weld_vertices = "
        << FLAGS_mesh_integrator_weld_vertices;
    params.mesh_integrator_params.mesh_integrator_weld_vertices =
        FLAGS_mesh_integrator_weld_vertices;
  }

  // ======= DECAY INTEGRATOR (TSDF/OCCUPANCY)=======
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "decay_integrator_deallocate_decayed_blocks")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "decay_integrator_deallocate_decayed_blocks = "
              << FLAGS_decay_integrator_deallocate_decayed_blocks;
    params.decay_integrator_base_params
        .decay_integrator_deallocate_decayed_blocks =
        FLAGS_decay_integrator_deallocate_decayed_blocks;
  }

  // ======= TSDF DECAY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_decay_factor").is_default) {
    LOG(INFO) << "command line parameter found: "
                 "tsdf_decay_factor = "
              << FLAGS_tsdf_decay_factor;
    params.tsdf_decay_integrator_params.tsdf_decay_factor =
        FLAGS_tsdf_decay_factor;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_decayed_weight_threshold")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "tsdf_decayed_weight_threshold = "
              << FLAGS_tsdf_decayed_weight_threshold;
    params.tsdf_decay_integrator_params.tsdf_decayed_weight_threshold =
        FLAGS_tsdf_decayed_weight_threshold;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_set_free_distance_on_decayed")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "tsdf_set_free_distance_on_decayed = "
              << FLAGS_tsdf_set_free_distance_on_decayed;
    params.tsdf_decay_integrator_params.tsdf_set_free_distance_on_decayed =
        FLAGS_tsdf_set_free_distance_on_decayed;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_decayed_free_distance_vox")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "tsdf_decayed_free_distance_vox = "
              << FLAGS_tsdf_decayed_free_distance_vox;
    params.tsdf_decay_integrator_params.tsdf_decayed_free_distance_vox =
        FLAGS_tsdf_decayed_free_distance_vox;
  }

  // ======= OCCUPANCY DECAY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("free_region_decay_probability")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "free_region_decay_probability = "
              << FLAGS_free_region_decay_probability;
    params.occupancy_decay_integrator_params.free_region_decay_probability =
        FLAGS_free_region_decay_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupied_region_decay_probability")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "occupied_region_decay_probability = "
              << FLAGS_occupied_region_decay_probability;
    params.occupancy_decay_integrator_params.occupied_region_decay_probability =
        FLAGS_occupied_region_decay_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupancy_decay_to_free")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "occupancy_decay_to_free = "
              << FLAGS_occupancy_decay_to_free;
    params.occupancy_decay_integrator_params.occupancy_decay_to_free =
        FLAGS_occupancy_decay_to_free;
  }

  // ======= FREESPACE INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("max_tsdf_distance_for_occupancy_m")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "max_tsdf_distance_for_occupancy_m = "
              << FLAGS_max_tsdf_distance_for_occupancy_m;
    params.freespace_integrator_params.max_tsdf_distance_for_occupancy_m =
        FLAGS_max_tsdf_distance_for_occupancy_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "max_unobserved_to_keep_consecutive_occupancy_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "max_unobserved_to_keep_consecutive_occupancy_ms = "
              << FLAGS_max_unobserved_to_keep_consecutive_occupancy_ms;
    params.freespace_integrator_params
        .max_unobserved_to_keep_consecutive_occupancy_ms =
        Time(FLAGS_max_unobserved_to_keep_consecutive_occupancy_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "min_duration_since_occupied_for_freespace_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "min_duration_since_occupied_for_freespace_ms = "
              << FLAGS_min_duration_since_occupied_for_freespace_ms;
    params.freespace_integrator_params
        .min_duration_since_occupied_for_freespace_ms =
        Time(FLAGS_min_duration_since_occupied_for_freespace_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "min_consecutive_occupancy_duration_for_reset_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "min_consecutive_occupancy_duration_for_reset_ms = "
              << FLAGS_min_consecutive_occupancy_duration_for_reset_ms;
    params.freespace_integrator_params
        .min_consecutive_occupancy_duration_for_reset_ms =
        Time(FLAGS_min_consecutive_occupancy_duration_for_reset_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("check_neighborhood").is_default) {
    LOG(INFO) << "command line parameter found: "
                 "check_neighborhood = "
              << FLAGS_check_neighborhood;
    params.freespace_integrator_params.check_neighborhood =
        FLAGS_check_neighborhood;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "initialize_to_high_confidence_freespace")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "initialize_to_high_confidence_freespace = "
              << FLAGS_initialize_to_high_confidence_freespace;
    params.freespace_integrator_params.initialize_to_high_confidence_freespace =
        FLAGS_initialize_to_high_confidence_freespace;
  }

  // return the written params
  return params;
}

}  // namespace nvblox
