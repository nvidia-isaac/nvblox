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
#pragma once

#include <gflags/gflags.h>

#include "nvblox/mapper/mapper_params.h"

namespace nvblox {

// <<<<<<<<<<<<<<<<<<<<<<<<<< DEFINE THE PARAMS >>>>>>>>>>>>>>>>>>>>>>>>>>

// ======= MAPPER =======
// depth preprocessing
DEFINE_bool(do_depth_preprocessing, mapper::kDefaultDoDepthPreprocessing,
            "Whether or not to run the preprocessing pipeline on the input "
            "depth image");
DEFINE_int32(depth_preprocessing_num_dilations, -1,
             "Number of times to run the invalid region dilation in the depth "
             "preprocessing pipeline.");

// ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY) =======
// max integration distance
DEFINE_double(projective_integrator_max_integration_distance_m, -1.0,
              "Maximum distance (in meters) from the camera at which to "
              "integrate data into the TSDF or occupancy grid.");
// truncation distance
DEFINE_double(projective_integrator_truncation_distance_vox, -1.0,
              "Truncation band (in voxels).");
// weighting
// NOTE(alexmillane): Only one of these should be true at once (we'll check for
// that). By default all are false and we use the internal defaults.
DEFINE_bool(weighting_scheme_constant, false,
            "Integration weighting scheme: constant");
DEFINE_bool(weighting_scheme_constant_dropoff, false,
            "Integration weighting scheme: constant + dropoff");
DEFINE_bool(weighting_scheme_inverse_square, false,
            "Integration weighting scheme: square");
DEFINE_bool(weighting_scheme_inverse_square_dropoff, false,
            "Integration weighting scheme: square + dropoff");
// max weight
DEFINE_double(
    projective_integrator_max_weight, -1.0,
    "The maximum weight that a voxel can accumulate through integration.");

// ======= OCCUPANCY INTEGRATOR =======
DEFINE_double(free_region_occupancy_probability, -1.0,
              "The inverse sensor model occupancy probability for voxels "
              "observed as free space.");
DEFINE_double(occupied_region_occupancy_probability, -1.0,
              "The inverse sensor model occupancy probability for voxels "
              "observed as occupied.");
DEFINE_double(
    unobserved_region_occupancy_probability, -1.0,
    "The inverse sensor model occupancy probability for unobserved voxels.");
DEFINE_double(occupied_region_half_width_m, -1.0,
              "Half the width of the region which is considered as occupied.");

// ======= ESDF INTEGRATOR =======
DEFINE_double(esdf_integrator_max_distance_m, -1.0,
              "The maximum distance which we integrate ESDF distances out to.");
DEFINE_double(esdf_integrator_min_weight, -1.0,
              "The minimum weight at which to consider a voxel a site.");
DEFINE_double(esdf_integrator_max_site_distance_vox, -1.0,
              "The maximum distance at which we consider a TSDF voxel a site.");

// ======= MESH INTEGRATOR =======
DEFINE_double(mesh_integrator_min_weight, -1.0,
              "The minimum weight a tsdf voxel must have before it is meshed.");
DEFINE_bool(mesh_integrator_weld_vertices, MeshIntegrator::kDefaultWeldVertices,
            "Whether or not to weld duplicate vertices in the mesh.");

// ======= TSDF DECAY INTEGRATOR =======
DEFINE_double(tsdf_decay_factor, -1.0,
              "Multiplicative factor used by TsdfDecay to decay the weights");

// ======= OCCUPANCY DECAY INTEGRATOR =======
DEFINE_double(free_region_decay_probability, -1.0,
              "The decay probability that is applied to the free region on "
              "decay. Must be in `[0.5, 1.0]`.");
DEFINE_double(occupied_region_decay_probability, -1.0,
              "The decay probability that is applied to the occupied region on "
              "decay. Must be in `[0.0, 0.5]`.");

// ======= FREESPACE INTEGRATOR =======
DEFINE_double(max_tsdf_distance_for_occupancy_m, -1.0,
              "Tsdf distance below which we assume a voxel to be occupied (non "
              "freespace).");
DEFINE_int32(max_unobserved_to_keep_consecutive_occupancy_ms, -1,
             "Maximum duration of no observed occupancy to keep consecutive "
             "occupancy alive.");
DEFINE_int32(min_duration_since_occupied_for_freespace_ms, -1,
             "Minimum duration since last observed occupancy to consider voxel "
             "as free.");
DEFINE_int32(min_consecutive_occupancy_duration_for_reset_ms, -1,
             "Minimum duration of consecutive occupancy to turn a high "
             "confidence free voxel back to occupied.");
DEFINE_bool(check_neighborhood, FreespaceIntegrator::kDefaultCheckNeighborhood,
            "Whether to check the occupancy of the neighboring voxels for the "
            "high confidence freespace update.");

// <<<<<<<<<<<<<<<<<<<<<<<<<< GET THE PARAMS >>>>>>>>>>>>>>>>>>>>>>>>>>

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

  // ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY) =======
  // max integration distance
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_max_integration_distance_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_max_integration_distance_m= "
              << FLAGS_projective_integrator_max_integration_distance_m;
    params.projective_integrator_max_integration_distance_m =
        FLAGS_projective_integrator_max_integration_distance_m;
  }
  // truncation distance
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_truncation_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_truncation_distance_vox = "
              << FLAGS_projective_integrator_truncation_distance_vox;
    params.projective_integrator_truncation_distance_vox =
        FLAGS_projective_integrator_truncation_distance_vox;
  }
  // weighting
  int num_weighting_schemes_requested = 0;
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_constant")
           .is_default) {
    LOG(INFO) << "Command line parameter found: weighting_scheme_constant = "
              << FLAGS_weighting_scheme_constant;
    params.projective_integrator_weighting_mode =
        WeightingFunctionType::kConstantWeight;
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_constant_dropoff")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: weighting_scheme_constant_dropoff = "
        << FLAGS_weighting_scheme_constant_dropoff;
    params.projective_integrator_weighting_mode =
        WeightingFunctionType::kConstantDropoffWeight;
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_inverse_square")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: weighting_scheme_inverse_square = "
        << FLAGS_weighting_scheme_inverse_square;
    params.projective_integrator_weighting_mode =
        WeightingFunctionType::kInverseSquareWeight;
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "weighting_scheme_inverse_square_dropoff")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "weighting_scheme_inverse_square_dropoff = "
              << FLAGS_weighting_scheme_inverse_square_dropoff;
    params.projective_integrator_weighting_mode =
        WeightingFunctionType::kInverseSquareDropoffWeight;
    ++num_weighting_schemes_requested;
  }
  CHECK_LT(num_weighting_schemes_requested, 2)
      << "You requested two or more weighting schemes on the command line. "
         "Maximum one.";
  // max weight
  if (!gflags::GetCommandLineFlagInfoOrDie("projective_integrator_max_weight")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: projective_integrator_max_weight = "
        << FLAGS_projective_integrator_max_weight;
    params.projective_integrator_max_weight =
        FLAGS_projective_integrator_max_weight;
  }

  // ======= OCCUPANCY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("free_region_occupancy_probability")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: free_region_occupancy_probability = "
        << FLAGS_free_region_occupancy_probability;
    params.free_region_occupancy_probability =
        FLAGS_free_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "occupied_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "occupied_region_occupancy_probability = "
              << FLAGS_occupied_region_occupancy_probability;
    params.occupied_region_occupancy_probability =
        FLAGS_occupied_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "unobserved_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "unobserved_region_occupancy_probability = "
              << FLAGS_unobserved_region_occupancy_probability;
    params.unobserved_region_occupancy_probability =
        FLAGS_unobserved_region_occupancy_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupied_region_half_width_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: occupied_region_half_width_m = "
              << FLAGS_occupied_region_half_width_m;
    params.occupied_region_half_width_m = FLAGS_occupied_region_half_width_m;
  }

  // ======= ESDF INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_integrator_min_weight = "
              << FLAGS_esdf_integrator_min_weight;
    params.esdf_integrator_min_weight = FLAGS_esdf_integrator_min_weight;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "esdf_integrator_max_site_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "esdf_integrator_max_site_distance_vox = "
              << FLAGS_esdf_integrator_max_site_distance_vox;
    params.esdf_integrator_max_site_distance_vox =
        FLAGS_esdf_integrator_max_site_distance_vox;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_max_distance_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: esdf_integrator_max_distance_m = "
        << FLAGS_esdf_integrator_max_distance_m;
    params.esdf_integrator_max_distance_m =
        FLAGS_esdf_integrator_max_distance_m;
  }

  // ======= MESH INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: mesh_integrator_min_weight = "
              << FLAGS_mesh_integrator_min_weight;
    params.mesh_integrator_min_weight = FLAGS_mesh_integrator_min_weight;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_weld_vertices")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: mesh_integrator_weld_vertices = "
        << FLAGS_mesh_integrator_weld_vertices;
    params.mesh_integrator_weld_vertices = FLAGS_mesh_integrator_weld_vertices;
  }

  // ======= TSDF DECAY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_decay_factor").is_default) {
    LOG(INFO) << "command line parameter found: "
                 "tsdf_decay_factor = "
              << FLAGS_tsdf_decay_factor;
    params.tsdf_decay_factor = FLAGS_tsdf_decay_factor;
  }

  // ======= OCCUPANCY DECAY INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("free_region_decay_probability")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "free_region_decay_probability = "
              << FLAGS_free_region_decay_probability;
    params.free_region_decay_probability = FLAGS_free_region_decay_probability;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupied_region_decay_probability")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "occupied_region_decay_probability = "
              << FLAGS_occupied_region_decay_probability;
    params.occupied_region_decay_probability =
        FLAGS_occupied_region_decay_probability;
  }

  // ======= FREESPACE INTEGRATOR =======
  if (!gflags::GetCommandLineFlagInfoOrDie("max_tsdf_distance_for_occupancy_m")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "max_tsdf_distance_for_occupancy_m = "
              << FLAGS_max_tsdf_distance_for_occupancy_m;
    params.max_tsdf_distance_for_occupancy_m =
        FLAGS_max_tsdf_distance_for_occupancy_m;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "max_unobserved_to_keep_consecutive_occupancy_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "max_unobserved_to_keep_consecutive_occupancy_ms = "
              << FLAGS_max_unobserved_to_keep_consecutive_occupancy_ms;
    params.max_unobserved_to_keep_consecutive_occupancy_ms =
        Time(FLAGS_max_unobserved_to_keep_consecutive_occupancy_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "min_duration_since_occupied_for_freespace_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "min_duration_since_occupied_for_freespace_ms = "
              << FLAGS_min_duration_since_occupied_for_freespace_ms;
    params.min_duration_since_occupied_for_freespace_ms =
        Time(FLAGS_min_duration_since_occupied_for_freespace_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "min_consecutive_occupancy_duration_for_reset_ms")
           .is_default) {
    LOG(INFO) << "command line parameter found: "
                 "min_consecutive_occupancy_duration_for_reset_ms = "
              << FLAGS_min_consecutive_occupancy_duration_for_reset_ms;
    params.min_consecutive_occupancy_duration_for_reset_ms =
        Time(FLAGS_min_consecutive_occupancy_duration_for_reset_ms);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("check_neighborhood").is_default) {
    LOG(INFO) << "command line parameter found: "
                 "check_neighborhood = "
              << FLAGS_check_neighborhood;
    params.check_neighborhood = FLAGS_check_neighborhood;
  }

  // return the written params
  return params;
}

}  // namespace nvblox