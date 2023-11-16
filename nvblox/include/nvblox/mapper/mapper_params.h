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

#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_occupancy_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/tsdf_decay_integrator.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/mesh/mesh_streamer.h"

namespace nvblox {

namespace mapper {
// NOTE(remos): These constexpr params should ideally be members of the mapper
// class. But because of a circular dependency we can not import the mapper
// header file here and forwarding does not work for static constexpr members.
// This is why we place them inside a "mapper" namespace to imitate the
// Mapper::kDefaultValue notation.
constexpr bool kDefaultDoDepthPreprocessing = false;
constexpr bool kDefaultDepthPreprocessingNumDilations = 4;
constexpr bool kDefaultMaintainMeshBlockStreamQueue = false;
}  // namespace mapper

struct MapperParams {
  // ======= MAPPER =======
  // Depth preprocessing
  bool do_depth_preprocessing = mapper::kDefaultDoDepthPreprocessing;
  int depth_preprocessing_num_dilations =
      mapper::kDefaultDepthPreprocessingNumDilations;
  // Mesh block streaming
  bool maintain_mesh_block_stream_queue =
      mapper::kDefaultMaintainMeshBlockStreamQueue;

  // ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY) =======
  // max integration distance
  float projective_integrator_max_integration_distance_m =
      ProjectiveIntegrator<void>::kDefaultMaxIntegrationDistanceM;
  float lidar_projective_integrator_max_integration_distance_m =
      ProjectiveIntegrator<void>::kDefaultLidarMaxIntegrationDistance;
  // truncation distance
  float projective_integrator_truncation_distance_vox =
      ProjectiveIntegrator<void>::kDefaultTruncationDistanceVox;
  // weighting
  WeightingFunctionType projective_integrator_weighting_mode =
      ProjectiveIntegrator<void>::kDefaultWeightingFunctionType;
  // max weight
  float projective_integrator_max_weight =
      ProjectiveIntegrator<void>::kDefaultMaxWeight;

  // ======= OCCUPANCY INTEGRATOR =======
  float free_region_occupancy_probability =
      ProjectiveOccupancyIntegrator::kDefaultFreeRegionOccupancyProbability;
  float occupied_region_occupancy_probability =
      ProjectiveOccupancyIntegrator::kDefaultOccupiedRegionOccupancyProbability;
  float unobserved_region_occupancy_probability =
      ProjectiveOccupancyIntegrator::
          kDefaultUnobservedRegionOccupancyProbability;
  float occupied_region_half_width_m =
      ProjectiveOccupancyIntegrator::kDefaultOccupiedRegionHalfWidthM;

  // ======= ESDF INTEGRATOR =======
  float esdf_integrator_max_distance_m =
      EsdfIntegrator::kDefaultMaxEsdfDistanceM;
  float esdf_integrator_min_weight = EsdfIntegrator::kDefaultMinTsdfWeight;
  float esdf_integrator_max_site_distance_vox =
      EsdfIntegrator::kDefaultMaxTsdfSiteDistanceVox;

  // ======= MESH INTEGRATOR =======
  float mesh_integrator_min_weight = MeshIntegrator::kDefaultMinWeight;
  bool mesh_integrator_weld_vertices = MeshIntegrator::kDefaultWeldVertices;

  // ======= TSDF DECAY INTEGRATOR =======
  float tsdf_decay_factor = TsdfDecayIntegrator::kDefaultTsdfDecayFactor;

  // ======= OCCUPANCY DECAY INTEGRATOR =======
  float free_region_decay_probability =
      OccupancyDecayIntegrator::kDefaultFreeRegionDecayProbability;
  float occupied_region_decay_probability =
      OccupancyDecayIntegrator::kDefaultOccupiedRegionDecayProbability;

  // ======= FREESPACE INTEGRATOR =======
  float max_tsdf_distance_for_occupancy_m{
      FreespaceIntegrator::kDefaultMaxTsdfDistanceForOccupancyM};
  Time max_unobserved_to_keep_consecutive_occupancy_ms{
      FreespaceIntegrator::kDefaultMaxUnobservedToKeepConsecutiveOccupancyMs};
  Time min_duration_since_occupied_for_freespace_ms{
      FreespaceIntegrator::kDefaultMinDurationSinceOccupiedForFreespaceMs};
  Time min_consecutive_occupancy_duration_for_reset_ms{
      FreespaceIntegrator::kDefaultMinConsecutiveOccupancyDurationForResetMs};
  bool check_neighborhood = FreespaceIntegrator::kDefaultCheckNeighborhood;

  // ======= MESH STREAMER =======
  bool mesh_streamer_exclude_blocks_above_height{
      MeshStreamerOldestBlocks::kDefaultExcludeBlocksAboveHeight};
  float mesh_streamer_exclusion_height_m{
      MeshStreamerOldestBlocks::kDefaultExclusionHeightM};
  bool mesh_streamer_exclude_blocks_outside_radius{
      MeshStreamerOldestBlocks::kDefaultExcludeBlocksOutsideRadius};
  float mesh_streamer_exclusion_radius_m{
      MeshStreamerOldestBlocks::kDefaultExclusionRadiusM};
};

}  // namespace nvblox
