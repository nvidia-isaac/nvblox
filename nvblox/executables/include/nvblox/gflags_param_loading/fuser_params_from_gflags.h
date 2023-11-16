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

#include "nvblox/executables/fuser.h"

namespace nvblox {

// ============================ DEFINE THE PARAMS ============================

DEFINE_double(voxel_size, 0.0f, "Voxel resolution in meters.");
DEFINE_bool(use_2d_esdf_mode, false, "Use the 2d ESDF mode (3D if false).");

// Mapping type
// NOTE(remos): Only one of these should be true at once (we'll check for
// that). By default all are false and we use the internal defaults.
DEFINE_bool(mapping_type_static_occupancy, false,
            "mapping type: kStaticOccupancy");
DEFINE_bool(mapping_type_dynamic, false, "mapping type: kDynamic");

// Multi mapper params
DEFINE_double(esdf_2d_min_height, MultiMapper::kDefaultEsdf2dMinHeight,
              "The minimum height, in meters, to consider obstacles part of "
              "the 2D ESDF slice.");
DEFINE_double(esdf_2d_max_height, MultiMapper::kDefaultEsdf2dMaxHeight,
              "The maximum height, in meters, to consider obstacles part of "
              "the 2D ESDF slice.");
DEFINE_double(
    esdf_slice_height, MultiMapper::kDefaultEsdf2dSliceHeight,
    "The *output* slice height for the distance slice and ESDF pointcloud. "
    "Does not need to be within min and max height below. In units of meters.");
DEFINE_int32(connected_mask_component_size_threshold,
             MultiMapper::kDefaultConnectedMaskComponentSizeThreshold,
             "The minimum number of pixels of a connected component in the "
             "mask image to count as a dynamic detection.");

// Dataset flags
DEFINE_int32(num_frames, -1,
             "Number of frames to process. Negative means process all.");

// The output paths
DEFINE_string(timing_output_path, "",
              "File in which to save the timing results.");
DEFINE_string(tsdf_output_path, "",
              "File in which to save the TSDF pointcloud.");
DEFINE_string(occupancy_output_path, "",
              "File in which to save the occupancy pointcloud.");
DEFINE_string(freespace_output_path, "",
              "File in which to save the freespace pointcloud.");
DEFINE_string(esdf_output_path, "",
              "File in which to save the ESDF pointcloud.");
DEFINE_string(mesh_output_path, "", "File in which to save the surface mesh.");
DEFINE_string(map_output_path, "", "File in which to save the serialized map.");
DEFINE_string(dynamic_overlay_path, "",
              "Folder to save the dynamic mask images. Note that these "
              "overlays show the mask before before being post processed.");

// Subsampling
DEFINE_int32(projective_frame_subsampling, 0,
             "By what amount to subsample the TSDF or occupancy frames. A "
             "subsample of 3 means only every 3rd frame is taken.");
DEFINE_int32(color_frame_subsampling, 0,
             "How much to subsample the color integration by.");
DEFINE_int32(mesh_frame_subsampling, 0,
             "How much to subsample the meshing by.");
DEFINE_int32(esdf_frame_subsampling, 0,
             "How much to subsample the ESDF integration by.");

// Dynamic detection
DEFINE_double(
    frame_rate, 0.0f,
    "The frame rate of the input depth frames in Hz. Only used if running "
    "dynamic detection.");

// ============================ GET THE PARAMS ============================

inline void get_multi_mapper_params_from_gflags(float* voxel_size,
                                                MappingType* mapping_type,
                                                EsdfMode* esdf_mode,
                                                MultiMapper::Params* params) {
  if (!gflags::GetCommandLineFlagInfoOrDie("voxel_size").is_default) {
    LOG(INFO) << "Command line parameter found: voxel_size = "
              << FLAGS_voxel_size;
    *voxel_size = static_cast<float>(FLAGS_voxel_size);
  }
  int num_mapping_types_requested = 0;
  if (!gflags::GetCommandLineFlagInfoOrDie("mapping_type_static_occupancy")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: mapping_type_static_occupancy = "
        << FLAGS_mapping_type_static_occupancy;
    *mapping_type = MappingType::kStaticOccupancy;
    ++num_mapping_types_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mapping_type_dynamic").is_default) {
    LOG(INFO) << "Command line parameter found: mapping_type_dynamic = "
              << FLAGS_mapping_type_dynamic;
    *mapping_type = MappingType::kDynamic;
    ++num_mapping_types_requested;
  }
  CHECK_LT(num_mapping_types_requested, 2)
      << "You requested more than one mapping type on the command line.";

  if (!gflags::GetCommandLineFlagInfoOrDie("use_2d_esdf_mode").is_default) {
    LOG(INFO) << "Command line parameter found: use_2d_esdf_mode = "
              << FLAGS_use_2d_esdf_mode;
    *esdf_mode = EsdfMode::k2D;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_2d_min_height").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_2d_min_height = "
              << FLAGS_esdf_2d_min_height;
    params->esdf_2d_min_height = static_cast<float>(FLAGS_esdf_2d_min_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_2d_max_height").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_2d_max_height = "
              << FLAGS_esdf_2d_max_height;
    params->esdf_2d_max_height = static_cast<float>(FLAGS_esdf_2d_max_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_slice_height").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_slice_height = "
              << FLAGS_esdf_slice_height;
    params->esdf_slice_height = static_cast<float>(FLAGS_esdf_slice_height);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "connected_mask_component_size_threshold")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "connected_mask_component_size_threshold = "
              << FLAGS_connected_mask_component_size_threshold;
    params->connected_mask_component_size_threshold =
        FLAGS_connected_mask_component_size_threshold;
  }
}

inline void set_fuser_params_from_gflags(Fuser* fuser_ptr) {
  // Dataset flags
  if (!gflags::GetCommandLineFlagInfoOrDie("num_frames").is_default) {
    LOG(INFO) << "Command line parameter found: num_frames = "
              << FLAGS_num_frames;
    fuser_ptr->num_frames_to_integrate_ = FLAGS_num_frames;
  }
  // Output paths
  if (!gflags::GetCommandLineFlagInfoOrDie("timing_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: timing_output_path = "
              << FLAGS_timing_output_path;
    fuser_ptr->timing_output_path_ = FLAGS_timing_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: tsdf_output_path = "
              << FLAGS_tsdf_output_path;
    fuser_ptr->tsdf_output_path_ = FLAGS_tsdf_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupancy_output_path")
           .is_default) {
    LOG(INFO) << "Command line parameter found: occupancy_output_path = "
              << FLAGS_occupancy_output_path;
    fuser_ptr->occupancy_output_path_ = FLAGS_occupancy_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("freespace_output_path")
           .is_default) {
    LOG(INFO) << "Command line parameter found: freespace_output_path = "
              << FLAGS_freespace_output_path;
    fuser_ptr->freespace_output_path_ = FLAGS_freespace_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_output_path = "
              << FLAGS_esdf_output_path;
    fuser_ptr->esdf_output_path_ = FLAGS_esdf_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: mesh_output_path = "
              << FLAGS_mesh_output_path;
    fuser_ptr->mesh_output_path_ = FLAGS_mesh_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("map_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: map_output_path = "
              << FLAGS_map_output_path;
    fuser_ptr->map_output_path_ = FLAGS_map_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dynamic_overlay_path").is_default) {
    LOG(INFO) << "Command line parameter found: dynamic_overlay_path = "
              << FLAGS_dynamic_overlay_path;
    fuser_ptr->dynamic_overlay_path_ = FLAGS_dynamic_overlay_path;
  }
  // Subsampling flags
  if (!gflags::GetCommandLineFlagInfoOrDie("projective_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: projective_frame_subsampling = "
              << FLAGS_projective_frame_subsampling;
    fuser_ptr->projective_frame_subsampling_ =
        FLAGS_projective_frame_subsampling;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("color_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: color_frame_subsampling = "
              << FLAGS_color_frame_subsampling;
    fuser_ptr->color_frame_subsampling_ = FLAGS_color_frame_subsampling;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: mesh_frame_subsampling = "
              << FLAGS_mesh_frame_subsampling;
    fuser_ptr->mesh_frame_subsampling_ = FLAGS_mesh_frame_subsampling;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_frame_subsampling = "
              << FLAGS_esdf_frame_subsampling;
    fuser_ptr->esdf_frame_subsampling_ = FLAGS_esdf_frame_subsampling;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("frame_rate").is_default) {
    LOG(INFO) << "command line parameter found: "
                 "frame_rate = "
              << FLAGS_frame_rate;
    constexpr int kSecondsToMilliSeconds = 1e3;
    fuser_ptr->frame_period_ms_ =
        nvblox::Time(kSecondsToMilliSeconds / FLAGS_frame_rate);
  }
}

}  // namespace nvblox
