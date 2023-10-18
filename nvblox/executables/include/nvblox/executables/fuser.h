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
#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>

#include "nvblox/datasets/data_loader.h"
#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/image_io.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mapper/multi_mapper.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/utils/logging.h"

namespace nvblox {

class Fuser {
 public:
  Fuser() = default;
  Fuser(std::unique_ptr<datasets::RgbdDataLoaderInterface>&& data_loader);

  // Runs an experiment
  int run();

  // Integrate certain layers.
  bool integrateFrame(const int frame_number);
  bool integrateFrames();

  // Write a dynamic overlay image to disk.
  bool outputDynamicOverlayImage(int frame_number);
  // Output a pointcloud tsdf as PLY file.
  bool outputTsdfPointcloudPly();
  // Output a pointcloud occupancy as PLY file.
  bool outputOccupancyPointcloudPly();
  // Output a pointcloud freespace as PLY file.
  bool outputFreespacePointcloudPly();
  // Output a pointcloud ESDF as PLY file.
  bool outputESDFPointcloudPly();
  // Output a file with the mesh.
  bool outputMeshPly();
  // Output timings to a file
  bool outputTimingsToFile();
  // Output the serialized map to a file
  bool outputMapToFile();

  // Get the static mapper (useful for experiments where we modify mapper
  // settings)
  Mapper& static_mapper();

  // MultiMapper - Contains two mappers
  std::shared_ptr<MultiMapper> multi_mapper_;

  // MultiMapper params
  float voxel_size_m_ = 0.05;
  MappingType mapping_type_ = MappingType::kStaticTsdf;
  EsdfMode esdf_mode_ = EsdfMode::k3D;

  // Dataset settings.
  int num_frames_to_integrate_ = std::numeric_limits<int>::max();
  std::unique_ptr<datasets::RgbdDataLoaderInterface> data_loader_;

  // Subsampling params
  int projective_frame_subsampling_ = 1;
  int color_frame_subsampling_ = 1;
  // By default we just do the mesh and esdf once at the end
  // (if output paths exist)
  int mesh_frame_subsampling_ = -1;
  int esdf_frame_subsampling_ = -1;

  // Param for dynamics
  nvblox::Time frame_period_ms_{33};  // 30 Hz

  // Output paths
  std::string timing_output_path_;
  std::string tsdf_output_path_;
  std::string esdf_output_path_;
  std::string occupancy_output_path_;
  std::string freespace_output_path_;
  std::string mesh_output_path_;
  std::string map_output_path_;
  std::string dynamic_overlay_path_;
};

}  //  namespace nvblox
