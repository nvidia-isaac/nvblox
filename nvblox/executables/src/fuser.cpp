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
#include "nvblox/executables/fuser.h"
#include <gflags/gflags.h>
#include "nvblox/gflags_param_loading/fuser_params_from_gflags.h"
#include "nvblox/gflags_param_loading/mapper_params_from_gflags.h"
#include "nvblox/utils/logging.h"

#include "nvblox/core/parameter_tree.h"
#include "nvblox/executables/fuser.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/utils/rates.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

constexpr float kDefaultDynamicIntegrationDistanceM = 4.0f;
DEFINE_double(dynamic_integrator_max_integration_distance_m,
              kDefaultDynamicIntegrationDistanceM,
              "Maximum distance (in meters) from the camera at which to "
              "integrate data into the dynamic occupancy grid.");

Fuser::Fuser(std::unique_ptr<datasets::RgbdDataLoaderInterface>&& data_loader)
    : data_loader_(std::move(data_loader)) {
  // Get the params needed to create the multi_mapper
  MultiMapper::Params multi_mapper_params;
  get_multi_mapper_params_from_gflags(&voxel_size_m_, &mapping_type_,
                                      &esdf_mode_, &multi_mapper_params);

  // Create the multi mapper
  // NOTE(remos): Mesh integration is not implemented for occupancy layers.
  multi_mapper_ = std::make_shared<MultiMapper>(
      voxel_size_m_, mapping_type_, esdf_mode_, MemoryType::kDevice);
  multi_mapper_->setMultiMapperParams(multi_mapper_params);

  // Init fuser params
  set_fuser_params_from_gflags(this);

  // Init mapper params (for the two mapper held by the multi mapper)
  MapperParams mapper_params = get_mapper_params_from_gflags();
  // Set the same params for both mappers for now
  multi_mapper_->setMapperParams(mapper_params, mapper_params);

  // NOTE(remos): We default the integration distance for the dynamic mapper to
  //              4 m to limit the computation time of the depth integration.
  LOG(INFO) << "Setting dynamic occupancy max integration distance to "
            << FLAGS_dynamic_integrator_max_integration_distance_m << " m.";
  multi_mapper_.get()
      ->masked_mapper()
      ->occupancy_integrator()
      .max_integration_distance_m(
          FLAGS_dynamic_integrator_max_integration_distance_m);

  // Dump parameters to the console before mapping starts
  LOG(INFO) << "\n\n-------------\nnvblox mapper parameters\n-------------\n"
            << multi_mapper_->getParametersAsString() << "-------------\n\n";
};

int Fuser::run() {
  LOG(INFO) << "Trying to integrate the first frame: ";
  if (!integrateFrames()) {
    LOG(FATAL)
        << "Failed to integrate first frame. Please check the file path.";
    return 1;
  }

  if (!occupancy_output_path_.empty()) {
    if (mapping_type_ == MappingType::kStaticOccupancy) {
      LOG(INFO) << "Outputting occupancy pointcloud ply file to "
                << occupancy_output_path_;
      outputOccupancyPointcloudPly();
    } else {
      LOG(ERROR) << "Occupancy pointcloud can not be stored to "
                 << occupancy_output_path_
                 << " because occupancy wasn't selected for static mapping.";
    }
  }

  if (!tsdf_output_path_.empty()) {
    if (mapping_type_ == MappingType::kStaticTsdf ||
        mapping_type_ == MappingType::kDynamic) {
      LOG(INFO) << "Outputting tsdf pointcloud ply file to "
                << tsdf_output_path_;
      outputTsdfPointcloudPly();
    } else {
      LOG(ERROR) << "TSDF pointcloud can not be stored to " << tsdf_output_path_
                 << " because tsdf wasn't selected for static mapping.";
    }
  }

  if (!mesh_output_path_.empty()) {
    LOG(INFO) << "Generating the mesh.";
    multi_mapper_->updateMesh();
    LOG(INFO) << "Outputting mesh ply file to " << mesh_output_path_;
    outputMeshPly();
  }

  if (!esdf_output_path_.empty()) {
    LOG(INFO) << "Generating the ESDF.";
    multi_mapper_->updateEsdf();
    LOG(INFO) << "Outputting ESDF pointcloud ply file to " << esdf_output_path_;
    outputESDFPointcloudPly();
  }

  if (!freespace_output_path_.empty()) {
    if (mapping_type_ == MappingType::kDynamic) {
      LOG(INFO) << "Outputting freespace pointcloud ply file to "
                << freespace_output_path_;
      outputFreespacePointcloudPly();
    } else {
      LOG(ERROR) << "Freespace pointcloud can not be stored to "
                 << freespace_output_path_
                 << " because kDynamic was not selected as mapping type.";
    }
  }

  if (!map_output_path_.empty()) {
    LOG(INFO) << "Outputting the serialized map to " << map_output_path_;
    outputMapToFile();
  }

  LOG(INFO) << nvblox::timing::Timing::Print() << "\n";
  LOG(INFO) << nvblox::timing::Rates::Print() << "\n";

  if (!timing_output_path_.empty()) {
    LOG(INFO) << "Writing timings to file.";
    outputTimingsToFile();
  }

  return 0;
}

bool Fuser::integrateFrames() {
  int frame_number = 0;
  while (frame_number < num_frames_to_integrate_ &&
         integrateFrame(frame_number++)) {
    timing::mark("Frame " + std::to_string(frame_number - 1), Color::Red());
    LOG(INFO) << "Integrating frame " << frame_number - 1;
  }
  LOG(INFO) << "Ran out of data at frame: " << frame_number - 1;
  return true;
}

bool Fuser::integrateFrame(const int frame_number) {
  timing::Rates::tick("fuser/integrate_frame");
  timing::Timer timer_file("fuser/file_loading");
  DepthImage depth_frame(MemoryType::kDevice);
  ColorImage color_frame(MemoryType::kDevice);
  Transform T_L_C;
  Camera camera;
  const datasets::DataLoadResult load_result =
      data_loader_->loadNext(&depth_frame, &T_L_C, &camera, &color_frame);
  timer_file.Stop();

  if (load_result == datasets::DataLoadResult::kBadFrame) {
    return true;  // Bad data but keep going
  }
  if (load_result == datasets::DataLoadResult::kNoMoreData) {
    return false;  // Shows over folks
  }

  // Depth integration
  timing::Timer per_frame_timer("fuser/time_per_frame");
  if ((frame_number + 1) % projective_frame_subsampling_ == 0) {
    timing::Timer timer_integrate("fuser/integrate_depth");
    timing::Rates::tick("fuser/integrate_depth");

    // Do the actual depth integration
    nvblox::Time time(frame_number * frame_period_ms_);
    multi_mapper_->integrateDepth(depth_frame, T_L_C, camera, time);
    timer_integrate.Stop();

    // Store the dynamic mask if required
    if (mapping_type_ == MappingType::kDynamic &&
        !dynamic_overlay_path_.empty()) {
      outputDynamicOverlayImage(frame_number);
    }
  }

  // Color integration
  if ((frame_number + 1) % color_frame_subsampling_ == 0) {
    timing::Timer timer_integrate_color("fuser/integrate_color");
    timing::Rates::tick("fuser/integrate_color");
    multi_mapper_->integrateColor(color_frame, T_L_C, camera);
    timer_integrate_color.Stop();
  }

  // Mesh update
  if (mesh_frame_subsampling_ > 0) {
    if ((frame_number + 1) % mesh_frame_subsampling_ == 0) {
      timing::Timer timer_mesh("fuser/mesh");
      timing::Rates::tick("fuser/mesh");
      multi_mapper_->updateMesh();
    }
  }

  // Esdf update
  if (esdf_frame_subsampling_ > 0) {
    if ((frame_number + 1) % esdf_frame_subsampling_ == 0) {
      timing::Timer timer_integrate_esdf("fuser/integrate_esdf");
      timing::Rates::tick("fuser/integrate_esdf");
      multi_mapper_->updateEsdf();
      timer_integrate_esdf.Stop();
    }
  }

  per_frame_timer.Stop();

  return true;
}

Mapper& Fuser::static_mapper() {
  return *multi_mapper_.get()->unmasked_mapper();
}

bool Fuser::outputDynamicOverlayImage(int frame_number) {
  timing::Timer timer_write("fuser/dynamic_mask/write");
  std::string full_path = dynamic_overlay_path_ + "/overlay_" +
                          std::to_string(frame_number) + ".png";
  return io::writeToPng(full_path,
                        multi_mapper_->getLastDynamicFrameMaskOverlay());
}

bool Fuser::outputTsdfPointcloudPly() {
  timing::Timer timer_write("fuser/tsdf/write");
  return static_mapper().saveTsdfAsPly(tsdf_output_path_);
}

bool Fuser::outputOccupancyPointcloudPly() {
  timing::Timer timer_write("fuser/occupancy/write");
  return static_mapper().saveOccupancyAsPly(occupancy_output_path_);
}

bool Fuser::outputFreespacePointcloudPly() {
  timing::Timer timer_write("fuser/freespace/write");
  return static_mapper().saveFreespaceAsPly(freespace_output_path_);
}

bool Fuser::outputESDFPointcloudPly() {
  timing::Timer timer_write("fuser/esdf/write");
  return static_mapper().saveEsdfAsPly(esdf_output_path_);
}

bool Fuser::outputMeshPly() {
  timing::Timer timer_write("fuser/mesh/write");
  return static_mapper().saveMeshAsPly(mesh_output_path_);
}

bool Fuser::outputTimingsToFile() {
  LOG(INFO) << "Writing timing to: " << timing_output_path_;
  std::ofstream timing_file(timing_output_path_);
  timing_file << nvblox::timing::Timing::Print();
  timing_file.close();
  return true;
}

bool Fuser::outputMapToFile() {
  timing::Timer timer_serialize("fuser/map/write");
  return static_mapper().saveLayerCake(map_output_path_);
}

}  //  namespace nvblox
