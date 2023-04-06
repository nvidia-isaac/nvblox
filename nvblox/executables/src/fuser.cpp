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
#include "nvblox/utils/logging.h"

#include "nvblox/executables/fuser.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/utils/timing.h"

// Layer params
DEFINE_double(voxel_size, 0.0f, "Voxel resolution in meters.");
DEFINE_bool(use_occupancy_layer, false,
            "Whether to use an occupancy grid for projective integration. If "
            "the flag is set to false a tsdf layer is used.");

// Dataset flags
DEFINE_int32(num_frames, -1,
             "Number of frames to process. Empty means process all.");

// The output paths
DEFINE_string(timing_output_path, "",
              "File in which to save the timing results.");
DEFINE_string(tsdf_output_path, "",
              "File in which to save the TSDF pointcloud.");
DEFINE_string(occupancy_output_path, "",
              "File in which to save the occupancy pointcloud.");
DEFINE_string(esdf_output_path, "",
              "File in which to save the ESDF pointcloud.");
DEFINE_string(mesh_output_path, "", "File in which to save the surface mesh.");
DEFINE_string(map_output_path, "", "File in which to save the serialize map.");

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

// Projective Integrator settings (TSDF and occupancy)
DEFINE_double(projective_integrator_max_integration_distance_m, -1.0,
              "Maximum distance (in meters) from the camera at which to "
              "integrate data into the TSDF or occupancy grid.");
DEFINE_double(projective_integrator_truncation_distance_vox, -1.0,
              "Truncation band (in voxels).");
DEFINE_double(
    tsdf_integrator_max_weight, -1.0,
    "The maximum weight that a tsdf voxel can accumulate through integration.");
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
              "Half the width of the region which is consided as occupied.");

// Mesh integrator settings
DEFINE_double(mesh_integrator_min_weight, -1.0,
              "The minimum weight a tsdf voxel must have before it is meshed.");
DEFINE_bool(mesh_integrator_weld_vertices, true,
            "Whether or not to weld duplicate vertices in the mesh.");

// Color integrator settings
DEFINE_double(color_integrator_max_integration_distance_m, -1.0,
              "Maximum distance (in meters) from the camera at which to "
              "integrate color into the voxel grid.");

// ESDF Integrator settings
DEFINE_double(esdf_integrator_min_weight, -1.0,
              "The minimum weight at which to consider a voxel a site.");
DEFINE_double(esdf_integrator_max_site_distance_vox, -1.0,
              "The maximum distance at which we consider a TSDF voxel a site.");
DEFINE_double(esdf_integrator_max_distance_m, -1.0,
              "The maximum distance which we integrate ESDF distances out to.");

// Integrator weighting scheme
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

namespace nvblox {

Fuser::Fuser(std::unique_ptr<datasets::RgbdDataLoaderInterface>&& data_loader)
    : data_loader_(std::move(data_loader)) {
  // NOTE(alexmillane): We require the voxel size and projective layer variant
  // before we construct the mapper, so we grab this parameters first and
  // separately.
  if (FLAGS_use_occupancy_layer) {
    projective_layer_type_ = ProjectiveLayerType::kOccupancy;
    LOG(INFO) << "Projective layer variant = Occupancy\n"
                 "Attention: ESDF and Mesh integration is not yet implemented "
                 "for occupancy.";
  } else {
    projective_layer_type_ = ProjectiveLayerType::kTsdf;
    LOG(INFO) << "Projective layer variant = TSDF"
                 " (for occupancy set the use_occupancy_layer flag)";
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("voxel_size").is_default) {
    LOG(INFO) << "Command line parameter found: voxel_size = "
              << FLAGS_voxel_size;
    setVoxelSize(static_cast<float>(FLAGS_voxel_size));
  }

  // Initialize the mapper
  mapper_ = std::make_unique<Mapper>(voxel_size_m_, MemoryType::kDevice,
                                     projective_layer_type_);

  // Default parameters
  mapper_->color_integrator().max_integration_distance_m(5.0f);
  mapper_->tsdf_integrator().max_integration_distance_m(5.0f);
  mapper_->tsdf_integrator().view_calculator().raycast_subsampling_factor(4);
  mapper_->occupancy_integrator().max_integration_distance_m(5.0f);
  mapper_->occupancy_integrator().view_calculator().raycast_subsampling_factor(
      4);
  mapper_->esdf_integrator().max_distance_m(4.0f);
  mapper_->esdf_integrator().min_weight(2.0f);

  // Pick commands off the command line
  readCommandLineFlags();
};

void Fuser::readCommandLineFlags() {
  // Dataset flags
  if (!gflags::GetCommandLineFlagInfoOrDie("num_frames").is_default) {
    LOG(INFO) << "Command line parameter found: num_frames = "
              << FLAGS_num_frames;
    num_frames_to_integrate_ = FLAGS_num_frames;
  }
  // Output paths
  if (!gflags::GetCommandLineFlagInfoOrDie("timing_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: timing_output_path = "
              << FLAGS_timing_output_path;
    timing_output_path_ = FLAGS_timing_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: tsdf_output_path = "
              << FLAGS_tsdf_output_path;
    tsdf_output_path_ = FLAGS_tsdf_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupancy_output_path")
           .is_default) {
    LOG(INFO) << "Command line parameter found: occupancy_output_path = "
              << FLAGS_occupancy_output_path;
    occupancy_output_path_ = FLAGS_occupancy_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_output_path = "
              << FLAGS_esdf_output_path;
    esdf_output_path_ = FLAGS_esdf_output_path;
    setEsdfMode(Mapper::EsdfMode::k3D);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: mesh_output_path = "
              << FLAGS_mesh_output_path;
    mesh_output_path_ = FLAGS_mesh_output_path;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("map_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: map_output_path = "
              << FLAGS_map_output_path;
    map_output_path_ = FLAGS_map_output_path;
  }
  // Subsampling flags
  if (!gflags::GetCommandLineFlagInfoOrDie("projective_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: projective_frame_subsampling = "
              << FLAGS_projective_frame_subsampling;
    setProjectiveFrameSubsampling(FLAGS_projective_frame_subsampling);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("color_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: color_frame_subsampling = "
              << FLAGS_color_frame_subsampling;
    setColorFrameSubsampling(FLAGS_color_frame_subsampling);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: mesh_frame_subsampling = "
              << FLAGS_mesh_frame_subsampling;
    setMeshFrameSubsampling(FLAGS_mesh_frame_subsampling);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_frame_subsampling = "
              << FLAGS_esdf_frame_subsampling;
    setEsdfFrameSubsampling(FLAGS_esdf_frame_subsampling);
  }
  // Projective integrator
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_max_integration_distance_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_max_integration_distance_m= "
              << FLAGS_projective_integrator_max_integration_distance_m;
    mapper_->tsdf_integrator().max_integration_distance_m(
        FLAGS_projective_integrator_max_integration_distance_m);
    mapper_->occupancy_integrator().max_integration_distance_m(
        FLAGS_projective_integrator_max_integration_distance_m);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "projective_integrator_truncation_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "projective_integrator_truncation_distance_vox = "
              << FLAGS_projective_integrator_truncation_distance_vox;
    mapper_->tsdf_integrator().truncation_distance_vox(
        FLAGS_projective_integrator_truncation_distance_vox);
    mapper_->occupancy_integrator().truncation_distance_vox(
        FLAGS_projective_integrator_truncation_distance_vox);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_integrator_max_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: tsdf_integrator_max_weight = "
              << FLAGS_tsdf_integrator_max_weight;
    mapper_->tsdf_integrator().max_weight(FLAGS_tsdf_integrator_max_weight);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("free_region_occupancy_probability")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: free_region_occupancy_probability = "
        << FLAGS_free_region_occupancy_probability;
    mapper_->occupancy_integrator().free_region_occupancy_probability(
        FLAGS_free_region_occupancy_probability);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "occupied_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "occupied_region_occupancy_probability = "
              << FLAGS_occupied_region_occupancy_probability;
    mapper_->occupancy_integrator().occupied_region_occupancy_probability(
        FLAGS_occupied_region_occupancy_probability);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "unobserved_region_occupancy_probability")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "unobserved_region_occupancy_probability = "
              << FLAGS_unobserved_region_occupancy_probability;
    mapper_->occupancy_integrator().unobserved_region_occupancy_probability(
        FLAGS_unobserved_region_occupancy_probability);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("occupied_region_half_width_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: occupied_region_half_width_m = "
              << FLAGS_occupied_region_half_width_m;
    mapper_->occupancy_integrator().occupied_region_half_width_m(
        FLAGS_occupied_region_half_width_m);
  }
  // Mesh integrator
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: mesh_integrator_min_weight = "
              << FLAGS_mesh_integrator_min_weight;
    mapper_->mesh_integrator().min_weight(FLAGS_mesh_integrator_min_weight);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("mesh_integrator_weld_vertices")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: mesh_integrator_weld_vertices = "
        << FLAGS_mesh_integrator_weld_vertices;
    mapper_->mesh_integrator().weld_vertices(
        FLAGS_mesh_integrator_weld_vertices);
  }
  // Color integrator
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "color_integrator_max_integration_distance_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "color_integrator_max_integration_distance_m = "
              << FLAGS_color_integrator_max_integration_distance_m;
    mapper_->color_integrator().max_integration_distance_m(
        FLAGS_color_integrator_max_integration_distance_m);
  }
  // ESDF integrator
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_min_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: esdf_integrator_min_weight = "
              << FLAGS_esdf_integrator_min_weight;
    mapper_->esdf_integrator().min_weight(FLAGS_esdf_integrator_min_weight);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "esdf_integrator_max_site_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "esdf_integrator_max_site_distance_vox = "
              << FLAGS_esdf_integrator_max_site_distance_vox;
    mapper_->esdf_integrator().max_site_distance_vox(
        FLAGS_esdf_integrator_max_site_distance_vox);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_integrator_max_distance_m")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: esdf_integrator_max_distance_m = "
        << FLAGS_esdf_integrator_max_distance_m;
    mapper_->esdf_integrator().max_distance_m(
        FLAGS_esdf_integrator_max_distance_m);
  }

  // Weighting scheme
  int num_weighting_schemes_requested = 0;
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_constant")
           .is_default) {
    LOG(INFO) << "Command line parameter found: weighting_scheme_constant = "
              << FLAGS_weighting_scheme_constant;
    mapper_->tsdf_integrator().weighting_function_type(
        WeightingFunctionType::kConstantWeight);
    mapper_->color_integrator().weighting_function_type(
        WeightingFunctionType::kConstantWeight);
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_constant_dropoff")
           .is_default) {
    LOG(INFO)
        << "Command line parameter found: weighting_scheme_constant_dropoff = "
        << FLAGS_weighting_scheme_constant_dropoff;
    mapper_->tsdf_integrator().weighting_function_type(
        WeightingFunctionType::kConstantDropoffWeight);
    mapper_->color_integrator().weighting_function_type(
        WeightingFunctionType::kConstantDropoffWeight);
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("weighting_scheme_inverse_square")
           .is_default) {
    LOG(INFO) << "Command line parameter found: weighting_scheme_square = "
              << FLAGS_weighting_scheme_inverse_square;
    mapper_->tsdf_integrator().weighting_function_type(
        WeightingFunctionType::kInverseSquareWeight);
    mapper_->color_integrator().weighting_function_type(
        WeightingFunctionType::kInverseSquareWeight);
    ++num_weighting_schemes_requested;
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "weighting_scheme_inverse_square_dropoff")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "weighting_scheme_inverse_square_dropoff = "
              << FLAGS_weighting_scheme_inverse_square_dropoff;
    mapper_->tsdf_integrator().weighting_function_type(
        WeightingFunctionType::kInverseSquareDropoffWeight);
    mapper_->color_integrator().weighting_function_type(
        WeightingFunctionType::kInverseSquareDropoffWeight);
    ++num_weighting_schemes_requested;
  }
  CHECK_LT(num_weighting_schemes_requested, 2)
      << "You requested two weighting schemes on the command line. Maximum "
         "one.";
}

int Fuser::run() {
  LOG(INFO) << "Trying to integrate the first frame: ";
  if (!integrateFrames()) {
    LOG(FATAL)
        << "Failed to integrate first frame. Please check the file path.";
    return 1;
  }

  if (!occupancy_output_path_.empty()) {
    if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
      LOG(INFO) << "Outputting occupancy pointcloud ply file to "
                << occupancy_output_path_;
      outputOccupancyPointcloudPly();
    } else {
      LOG(ERROR)
          << "Occupancy pointcloud can not be stored to "
          << occupancy_output_path_
          << " because occupancy wasn't selected as projective layer variant.\n"
             "Please set the use_occupancy_layer flag for an occupancy output.";
    }
  }

  if (!tsdf_output_path_.empty()) {
    if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
      LOG(INFO) << "Outputting tsdf pointcloud ply file to "
                << tsdf_output_path_;
      outputTsdfPointcloudPly();
    } else {
      LOG(ERROR)
          << "TSDF pointcloud can not be stored to " << tsdf_output_path_
          << " because tsdf wasn't selected as projective layer variant.\n"
             "Please leave/set the use_occupancy_layer flag false for an tsdf "
             "output.";
    }
  }

  if (!mesh_output_path_.empty()) {
    LOG(INFO) << "Generating the mesh.";
    mapper_->updateMesh();
    LOG(INFO) << "Outputting mesh ply file to " << mesh_output_path_;
    outputMeshPly();
  }

  if (!esdf_output_path_.empty()) {
    LOG(INFO) << "Generating the ESDF.";
    updateEsdf();
    LOG(INFO) << "Outputting ESDF pointcloud ply file to " << esdf_output_path_;
    outputESDFPointcloudPly();
  }

  if (!map_output_path_.empty()) {
    LOG(INFO) << "Outputting the serialized map to " << map_output_path_;
    outputMapToFile();
  }

  LOG(INFO) << nvblox::timing::Timing::Print();

  LOG(INFO) << "Writing timings to file.";
  outputTimingsToFile();

  return 0;
}

Mapper& Fuser::mapper() { return *mapper_; }

void Fuser::setVoxelSize(float voxel_size) { voxel_size_m_ = voxel_size; }

void Fuser::setProjectiveFrameSubsampling(int subsample) {
  projective_frame_subsampling_ = subsample;
}

void Fuser::setColorFrameSubsampling(int subsample) {
  color_frame_subsampling_ = subsample;
}

void Fuser::setMeshFrameSubsampling(int subsample) {
  mesh_frame_subsampling_ = subsample;
}

void Fuser::setEsdfFrameSubsampling(int subsample) {
  esdf_frame_subsampling_ = subsample;
}

void Fuser::setEsdfMode(Mapper::EsdfMode esdf_mode) {
  if (esdf_mode_ != Mapper::EsdfMode::kUnset) {
    LOG(WARNING) << "EsdfMode already set. Cannot change once set. Not "
                    "doing anything.";
  }
  esdf_mode_ = esdf_mode;
}

bool Fuser::integrateFrame(const int frame_number) {
  timing::Timer timer_file("fuser/file_loading");
  DepthImage depth_frame;
  ColorImage color_frame;
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

  timing::Timer per_frame_timer("fuser/time_per_frame");
  if ((frame_number + 1) % projective_frame_subsampling_ == 0) {
    timing::Timer timer_integrate("fuser/projective_integration");
    mapper_->integrateDepth(depth_frame, T_L_C, camera);
    timer_integrate.Stop();
  }

  if ((frame_number + 1) % color_frame_subsampling_ == 0) {
    timing::Timer timer_integrate_color("fuser/integrate_color");
    mapper_->integrateColor(color_frame, T_L_C, camera);
    timer_integrate_color.Stop();
  }

  if (mesh_frame_subsampling_ > 0) {
    if ((frame_number + 1) % mesh_frame_subsampling_ == 0) {
      timing::Timer timer_mesh("fuser/mesh");
      mapper_->updateMesh();
    }
  }

  if (esdf_frame_subsampling_ > 0) {
    if ((frame_number + 1) % esdf_frame_subsampling_ == 0) {
      timing::Timer timer_integrate_esdf("fuser/integrate_esdf");
      updateEsdf();
      timer_integrate_esdf.Stop();
    }
  }

  per_frame_timer.Stop();

  return true;
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

void Fuser::updateEsdf() {
  switch (esdf_mode_) {
    case Mapper::EsdfMode::kUnset:
      break;
    case Mapper::EsdfMode::k3D:
      mapper_->updateEsdf();
      break;
    case Mapper::EsdfMode::k2D:
      mapper_->updateEsdfSlice(z_min_, z_max_, z_slice_);
      break;
  }
}

bool Fuser::outputTsdfPointcloudPly() {
  timing::Timer timer_write("fuser/tsdf/write");
  return io::outputVoxelLayerToPly(mapper_->tsdf_layer(), tsdf_output_path_);
}

bool Fuser::outputOccupancyPointcloudPly() {
  timing::Timer timer_write("fuser/occupancy/write");
  return io::outputVoxelLayerToPly(mapper_->occupancy_layer(),
                                   occupancy_output_path_);
}

bool Fuser::outputESDFPointcloudPly() {
  timing::Timer timer_write("fuser/esdf/write");
  return io::outputVoxelLayerToPly(mapper_->esdf_layer(), esdf_output_path_);
}

bool Fuser::outputMeshPly() {
  timing::Timer timer_write("fuser/mesh/write");
  return io::outputMeshLayerToPly(mapper_->mesh_layer(), mesh_output_path_);
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
  return mapper_->saveMap(map_output_path_);
}

}  //  namespace nvblox
