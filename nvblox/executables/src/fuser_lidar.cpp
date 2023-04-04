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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/executables/fuser_lidar.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/utils/timing.h"

#include "nvblox/core/cuda/image_operation.h"

// Layer params
DEFINE_double(voxel_size, 0.0f, "Voxel resolution in meters.");

// Dataset flags
DEFINE_int32(num_frames, -1,
             "Number of frames to process. Empty means process all.");

// The output paths
DEFINE_string(timing_output_path, "",
              "File in which to save the timing results.");
DEFINE_string(esdf_output_path, "",
              "File in which to save the ESDF pointcloud.");
DEFINE_string(mesh_output_path, "", "File in which to save the surface mesh.");
DEFINE_string(map_output_path, "", "File in which to save the serialize map.");
DEFINE_string(obstacle_output_path, "",
              "File in which to save the obstacle pointcloud map.");

// Subsampling
DEFINE_int32(tsdf_frame_subsampling, 0,
             "By what amount to subsample the TSDF frames. A subsample of 3 "
             "means only every 3rd frame is taken.");
DEFINE_int32(color_frame_subsampling, 0,
             "How much to subsample the color integration by.");
DEFINE_int32(mesh_frame_subsampling, 0,
             "How much to subsample the meshing by.");
DEFINE_int32(esdf_frame_subsampling, 0,
             "How much to subsample the ESDF integration by.");

// TSDF Integrator settings
DEFINE_double(tsdf_integrator_max_integration_distance_m, -1.0,
              "Maximum distance (in meters) from the camera at which to "
              "integrate data into the TSDF.");
DEFINE_double(tsdf_integrator_truncation_distance_vox, -1.0,
              "Truncation band (in voxels).");
DEFINE_double(
    tsdf_integrator_max_weight, -1.0,
    "The maximum weight that a tsdf voxel can accumulate through integration.");

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
DEFINE_int32(esdf_mode, 0, "The ESDF mode. 0: k3D, 1: k2D, 2: kUnset");
DEFINE_double(esdf_zmin, 0.5, "zmin of the 2D ESDF map");
DEFINE_double(esdf_zmax, 1.0, "zmax of the 2D ESDF map");
DEFINE_double(esdf_z_slice, 0.75, "z_slice of the 2D ESDF map");

namespace nvblox {
FuserLidar::FuserLidar(
    std::unique_ptr<datasets::RgbdDataLoaderInterface>&& data_loader)
    : data_loader_(std::move(data_loader)) {
  // NOTE(alexmillane): We require the voxel size before we construct the
  // mapper, so we grab this parameter first and separately.
  if (!gflags::GetCommandLineFlagInfoOrDie("voxel_size").is_default) {
    LOG(INFO) << "Command line parameter found: voxel_size = "
              << FLAGS_voxel_size;
    setVoxelSize(static_cast<float>(FLAGS_voxel_size));
  }

  // Initialize the mapper
  mapper_ = std::make_unique<RgbdMapper>(voxel_size_m_);

  // Default parameters
  mapper_->mesh_integrator().min_weight(2.0f);
  mapper_->color_integrator().max_integration_distance_m(5.0f);
  mapper_->lidar_tsdf_integrator().max_integration_distance_m(1.0f);
  mapper_->lidar_tsdf_integrator().view_calculator().raycast_subsampling_factor(
      4);
  mapper_->esdf_integrator().max_distance_m(4.0f);
  mapper_->esdf_integrator().min_weight(2.0f);

  // Pick commands off the command line
  readCommandLineFlags();
};

void FuserLidar::readCommandLineFlags() {
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
  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_output_path = "
              << FLAGS_esdf_output_path;
    esdf_output_path_ = FLAGS_esdf_output_path;
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
  if (!gflags::GetCommandLineFlagInfoOrDie("obstacle_output_path").is_default) {
    LOG(INFO) << "Command line parameter found: obstacle_output_path = "
              << FLAGS_obstacle_output_path;
    obs_output_path_ = FLAGS_obstacle_output_path;
  }

  // Subsampling flags
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_frame_subsampling")
           .is_default) {
    LOG(INFO) << "Command line parameter found: tsdf_frame_subsampling = "
              << FLAGS_tsdf_frame_subsampling;
    setTsdfFrameSubsampling(FLAGS_tsdf_frame_subsampling);
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

  // TSDF integrator
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "tsdf_integrator_max_integration_distance_m")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "tsdf_integrator_max_integration_distance_m= "
              << FLAGS_tsdf_integrator_max_integration_distance_m;
    mapper_->lidar_tsdf_integrator().max_integration_distance_m(
        FLAGS_tsdf_integrator_max_integration_distance_m);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie(
           "tsdf_integrator_truncation_distance_vox")
           .is_default) {
    LOG(INFO) << "Command line parameter found: "
                 "tsdf_integrator_truncation_distance_vox = "
              << FLAGS_tsdf_integrator_truncation_distance_vox;
    mapper_->lidar_tsdf_integrator().truncation_distance_vox(
        FLAGS_tsdf_integrator_truncation_distance_vox);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("tsdf_integrator_max_weight")
           .is_default) {
    LOG(INFO) << "Command line parameter found: tsdf_integrator_max_weight = "
              << FLAGS_tsdf_integrator_max_weight;
    mapper_->lidar_tsdf_integrator().max_weight(
        FLAGS_tsdf_integrator_max_weight);
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

  if (!gflags::GetCommandLineFlagInfoOrDie("esdf_mode").is_default) {
    LOG(INFO) << "Command line parameter found: esdf_mode = "
              << FLAGS_esdf_mode;
    if (FLAGS_esdf_mode == 0) {
      setEsdfMode(RgbdMapper::EsdfMode::k3D);
    } else if (FLAGS_esdf_mode == 1) {
      setEsdfMode(RgbdMapper::EsdfMode::k2D);
      z_min_ = FLAGS_esdf_zmin;
      z_max_ = FLAGS_esdf_zmax;
      z_slice_ = FLAGS_esdf_z_slice;
    } else if (FLAGS_esdf_mode == 2) {
      setEsdfMode(RgbdMapper::EsdfMode::kUnset);
    }
  }
}

// NOTE(gogojjh): the overall procedures running function
int FuserLidar::run() {
  LOG(INFO) << "Trying to integrate the first frame: ";
  if (!integrateFrames()) {
    LOG(FATAL)
        << "Failed to integrate first frame. Please check the file path.";
    return 1;
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
    outputPointcloudPly();
  }

  if (!map_output_path_.empty()) {
    LOG(INFO) << "Outputting the serialized map to " << map_output_path_;
    outputMapToFile();
  }

  if (!obs_output_path_.empty()) {
    LOG(INFO) << "Outputting Obstacle based on the ESDF map ply file to "
              << obs_output_path_;
    outputObstaclePointcloudPly();
  }

  // std::vector<std::string> keywords = {
  //     std::string("integrate"), std::string("normal"), std::string("write")};
  std::vector<std::string> keywords = {std::string("fuser"),
                                       std::string("write")};
  LOG(INFO) << nvblox::timing::Timing::Print(keywords);

  LOG(INFO) << "Writing timings to file.";
  outputTimingsToFile();

  return 0;
}

RgbdMapper& FuserLidar::mapper() { return *mapper_; }

void FuserLidar::setVoxelSize(float voxel_size) { voxel_size_m_ = voxel_size; }

void FuserLidar::setTsdfFrameSubsampling(int subsample) {
  tsdf_frame_subsampling_ = subsample;
}

void FuserLidar::setColorFrameSubsampling(int subsample) {
  color_frame_subsampling_ = subsample;
}

void FuserLidar::setMeshFrameSubsampling(int subsample) {
  mesh_frame_subsampling_ = subsample;
}

void FuserLidar::setEsdfFrameSubsampling(int subsample) {
  esdf_frame_subsampling_ = subsample;
}

void FuserLidar::setEsdfMode(RgbdMapper::EsdfMode esdf_mode) {
  if (esdf_mode_ != RgbdMapper::EsdfMode::kUnset) {
    LOG(WARNING) << "EsdfMode already set. Cannot change once set once. Not "
                    "doing anything.";
  }
  esdf_mode_ = esdf_mode;
}

// NOTE(gogojjh): this function will run the tsdf, mesh, and esdf integration
// for each incoming frame
bool FuserLidar::integrateFrame(const int frame_number) {
  timing::Timer timer_file("fuser/file_loading");
  DepthImage depth_frame;
  DepthImage height_frame;
  ColorImage color_frame;
  Transform T_W_B;
  CameraPinhole camera;
  OSLidar oslidar;
  const datasets::DataLoadResult load_result = data_loader_->loadNext(
      &depth_frame, &T_W_B, &camera, &oslidar, &height_frame, &color_frame);
  timer_file.Stop();

  if (load_result == datasets::DataLoadResult::kBadFrame) {
    LOG(INFO) << "Bad frame: wrong parameters of intrinsics or extrinsics";
    return true;  // Bad data but keep going
  }
  if (load_result == datasets::DataLoadResult::kNoMoreData) {
    LOG(INFO) << "No more data: lack of depth_frame, height_frame, "
                 "or color_frame";
    return false;  // Shows over folks
  }

  timing::Timer per_frame_timer("fuser/time_per_frame");

  if ((frame_number + 1) % tsdf_frame_subsampling_ == 0) {
    oslidar.setDepthFrameCUDA(depth_frame.dataPtr());
    oslidar.setHeightFrameCUDA(height_frame.dataPtr());

    timing::Timer timer_normal("fuser/compute_normal_image");  // 0.7ms
    nvblox::cuda::getNormalImageOSLidar(oslidar);
    timer_normal.Stop();

    timing::Timer timer_integrate("fuser/integrate_tsdf");
    mapper_->integrateOSLidarDepth(depth_frame, T_W_B, oslidar);
    timer_integrate.Stop();

    nvblox::cuda::freeNormalImageOSLidar(oslidar);
  }

  Transform T_W_C = T_W_B * T_B_C_;
  if (color_frame_subsampling_ > 0) {
    if ((frame_number + 1) % color_frame_subsampling_ == 0) {
      timing::Timer timer_integrate_color("fuser/integrate_color");
      mapper_->integrateColor(color_frame, T_W_C, camera);
      timer_integrate_color.Stop();
    }
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

// NOTE(gogojjh): Running all TSDF integrations
bool FuserLidar::integrateFrames() {
  int frame_number = 0;
  while (frame_number < num_frames_to_integrate_ &&
         integrateFrame(frame_number++)) {
    timing::mark("Frame " + std::to_string(frame_number - 1), Color::Red());
    LOG(INFO) << "Integrating frame " << frame_number - 1;
  }
  LOG(INFO) << "Ran out of data at frame: " << frame_number - 1;
  return true;
}

void FuserLidar::updateEsdf() {
  switch (esdf_mode_) {
    case RgbdMapper::EsdfMode::kUnset:
      break;
    case RgbdMapper::EsdfMode::k3D:
      mapper_->updateEsdf();
      break;
    case RgbdMapper::EsdfMode::k2D:
      mapper_->updateEsdfSlice(z_min_, z_max_, z_slice_);
      break;
  }
}

bool FuserLidar::outputPointcloudPly() {
  timing::Timer timer_write("fuser/esdf/write");
  return io::outputVoxelLayerToPly(mapper_->esdf_layer(), esdf_output_path_);
}

bool FuserLidar::outputMeshPly() {
  timing::Timer timer_write("fuser/mesh/write");
  return io::outputMeshLayerToPly(mapper_->mesh_layer(), mesh_output_path_);
}

bool FuserLidar::outputObstaclePointcloudPly() {
  timing::Timer timer_write("fuser/obstacle/write");
  LOG(INFO)
      << "[NOTE] The output of the obstacle point cloud is under construction";
  return io::outputObstacleToPly(mapper_->esdf_layer(), obs_output_path_);
}

bool FuserLidar::outputTimingsToFile() {
  LOG(INFO) << "Writing timing to: " << timing_output_path_;
  std::ofstream timing_file(timing_output_path_);
  timing_file << nvblox::timing::Timing::Print();
  timing_file.close();
  return true;
}

bool FuserLidar::outputMapToFile() {
  timing::Timer timer_serialize("fuser/map/write");
  return mapper_->saveMap(map_output_path_);
}

}  //  namespace nvblox
