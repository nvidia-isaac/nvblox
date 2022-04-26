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
#include "nvblox/experiments/common/fuse_3dmatch.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/datasets/parse_3dmatch.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
namespace experiments {

Fuse3DMatch::Fuse3DMatch(const std::string& base_path,
                         const std::string& timing_output_path,
                         const std::string& mesh_output_path,
                         const std::string& esdf_output_path)
    : base_path_(base_path),
      timing_output_path_(timing_output_path),
      mesh_output_path_(mesh_output_path),
      esdf_output_path_(esdf_output_path),
      mapper_(voxel_size_m_) {
  initializeImageLoaders();
  // Params
  mapper_.mesh_integrator().min_weight() = 2.0f;
  mapper_.color_integrator().max_integration_distance_m(5.0f);
  mapper_.tsdf_integrator().max_integration_distance_m(5.0f);
  mapper_.tsdf_integrator().view_calculator().raycast_subsampling_factor(4);
  mapper_.esdf_integrator().max_distance_m(4.0f);
  mapper_.esdf_integrator().min_weight(2.0f);
};

int Fuse3DMatch::run() {
  LOG(INFO) << "Trying to integrate the first frame: ";
  if (!integrateFrames()) {
    LOG(FATAL)
        << "Failed to integrate first frame. Please check the file path.";
    return 1;
  }

  if (!mesh_output_path_.empty()) {
    LOG(INFO) << "Outputting mesh ply file to " << mesh_output_path_;
    outputMeshPly();
  }

  if (!esdf_output_path_.empty()) {
    LOG(INFO) << "Outputting ESDF pointcloud ply file to " << esdf_output_path_;
    outputPointcloudPly();
  }

  LOG(INFO) << nvblox::timing::Timing::Print();

  LOG(INFO) << "Writing timings to file.";
  outputTimingsToFile();

  return 0;
}
void Fuse3DMatch::setVoxelSize(float voxel_size) { voxel_size_m_ = voxel_size; }
void Fuse3DMatch::setTsdfFrameSubsampling(int subsample) {
  tsdf_frame_subsampling_ = subsample;
}
void Fuse3DMatch::setColorFrameSubsampling(int subsample) {
  color_frame_subsampling_ = subsample;
}
void Fuse3DMatch::setMeshFrameSubsampling(int subsample) {
  mesh_frame_subsampling_ = subsample;
}
void Fuse3DMatch::setEsdfFrameSubsampling(int subsample) {
  esdf_frame_subsampling_ = subsample;
}

void Fuse3DMatch::setBasePath(const std::string& base_path) {
  base_path_ = base_path;
  initializeImageLoaders();
}

void Fuse3DMatch::setSequenceNumber(int sequence_num) {
  sequence_num_ = sequence_num;
  initializeImageLoaders();
}

void Fuse3DMatch::initializeImageLoaders(bool multithreaded) {
  // NOTE(alexmillane): On my desktop the performance of threaded image loading
  // seems to saturate at around 6 threads. On machines with less cores (I have
  // 12 physical) presumably the saturation point will be less.
  if (multithreaded) {
    constexpr unsigned int kMaxLoadingThreads = 6;
    unsigned int num_loading_threads =
        std::min(kMaxLoadingThreads, std::thread::hardware_concurrency() / 2);
    CHECK_GT(num_loading_threads, 0)
        << "std::thread::hardware_concurrency() returned 0";
    LOG(INFO) << "Using " << num_loading_threads
              << " threads for loading images.";
    depth_image_loader_ =
        datasets::threedmatch::createMultithreadedDepthImageLoader(
            base_path_, sequence_num_, num_loading_threads,
            MemoryType::kDevice);
    color_image_loader_ =
        datasets::threedmatch::createMultithreadedColorImageLoader(
            base_path_, sequence_num_, num_loading_threads,
            MemoryType::kDevice);
  } else {
    depth_image_loader_ = datasets::threedmatch::createDepthImageLoader(
        base_path_, sequence_num_, MemoryType::kDevice);
    color_image_loader_ = datasets::threedmatch::createColorImageLoader(
        base_path_, sequence_num_, MemoryType::kDevice);
  }
}

bool Fuse3DMatch::integrateFrame(const int frame_number) {
  timing::Timer timer_file("3dmatch/file_loading");

  // Get the camera for this frame.
  timing::Timer timer_file_intrinsics("file_loading/intrinsics");
  Eigen::Matrix3f camera_intrinsics;
  if (!datasets::threedmatch::parseCameraFromFile(
          datasets::threedmatch::getPathForCameraIntrinsics(base_path_),
          &camera_intrinsics)) {
    return false;
  }
  timer_file_intrinsics.Stop();

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  DepthImage depth_frame;
  if (!depth_image_loader_->getNextImage(&depth_frame)) {
    return false;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  CHECK(color_image_loader_);
  timing::Timer timer_file_color("file_loading/color_image");
  ColorImage color_frame;
  if (!color_image_loader_->getNextImage(&color_frame)) {
    return false;
  }
  timer_file_color.Stop();

  // Get the transform.
  timing::Timer timer_file_camera("file_loading/camera");
  Transform T_L_C;
  if (!datasets::threedmatch::parsePoseFromFile(
          datasets::threedmatch::getPathForFramePose(base_path_, sequence_num_,
                                                     frame_number),
          &T_L_C)) {
    return false;
  }

  // Rotate the world frame since Y is up in the normal 3D match dasets.
  Eigen::Quaternionf q_L_O =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 1, 0), Vector3f(0, 0, 1));
  T_L_C = q_L_O * T_L_C;

  // Create a camera object.
  int image_width = depth_frame.cols();
  int image_height = depth_frame.rows();
  const auto camera = Camera::fromIntrinsicsMatrix(camera_intrinsics,
                                                   image_width, image_height);

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C.matrix().allFinite() || !camera_intrinsics.allFinite() ||
      std::abs(T_L_C.matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad CSV data.";
    return true;  // Bad data, but keep going.
  }

  timer_file_camera.Stop();
  timer_file.Stop();

  timing::Timer per_frame_timer("3dmatch/time_per_frame");
  if ((frame_number + 1) % tsdf_frame_subsampling_ == 0) {
    timing::Timer timer_integrate("3dmatch/integrate_tsdf");
    mapper_.integrateDepth(depth_frame, T_L_C, camera);
    timer_integrate.Stop();
  }

  if ((frame_number + 1) % color_frame_subsampling_ == 0) {
    timing::Timer timer_integrate_color("3dmatch/integrate_color");
    mapper_.integrateColor(color_frame, T_L_C, camera);
    timer_integrate_color.Stop();
  }

  if ((frame_number + 1) % mesh_frame_subsampling_ == 0) {
    timing::Timer timer_mesh("3dmatch/mesh");
    mapper_.updateMesh();
  }

  if ((frame_number + 1) % esdf_frame_subsampling_ == 0) {
    timing::Timer timer_integrate_esdf("3dmatch/integrate_esdf");
    mapper_.updateEsdfSlice(z_min_, z_max_, z_slice_);
    timer_integrate_esdf.Stop();
  }

  per_frame_timer.Stop();

  return true;
}

bool Fuse3DMatch::integrateFrames() {
  int frame_number = 0;
  while (frame_number < num_frames_to_integrate_ &&
         integrateFrame(frame_number++)) {
    timing::mark("Frame " + std::to_string(frame_number - 1), Color::Red());
    LOG(INFO) << "Integrating frame " << frame_number - 1;
  }
  return true;
}

bool Fuse3DMatch::outputPointcloudPly() {
  timing::Timer timer_write("3dmatch/esdf/write");
  return io::outputVoxelLayerToPly(mapper_.esdf_layer(), esdf_output_path_);
}

bool Fuse3DMatch::outputMeshPly() {
  timing::Timer timer_write("3dmatch/mesh/write");
  return io::outputMeshLayerToPly(mapper_.mesh_layer(), mesh_output_path_);
}
bool Fuse3DMatch::outputTimingsToFile() {
  LOG(INFO) << "Writing timing to: " << timing_output_path_;
  std::ofstream timing_file(timing_output_path_);
  timing_file << nvblox::timing::Timing::Print();
  timing_file.close();
  return true;
}

Fuse3DMatch Fuse3DMatch::createFromCommandLineArgs(int argc, char* argv[]) {
  std::string base_path = "";
  std::string timing_output_path = "./3dmatch_timings.txt";
  std::string esdf_output_path = "";
  std::string mesh_output_path = "";
  if (argc < 2) {
    // Try out running on the test datasets.
    base_path = "../tests/data/3dmatch";
    LOG(WARNING) << "No 3DMatch file path specified; defaulting to the test "
                    "directory.";
  } else {
    base_path = argv[1];
    LOG(INFO) << "Loading 3DMatch files from " << base_path;
  }
  // Optionally overwrite the output paths.
  if (argc >= 3) {
    timing_output_path = argv[2];
  }
  if (argc >= 3) {
    esdf_output_path = argv[3];
  }
  if (argc >= 4) {
    mesh_output_path = argv[4];
  }

  nvblox::experiments::Fuse3DMatch fuser(base_path, timing_output_path,
                                         mesh_output_path, esdf_output_path);

  return fuser;
}

}  //  namespace experiments
}  //  namespace nvblox
