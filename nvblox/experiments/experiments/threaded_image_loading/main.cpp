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
#include <iostream>

#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>

#include "nvblox/core/blox.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/layer.h"
#include "nvblox/datasets/parse_3dmatch.h"

#include "nvblox/experiments/common/fuse_3dmatch.h"

DEFINE_bool(single_thread, false, "Load images using a single thread.");
DEFINE_bool(multi_thread, false, "Load images using multiple threads");
DEFINE_int32(num_threads, -1,
             "Number of threads to load images with in multithreaded mode.");
DEFINE_int32(num_images, -1, "Number of images to process in this experiment.");
DEFINE_string(timing_output_path, "",
              "File in which to save the timing results.");

DECLARE_bool(alsologtostderr);

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  std::string base_path;
  if (argc < 2) {
    LOG(FATAL) << "Must pass path the 3DMatch dataset root directory.";
    return 0;
  } else {
    LOG(INFO) << "Loading data from: " << argv[1];
    base_path = argv[1];
  }

  CHECK(FLAGS_single_thread || FLAGS_multi_thread)
      << "Must select either single or multi-threaded.";
  if (FLAGS_multi_thread) {
    CHECK_GT(FLAGS_num_threads, 0)
        << "Please specify the number of threads to run with.";
  }

  std::string esdf_output_path = "./3dmatch_esdf_pointcloud.ply";
  std::string mesh_output_path = "./3dmatch_mesh.ply";
  experiments::Fuse3DMatch fuser(base_path, FLAGS_timing_output_path,
                                 mesh_output_path, esdf_output_path);
  if (FLAGS_num_images > 0) {
    fuser.num_frames_to_integrate_ = FLAGS_num_images;
  }

  // Replacing the image loader with the one we are asked to test.
  if (FLAGS_single_thread) {
    fuser.depth_image_loader_ = datasets::threedmatch::createDepthImageLoader(
        base_path, fuser.sequence_num_);
    fuser.color_image_loader_ = datasets::threedmatch::createColorImageLoader(
        base_path, fuser.sequence_num_);
  } else {
    fuser.depth_image_loader_ =
        datasets::threedmatch::createMultithreadedDepthImageLoader(
            base_path, fuser.sequence_num_, FLAGS_num_threads);
    fuser.color_image_loader_ =
        datasets::threedmatch::createMultithreadedColorImageLoader(
            base_path, fuser.sequence_num_, FLAGS_num_threads);
  }

  warmupCuda();
  fuser.run();

  return 0;
}
