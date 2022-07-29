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
#include <glog/logging.h>

#include "nvblox/core/blox.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/layer.h"
#include "nvblox/datasets/3dmatch.h"

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/executables/fuser.h"

DEFINE_bool(single_thread, false, "Load images using a single thread.");
DEFINE_bool(multi_thread, false, "Load images using multiple threads");
DEFINE_int32(num_threads, -1,
             "Number of threads to load images with in multithreaded mode.");

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

  constexpr int seq_id = 1;
  std::unique_ptr<Fuser> fuser =
      datasets::threedmatch::createFuser(base_path, seq_id);

  // Replacing the image loader with the one we are asked to test.
  if (FLAGS_single_thread) {
    constexpr bool multithreaded = false;
    fuser->data_loader_ = std::make_unique<datasets::threedmatch::DataLoader>(
        base_path, seq_id, multithreaded);
  } else {
    constexpr bool multithreaded = true;
    fuser->data_loader_ = std::make_unique<datasets::threedmatch::DataLoader>(
        base_path, seq_id, multithreaded);
  }

  warmupCuda();
  fuser->run();

  return 0;
}
