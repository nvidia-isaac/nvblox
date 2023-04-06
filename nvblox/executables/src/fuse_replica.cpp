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

#include <fstream>
#include <sstream>
#include <string>

#include <iomanip>
#include <iostream>

#include <gflags/gflags.h>
#include "nvblox/utils/logging.h"

#include "nvblox/core/types.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/datasets/replica.h"
#include "nvblox/executables/fuser.h"
#include "nvblox/sensors/image.h"

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  // Path to the dataset
  if (argc < 2) {
    LOG(ERROR) << "No path to data directory given, failing.";
    return 1;
  }
  std::string base_path = argv[1];
  LOG(INFO) << "Loading Replica files from " << base_path;

  // Fuser
  std::unique_ptr<Fuser> fuser = datasets::replica::createFuser(base_path);
  if (!fuser) {
    LOG(FATAL) << "Creation of the Fuser failed";
  }

  // Mesh location (optional)
  if (argc >= 3) {
    fuser->mesh_output_path_ = argv[2];
    LOG(INFO) << "Mesh location:" << fuser->mesh_output_path_;
  }

  // Make sure the layers are the correct resolution.
  return fuser->run();
}
