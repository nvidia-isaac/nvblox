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
#include "nvblox/utils/logging.h"

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/executables/fuser.h"

DECLARE_bool(alsologtostderr);

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  // Get the dataset
  std::string base_path;
  if (argc < 2) {
    // Try out running on the test datasets.
    base_path = "../tests/data/3dmatch";
    LOG(WARNING) << "No 3DMatch file path specified; defaulting to the test "
                    "directory.";
  } else {
    base_path = argv[1];
    LOG(INFO) << "Loading 3DMatch files from " << base_path;
  }

  // Fuser
  // NOTE(alexmillane): Hardcode the sequence ID.
  constexpr int seq_id = 1;
  std::unique_ptr<Fuser> fuser =
      datasets::threedmatch::createFuser(base_path, seq_id);
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
