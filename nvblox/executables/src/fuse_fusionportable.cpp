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

#include "nvblox/datasets/fusionportable.h"
#include "nvblox/executables/fuser_lidar.h"

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
    base_path = "../tests/data/fusionportable";
    LOG(WARNING)
        << "No FusionProtable file path specified; defaulting to the test "
           "directory.";
  } else {
    base_path = argv[1];
    LOG(INFO) << "Loading FusionPortable files from " << base_path;
  }

  // Fuser
  // NOTE(alexmillane): Hardcode the sequence ID.
  constexpr int seq_id = 1;
  std::unique_ptr<FuserLidar> fuser_lidar =
      datasets::fusionportable::createFuser(base_path, seq_id);

  // Mesh location (optional)
  if (argc >= 3) {
    fuser_lidar->mesh_output_path_ = argv[2];
    LOG(INFO) << "Mesh location:" << fuser_lidar->mesh_output_path_;
  }

  // NOTE(jjiao): set extrinsics from the base_link to the camera
  Eigen::Quaternionf Qcb(0.500292, 0.490181, -0.508467, 0.500889);
  Eigen::Vector3f tcb(0.067436, -0.022029, -0.078333);
  Eigen::Quaternionf Qbc = Qcb.inverse();
  Eigen::Vector3f tbc = -(Qbc * tcb);

  Eigen::Matrix4f Tbc = Eigen::Matrix4f::Identity();
  Tbc.block<3, 3>(0, 0) = Qbc.toRotationMatrix();
  Tbc.block<3, 1>(0, 3) = tbc;
  fuser_lidar->T_B_C_ = Transform(Tbc);

  // std::cout << fuser_lidar->T_B_C_.matrix() << std::endl;
  // std::cout << std::endl;
  // std::cout << Tbc << std::endl;

  // Make sure the layers are the correct resolution.
  return fuser_lidar->run();
}
