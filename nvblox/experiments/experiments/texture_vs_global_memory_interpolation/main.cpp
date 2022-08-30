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
#include <glog/logging.h>

#include "nvblox/utils/timing.h"

#include "nvblox/experiments/common/fuse_3dmatch.h"
#include "nvblox/experiments/integrators/experimental_projective_tsdf_integrators.h"

DECLARE_bool(alsologtostderr);

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  auto fuser =
      nvblox::experiments::Fuser::createFromCommandLineArgs(argc, argv);

  // Run texture-based interpolation experiment
  *fuser.mapper().tsdf_integrator_ptr() = nvblox::experiments::ProjectiveTsdfIntegratorExperimentsTexture>();
  int res_flag = fuser.run();
  if (res_flag != 0) {
    return res_flag;
  }
  const std::string texture_based_timing = nvblox::timing::Timing::Print();
  nvblox::timing::Timing::Reset();

  // Run global-memory-based interpolation experiment
  fuser.tsdf_integrator_ = std::make_shared<
      nvblox::experiments::ProjectiveTsdfIntegratorExperimentsGlobal>();
  res_flag = fuser.run();
  const std::string global_based_timing = nvblox::timing::Timing::Print();

  // Printing the results
  std::cout << "\n\n\nTexture-based interpolation timings" << std::endl;
  std::cout << "-----------" << std::endl;
  std::cout << texture_based_timing << std::endl;
  std::cout << "\n\n\nGlobal-based interpolation timings" << std::endl;
  std::cout << "-----------" << std::endl;
  std::cout << global_based_timing << std::endl;

  return res_flag;
}
