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
#include <iostream>

#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>

#include "nvblox/core/cuda/warmup.h"
#include "nvblox/utils/timing.h"

#include "vector_copies.h"

DECLARE_bool(alsologtostderr);

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  warmupCuda();
  // Do a warmup run and then reset the times.
  runVectorCopyExperiments(1);
  nvblox::timing::Timing::Reset();

  runVectorCopyExperiments(1000);

  std::cout << nvblox::timing::Timing::Print();

  // Optionally output to file as well.
  if (argc >= 2) {
    std::ofstream outfile(argv[1]);
    if (outfile.is_open()) {
      outfile << nvblox::timing::Timing::Print();
      outfile.close();
    }
  }

  return 0;
}
