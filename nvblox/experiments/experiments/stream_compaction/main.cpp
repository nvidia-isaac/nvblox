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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/utils/timing.h"

#include "nvblox/experiments/stream_compaction.h"

using namespace nvblox;

DEFINE_int32(num_compactions, 100,
             "How many stream compactions to run and average timings over.");
DEFINE_int32(num_bytes, 1e7, "How many bytes in the data to compact.");
DEFINE_string(timing_output_path, "./timings.txt",
              "File in which to save the timing results.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  // Params
  const int num_trials = FLAGS_num_compactions;
  const int num_bytes = FLAGS_num_bytes;

  // Warmup
  warmupCuda();

  // Data
  const int kNumElements = num_bytes / sizeof(int);
  std::vector<int> data(kNumElements);
  std::iota(data.begin(), data.end(), 0);
  device_vector<int> data_device(data);
  host_vector<int> data_host(kNumElements);
  device_vector<int> data_compact_device(kNumElements);
  host_vector<int> data_compact_host(kNumElements);

  // Stencil
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distribution(0, 1);
  host_vector<bool> stencil(kNumElements);
  std::generate(stencil.begin(), stencil.end(),
                [&]() -> bool { return static_cast<bool>(distribution(gen)); });
  const device_vector<bool> stencil_device = stencil;
  host_vector<bool> stencil_host(kNumElements);

  // Experiment #1
  // - transfer data and stencil device -> host
  // - stream compaction on host
  // - transfer compact stream host -> device
  timing::Timer host_timer("host_compaction");
  for (int trial_idx = 0; trial_idx < num_trials; trial_idx++) {
    // Stecil + data, device -> host
    data_host = data_device;
    stencil_host = stencil_device;
    // Stream compaction on host
    data_compact_host.resize(0);
    for (int i = 0; i < stencil_host.size(); i++) {
      if (stencil_host[i]) {
        data_compact_host.push_back(data_host[i]);
      }
    }
    // Data, host -> device
    data_compact_device = data_compact_host;
  }
  host_timer.Stop();

  // Experiment #2
  // - perform compaction on the device
  timing::Timer device_timer("device_compaction");
  experiments::StreamCompactor stream_compactor;
  for (int trial_idx = 0; trial_idx < num_trials; trial_idx++) {
    stream_compactor.streamCompactionOnGPU(data_device, stencil_device,
                                           &data_compact_device);
  }
  device_timer.Stop();

  std::cout << timing::Timing::Print() << std::endl;
  std::ofstream timing_file(FLAGS_timing_output_path, std::ofstream::out);
  timing_file << timing::Timing::Print();
  timing_file.close();
}