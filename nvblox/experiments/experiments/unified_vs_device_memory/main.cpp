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
#include "nvblox/executables/fuser.h"

DEFINE_bool(unified_memory, false, "Run the test using unified memory");
DEFINE_bool(device_memory, false, "Run the test using device memory");

DECLARE_bool(alsologtostderr);

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  if ((!FLAGS_unified_memory && !FLAGS_device_memory) ||
      (FLAGS_unified_memory && FLAGS_device_memory)) {
    std::cout << "Must select either --unified_memory or --device_memory\n";
    return 0;
  }

  std::string dataset_base_path;
  if (argc < 2) {
    // Try out running on the test datasets.
    dataset_base_path = "../tests/data/3dmatch";
    std::cout << "Please specify 3DMatch file path.\n";
  } else {
    dataset_base_path = argv[1];
    std::cout << "Loading 3DMatch files from " << dataset_base_path << ".\n";
  }

  constexpr int seq_id = 1;
  std::unique_ptr<Fuser> fuser =
      datasets::threedmatch::createFuser(dataset_base_path, seq_id);

  // Replacing the TSDF and colors layers with layers stored in the appropriate
  // memory type
  if (FLAGS_device_memory) {
    fuser->mapper().layers() =
        LayerCake::create<TsdfLayer, ColorLayer, EsdfLayer, MeshLayer>(
            fuser->voxel_size_m_, MemoryType::kDevice);
  } else {
    fuser->mapper().layers() =
        LayerCake::create<TsdfLayer, ColorLayer, EsdfLayer, MeshLayer>(
            fuser->voxel_size_m_, MemoryType::kUnified, MemoryType::kUnified,
            MemoryType::kDevice, MemoryType::kDevice);
  }

  warmupCuda();
  fuser->run();

  return 0;
}