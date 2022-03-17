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

#include <iostream>

#include "nvblox/core/common_names.h"

#include "nvblox/experiments/cake_static.h"

using namespace nvblox;

// Globals for convenience
const float voxel_size = 0.1f;
const MemoryType memory_type = MemoryType::kDevice;

void addRetrieve() {
  std::cout << "addRetrieve()" << std::endl;

  experiments::LayerCakeStatic<TsdfLayer> cake_1(voxel_size, memory_type);

  TsdfLayer* tsdf_layer = cake_1.getPtr<TsdfLayer>();
  // EsdfLayer* esdf_layer = cake.getPtr<EsdfLayer>(); // Compile error (missing
  //                                                      layer)

  experiments::LayerCakeStatic<TsdfLayer, ColorLayer, MeshLayer, EsdfLayer>
      cake_2(voxel_size, memory_type);

  tsdf_layer = cake_2.getPtr<TsdfLayer>();
  EsdfLayer* esdf_layer = cake_2.getPtr<EsdfLayer>();

  experiments::LayerCakeStatic<TsdfLayer, TsdfLayer, TsdfLayer, EsdfLayer>
      cake_3(voxel_size, memory_type);
  CHECK_EQ(cake_3.count<TsdfLayer>(), 3);
  // tsdf_layer = cake_3.getPtr<TsdfLayer>(); // Compile error (more than one
  //                                                            TsdfLayer)

  std::cout << "\n\n" << std::endl;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  addRetrieve();

  return 0;
}
