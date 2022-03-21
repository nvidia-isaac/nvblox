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

#include "nvblox/experiments/cake_dynamic.h"
#include "nvblox/experiments/load.h"
#include "nvblox/experiments/user_defined_block.h"

using namespace nvblox;

// Globals for convenience
const float voxel_size = 0.1f;
const MemoryType memory_type = MemoryType::kDevice;

void addRetrieve() {
  std::cout << "addRetrieve()" << std::endl;

  experiments::LayerCakeDynamic cake(voxel_size, memory_type);

  cake.add<TsdfLayer>();
  TsdfLayer* tsdf_layer = cake.getPtr<TsdfLayer>();

  std::cout << "\n\n" << std::endl;
}

void loadLayers() {
  std::cout << "loadLayers()." << std::endl;

  experiments::LayerCakeDynamic cake(voxel_size, memory_type);

  std::string filename = "test.cake";
  experiments::io::load<TsdfLayer, EsdfLayer>(filename, &cake);

  std::cout << "\n\n" << std::endl;
}

void loadCustomLayer() {
  std::cout << "loadCustomLayer()." << std::endl;

  const float block_size = 1.0f;
  experiments::UserDefinedLayer user_defined_layer(block_size, memory_type);

  experiments::LayerCakeDynamic cake(voxel_size, memory_type);

  std::string filename = "test.cake";
  experiments::io::load<experiments::UserDefinedLayer>(filename, &cake);

  std::cout << "\n\n" << std::endl;
}

void buildCake() {
  std::cout << "buildCake()" << std::endl;

  experiments::LayerCakeDynamic cake =
      experiments::LayerCakeDynamic::create<TsdfLayer, ColorLayer, EsdfLayer,
                                            MeshLayer>(voxel_size, memory_type);

  // Static assert
  // NOTE: This creates a static assertion fail due to duplicated LayerTypes.
  // LayerCakeDynamic cake_error =
  //     LayerCakeDynamic::create<TsdfLayer, ColorLayer, TsdfLayer>(voxel_size,
  //                                                                memory_type);

  std::cout << "\n\n" << std::endl;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  addRetrieve();
  loadLayers();
  loadCustomLayer();
  buildCake();

  return 0;
}
