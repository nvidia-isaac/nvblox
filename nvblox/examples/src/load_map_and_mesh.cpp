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

#include "gflags/gflags.h"
#include "nvblox/utils/logging.h"

#include "nvblox/nvblox.h"

#include "nvblox/serialization/internal/layer_type_register.h"

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  CHECK_GT(argc, 0) << "Please pass the path to the map to load.";
  const std::string base_path = argv[1];
  LOG(INFO) << "Loading map from " << base_path;

  // Load
  LayerCake cake = io::loadLayerCakeFromFile(base_path, MemoryType::kDevice);
  LOG(INFO) << "Loaded cake with voxel size: " << cake.voxel_size();

  // Print the names of the loaded layers.
  LOG(INFO) << "Loaded layers:";
  for (const auto& id_layer_pair : cake.get_layers()) {
    LOG(INFO) << "Loaded: "
              << LayerTypeRegister::getLayerName(id_layer_pair.first);
  }

  // Mesh
  LOG(INFO) << "Meshing";
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer(cake.block_size(), MemoryType::kDevice);
  mesh_integrator.integrateMeshFromDistanceField(cake.get<TsdfLayer>(),
                                                 &mesh_layer);
  mesh_integrator.colorMesh(cake.get<ColorLayer>(), &mesh_layer);
  LOG(INFO) << "Done";

  std::string output_path;
  if (argc > 2) {
    output_path = argv[2];
  } else {
    output_path = "./mesh.ply";
  }
  LOG(INFO) << "Writing mesh to: " << output_path;
  CHECK(io::outputMeshLayerToPly(mesh_layer, output_path));
  LOG(INFO) << "Done";

  // Esdf
  if (cake.exists<EsdfLayer>()) {
    if (argc > 3) {
      output_path = argv[3];
    } else {
      output_path = "./esdf.ply";
    }
    LOG(INFO) << "Writing esdf to: " << output_path;
    CHECK(io::outputVoxelLayerToPly(cake.get<EsdfLayer>(), output_path));
    LOG(INFO) << "Done";
  }

  LOG(INFO) << "Finished running example.";
}
