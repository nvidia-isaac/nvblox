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
#include "nvblox/io/mesh_io.h"

#include <vector>

#include "nvblox/io/ply_writer.h"
#include "nvblox/mesh/mesh.h"

namespace nvblox {
namespace io {

bool outputMeshLayerToPly(const BlockLayer<MeshBlock>& layer,
                          const std::string& filename) {
  // TODO: doesn't support intensity yet!!!!
  const Mesh mesh = Mesh::fromLayer(layer);

  // Create the ply writer object
  io::PlyWriter writer(filename);
  writer.setPoints(&mesh.vertices);
  writer.setTriangles(&mesh.triangles);
  if (mesh.normals.size() > 0) {
    writer.setNormals(&mesh.normals);
  }
  if (mesh.colors.size() > 0) {
    writer.setColors(&mesh.colors);
  }
  return writer.write();
}

bool outputMeshLayerToPly(const BlockLayer<MeshBlock>& layer,
                          const char* filename) {
  return outputMeshLayerToPly(layer, std::string(filename));
}

}  // namespace io
}  // namespace nvblox
