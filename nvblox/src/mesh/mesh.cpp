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
#include <nvblox/mesh/mesh.h>

namespace nvblox {

Mesh Mesh::fromLayer(const BlockLayer<MeshBlock>& layer) {
  Mesh mesh;

  // Keep track of the vertex index.
  int next_index = 0;

  // Iterate over every block in the layer.
  const std::vector<Index3D> indices = layer.getAllBlockIndices();

  for (const Index3D& index : indices) {
    MeshBlock::ConstPtr block = layer.getBlockAtIndex(index);

    // Copy over.
    unified_vector<Vector3f> vertices;
    vertices.copyFrom(block->vertices);
    mesh.vertices.resize(mesh.vertices.size() + vertices.size());
    std::copy(vertices.begin(), vertices.end(),
              mesh.vertices.begin() + next_index);

    unified_vector<Vector3f> normals;
    normals.copyFrom(block->normals);
    mesh.normals.resize(mesh.normals.size() + normals.size());
    std::copy(normals.begin(), normals.end(),
              mesh.normals.begin() + next_index);

    unified_vector<Color> colors;
    colors.copyFrom(block->colors);
    mesh.colors.resize(mesh.colors.size() + colors.size());
    std::copy(colors.begin(), colors.end(), mesh.colors.begin() + next_index);

    // Our simple mesh implementation has:
    // - per vertex colors
    // - per vertex normals
    CHECK((vertices.size() == normals.size()) || (normals.size() == 0));
    CHECK((vertices.size() == vertices.size()) || (colors.size() == 0));

    // Copy over the triangles.
    unified_vector<int> triangles;
    triangles.copyFrom(block->triangles);
    std::vector<int> triangle_indices(triangles.size());
    // Increment all triangle indices.
    std::transform(triangles.begin(), triangles.end(), triangle_indices.begin(),
                   std::bind2nd(std::plus<int>(), next_index));

    mesh.triangles.insert(mesh.triangles.end(), triangle_indices.begin(),
                          triangle_indices.end());

    next_index += vertices.size();
  }

  return mesh;
}

}  // namespace nvblox
