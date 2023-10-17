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
#pragma once

#include <nvblox/core/types.h>
#include <nvblox/map/layer.h>
#include <nvblox/mesh/mesh_block.h>
#include <nvblox/map/common_names.h>

namespace nvblox {

/// A structure which holds a combined Mesh for CPU access.
/// Generally produced as the result of fusing MeshBlocks in a Layer<MeshBlock>
/// into a single mesh.
/// NOTE(alexmillane): Currently only on the CPU.
struct Mesh {
  // Data
  std::vector<Vector3f> vertices;
  std::vector<Vector3f> normals;
  std::vector<int> triangles;
  std::vector<Color> colors;

  /// Create a combined Mesh object from a MeshBlock layer. Useful for mesh
  /// output.
  static Mesh fromLayer(const MeshLayer& layer);
};

}  // namespace nvblox
