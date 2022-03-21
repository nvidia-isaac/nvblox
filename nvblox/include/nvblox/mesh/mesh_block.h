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

#include <memory>

#include "nvblox/core/blox.h"
#include "nvblox/core/color.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {

// A mesh block containing all of the triangles from this block.
// Each block contains only the UPPER part of its neighbors: i.e., the max
// x, y, and z axes. Its neighbors are responsible for the rest.
struct MeshBlock {
  typedef std::shared_ptr<MeshBlock> Ptr;
  typedef std::shared_ptr<const MeshBlock> ConstPtr;

  MeshBlock(MemoryType memory_type = MemoryType::kDevice);

  // Mesh Data
  // These unified vectors contain the mesh data for this block. Note that
  // Colors and/or intensities are optional. The "triangles" vector is a vector
  // of indices into the vertices vector. Triplets of consecutive elements form
  // triangles with the indexed vertices as their corners.
  unified_vector<Vector3f> vertices;
  unified_vector<Vector3f> normals;
  unified_vector<Color> colors;
  unified_vector<float> intensities;
  unified_vector<int> triangles;

  void clear();

  // Resize and reserve.
  void resizeToNumberOfVertices(size_t new_size);
  void reserveNumberOfVertices(size_t new_capacity);

  size_t size() const;
  size_t capacity() const;

  // Resize colors/intensities such that:
  // `colors.size()/intensities.size() == vertices.size()`
  void expandColorsToMatchVertices();
  void expandIntensitiesToMatchVertices();

  // Copy mesh data to the CPU.
  std::vector<Vector3f> getVertexVectorOnCPU() const;
  std::vector<Vector3f> getNormalVectorOnCPU() const;
  std::vector<int> getTriangleVectorOnCPU() const;
  std::vector<Color> getColorVectorOnCPU() const;

  // Note(alexmillane): Memory type ignored, MeshBlocks live in CPU memory.
  static Ptr allocate(MemoryType memory_type);
};

// Helper struct for mesh blocks on CUDA.
// NOTE: We need this because we cant pass MeshBlock to kernel functions because
// of the presence of unified_vector members.
struct CudaMeshBlock {
  CudaMeshBlock() = default;
  CudaMeshBlock(MeshBlock* block);

  Vector3f* vertices;
  Vector3f* normals;
  int* triangles;
  Color* colors;
  int size;
};

}  // namespace nvblox