// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {
namespace marching_cubes {

/// This (internal) struct contains intermediate results of marching cubes.
struct PerVoxelMarchingCubesResults {
  /// The 3D positions of the corners of a 2x2x2 cube of voxels formed by the
  /// surrounding voxel neighbours in the positive direction of each coordinate.
  Vector3f vertex_coords[8];
  /// The value of the TSDF at each of the neighbouring voxels described above.
  float vertex_sdf[8];
  /// Does this voxel contain a mesh? (Does it straddle a zero level set?)
  bool contains_mesh = false;
  /// The index into the marching cubes triangle table (found in
  /// nvblox/mesh/impl/marching_cubes_table.h). This index is determined based
  /// on the tsdf configuration (+/-) of the surrounding voxels and is the main
  /// contribution of marching cubes algorithm.
  uint8_t marching_cubes_table_index = 0;
  /// At the end of marching cubes, vertices calculated for this and other
  /// voxels in this MeshBlock are stored in a single vector. This member
  /// indicates where in this vertex vector the vertices associated with this
  /// voxel begin. It is calculated through an exclusive prefix sum of the
  /// numbers of vertices in each voxel of this MeshBlock.
  int vertex_vector_start_index;
};

// Performs the marching cubes algorithm to generate a mesh layer from a TSDF.
// Implementation taken heavily from Open Chisel
// https://github.com/personalrobotics/OpenChisel

// Calculate the vertex configuration of a given set of neighbor distances.
__host__ __device__ int calculateVertexConfiguration(const float vertex_sdf[8]);

// This is for blocks access in the kernel.
__host__ __device__ int neighborIndexFromDirection(const Index3D& direction);
__host__ __device__ Index3D directionFromNeighborIndex(const int index);

// Output (edge coords) is 12 long. Should be preallocated.
__host__ __device__ void interpolateEdgeVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    Eigen::Matrix<float, 3, 12>* edge_coords);

/// Performs linear interpolation on two cube corners to find the approximate
/// zero crossing (surface) value.
__host__ __device__ Vector3f interpolateVertex(const Vector3f& vertex1,
                                               const Vector3f& vertex2,
                                               float sdf1, float sdf2);

// Actually populate the mesh block.
__host__ void meshCube(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    MeshBlock* mesh);

}  // namespace marching_cubes
}  // namespace nvblox

#include "nvblox/mesh/internal/impl/marching_cubes_impl.h"
