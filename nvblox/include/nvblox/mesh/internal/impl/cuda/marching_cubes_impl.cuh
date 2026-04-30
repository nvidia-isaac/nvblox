// #pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"

namespace nvblox {
namespace marching_cubes {

__device__ inline void calculateOutputIndex(
    PerVoxelMarchingCubesResults* marching_cubes_results, int* size) {
  // How many vertices in this voxel
  const uint8_t table_index =
      marching_cubes_results->marching_cubes_table_index;
  const int num_vertices_in_voxel = kNumVertsTable[table_index];

  // No edges in this cube.
  if (num_vertices_in_voxel == 0) {
    return;
  }

  // Calculate:
  // - the start index where this voxel starts outputing, and
  // - the total number of vertices in this mesh block (once all threads
  //   finish).
  marching_cubes_results->vertex_vector_start_index =
      atomicAdd(size, num_vertices_in_voxel);
}

template <typename AppearanceType>
__device__ void calculateVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    CudaMeshBlock<AppearanceType>* mesh) {
  const uint8_t table_index = marching_cubes_results.marching_cubes_table_index;
  const int num_triangles_in_voxel = kNumTrianglesTable[table_index];

  if (num_triangles_in_voxel == 0) {
    return;
  }

  // The position in the block that we start output for this voxel.
  int next_index = marching_cubes_results.vertex_vector_start_index;

  Eigen::Matrix<float, 3, kNumEdges> edge_vertex_coordinates;
  interpolateEdgeVertices(marching_cubes_results, &edge_vertex_coordinates);

  const int8_t* table_row = kTriangleTable[table_index];
  int table_col = 0;
  for (int i = 0; i < num_triangles_in_voxel; i++) {
    mesh->vertices[next_index] =
        edge_vertex_coordinates.col(table_row[table_col + 2]);
    mesh->vertices[next_index + 1] =
        edge_vertex_coordinates.col(table_row[table_col + 1]);
    mesh->vertices[next_index + 2] =
        edge_vertex_coordinates.col(table_row[table_col]);
    mesh->triangles[next_index] = next_index;
    mesh->triangles[next_index + 1] = next_index + 1;
    mesh->triangles[next_index + 2] = next_index + 2;
    const Vector3f& p0 = mesh->vertices[next_index];
    const Vector3f& p1 = mesh->vertices[next_index + 1];
    const Vector3f& p2 = mesh->vertices[next_index + 2];
    Vector3f px = (p1 - p0);
    Vector3f py = (p2 - p0);
    Vector3f n = px.cross(py).normalized();
    mesh->vertex_normals[next_index] = n;
    mesh->vertex_normals[next_index + 1] = n;
    mesh->vertex_normals[next_index + 2] = n;
    next_index += kVerticesPerTriangle;
    table_col += kVerticesPerTriangle;
  }
}

/// Perform marching cubes on pre-read corner SDF values and positions.
/// Returns the number of triangles (0 = no mesh). Populates edge_vertices
/// and sets *table_row_out to the triangle table row for this configuration.
__device__ inline int marchingCubesFromCorners(
    const float sdf_values[kNumCorners],
    const Vector3f corner_positions[kNumCorners],
    Vector3f edge_vertices[kNumEdges], const int8_t** table_row_out) {
  // Build the 8-bit configuration index: bit i is set if corner i is inside
  // the surface (SDF < 0). This indexes into the marching cubes lookup tables.
  int table_index = 0;
  for (int i = 0; i < kNumCorners; ++i) {
    if (sdf_values[i] < 0.0f) {
      table_index |= (1 << i);
    }
  }

  // All corners on the same side of the surface -- no zero crossing.
  if (table_index == 0 || table_index == kAllInsideConfig) return 0;

  // Interpolate vertex positions along each of the 12 cube edges where the
  // surface crosses (approximate zero-crossing via linear interpolation).
  for (int e = 0; e < kNumEdges; ++e) {
    int v0 = kEdgeIndexPairs[e][0];
    int v1 = kEdgeIndexPairs[e][1];
    edge_vertices[e] =
        interpolateVertex(corner_positions[v0], corner_positions[v1],
                          sdf_values[v0], sdf_values[v1]);
  }

  // Return the triangle table row and count for this configuration.
  *table_row_out = kTriangleTable[table_index];
  return kNumTrianglesTable[table_index];
}

}  // namespace marching_cubes
}  // namespace nvblox
