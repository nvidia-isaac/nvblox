// #pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"

namespace nvblox {
namespace marching_cubes {

__device__ void calculateOutputIndex(
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

__device__ void calculateVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    CudaMeshBlock* mesh) {
  const uint8_t table_index = marching_cubes_results.marching_cubes_table_index;
  const int num_triangles_in_voxel = kNumTrianglesTable[table_index];

  if (num_triangles_in_voxel == 0) {
    return;
  }

  // The position in the block that we start output for this voxel.
  int next_index = marching_cubes_results.vertex_vector_start_index;

  Eigen::Matrix<float, 3, 12> edge_vertex_coordinates;
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
    mesh->normals[next_index] = n;
    mesh->normals[next_index + 1] = n;
    mesh->normals[next_index + 2] = n;
    next_index += 3;
    table_col += 3;
  }
}

}  // namespace marching_cubes
}  // namespace nvblox