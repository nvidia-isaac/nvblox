#include "nvblox/core/types.h"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"

namespace nvblox {
namespace marching_cubes {
inline int calculateVertexConfiguration(const float vertex_sdf[8]) {
  return (vertex_sdf[0] < 0 ? (1 << 0) : 0) |
         (vertex_sdf[1] < 0 ? (1 << 1) : 0) |
         (vertex_sdf[2] < 0 ? (1 << 2) : 0) |
         (vertex_sdf[3] < 0 ? (1 << 3) : 0) |
         (vertex_sdf[4] < 0 ? (1 << 4) : 0) |
         (vertex_sdf[5] < 0 ? (1 << 5) : 0) |
         (vertex_sdf[6] < 0 ? (1 << 6) : 0) |
         (vertex_sdf[7] < 0 ? (1 << 7) : 0);
}

inline int neighborIndexFromDirection(const Index3D& direction) {
  return (direction.x() << 2 | direction.y() << 1 | direction.z() << 0);
}

inline Index3D directionFromNeighborIndex(const int index) {
  return Index3D((index & (1 << 2)) >> 2, (index & (1 << 1)) >> 1,
                 (index & (1 << 0)) >> 0);
}

/// Performs linear interpolation on two cube corners to find the approximate
/// zero crossing (surface) value.
inline Vector3f interpolateVertex(const Vector3f& vertex1,
                                  const Vector3f& vertex2, float sdf1,
                                  float sdf2) {
  constexpr float kMinSdfDifference = 1e-4;
  const float sdf_diff = sdf1 - sdf2;
  // Only compute the actual interpolation value if the sdf_difference is not
  // too small, this is to counteract issues with floating point precision.
  if (abs(sdf_diff) >= kMinSdfDifference) {
    const float t = sdf1 / sdf_diff;
    return Vector3f(vertex1 + t * (vertex2 - vertex1));
  } else {
    return Vector3f(0.5 * (vertex1 + vertex2));
  }
}

inline void interpolateEdgeVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    Eigen::Matrix<float, 3, 12>* edge_coords) {
  CHECK(edge_coords != NULL);
  for (size_t i = 0; i < 12; ++i) {
    const uint8_t* pairs = kEdgeIndexPairs[i];
    const int edge0 = pairs[0];
    const int edge1 = pairs[1];
    // Only interpolate along edges where there is a zero crossing.
    if ((marching_cubes_results.vertex_sdf[edge0] < 0 &&
         marching_cubes_results.vertex_sdf[edge1] >= 0) ||
        (marching_cubes_results.vertex_sdf[edge0] >= 0 &&
         marching_cubes_results.vertex_sdf[edge1] < 0))
      edge_coords->col(i) =
          interpolateVertex(marching_cubes_results.vertex_coords[edge0],
                            marching_cubes_results.vertex_coords[edge1],
                            marching_cubes_results.vertex_sdf[edge0],
                            marching_cubes_results.vertex_sdf[edge1]);
  }
}

inline void meshCube(const PerVoxelMarchingCubesResults& marching_cubes_results,
                     MeshBlock* mesh) {
  CHECK_NOTNULL(mesh);
  const int table_index = marching_cubes_results.marching_cubes_table_index;

  // No edges in this cube.
  if (table_index == 0 || table_index == 255) {
    return;
  }

  Eigen::Matrix<float, 3, 12> edge_vertex_coordinates;
  interpolateEdgeVertices(marching_cubes_results, &edge_vertex_coordinates);

  const int8_t* table_row = kTriangleTable[table_index];

  int table_col = 0;
  int next_index = mesh->vertices.size();

  // Resize the mesh block to contain all the vertices.
  int num_triangles = kNumTrianglesTable[table_index];

  table_col = 0;
  for (int i = 0; i < num_triangles; i++) {
    mesh->vertices.push_back(
        edge_vertex_coordinates.col(table_row[table_col + 2]));
    mesh->vertices.push_back(
        edge_vertex_coordinates.col(table_row[table_col + 1]));
    mesh->vertices.push_back(edge_vertex_coordinates.col(table_row[table_col]));
    mesh->triangles.push_back(next_index);
    mesh->triangles.push_back(next_index + 1);
    mesh->triangles.push_back(next_index + 2);
    const Vector3f& p0 = mesh->vertices[next_index];
    const Vector3f& p1 = mesh->vertices[next_index + 1];
    const Vector3f& p2 = mesh->vertices[next_index + 2];
    Vector3f px = (p1 - p0);
    Vector3f py = (p2 - p0);
    Vector3f n = px.cross(py).normalized();
    mesh->normals.push_back(n);
    mesh->normals.push_back(n);
    mesh->normals.push_back(n);
    next_index += 3;
    table_col += 3;
  }
}

}  // namespace marching_cubes

}  // namespace nvblox
