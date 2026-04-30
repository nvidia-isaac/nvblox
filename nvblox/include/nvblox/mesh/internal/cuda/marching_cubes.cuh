#pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh_block.h"

#include "nvblox/mesh/internal/marching_cubes.h"

namespace nvblox {
namespace marching_cubes {

__device__ void calculateOutputIndex(
    PerVoxelMarchingCubesResults* marching_cubes_results, int* size);

template <typename AppearanceType>
__device__ void calculateVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    CudaMeshBlock<AppearanceType>* mesh);

/// Perform marching cubes on pre-read corner SDF values and positions.
/// Returns the number of triangles (0 = no mesh). Populates edge_vertices
/// and sets *table_row_out to the triangle table row for this configuration.
__device__ int marchingCubesFromCorners(
    const float sdf_values[kNumCorners],
    const Vector3f corner_positions[kNumCorners],
    Vector3f edge_vertices[kNumEdges], const int8_t** table_row_out);

}  // namespace marching_cubes
}  // namespace nvblox

#include "nvblox/mesh/internal/impl/cuda/marching_cubes_impl.cuh"
