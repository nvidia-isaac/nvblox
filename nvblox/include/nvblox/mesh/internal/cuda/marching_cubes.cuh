#pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh_block.h"

#include "nvblox/mesh/internal/marching_cubes.h"

namespace nvblox {
namespace marching_cubes {

__device__ void calculateOutputIndex(
    PerVoxelMarchingCubesResults* marching_cubes_results, int* size);

__device__ void calculateVertices(
    const PerVoxelMarchingCubesResults& marching_cubes_results,
    CudaMeshBlock* mesh);

}  // namespace marching_cubes
}  // namespace nvblox

#include "nvblox/mesh/internal/impl/cuda/marching_cubes_impl.cuh"
