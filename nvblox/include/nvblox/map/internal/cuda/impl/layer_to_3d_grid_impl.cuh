/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/gpu_hash/internal/cuda/impl/gpu_layer_view_impl.cuh"
#include "nvblox/map/unified_3d_grid.h"

namespace nvblox {

template <typename InputVoxelType, typename OutputCellType,
          typename ConversionOperatorType>
__device__ void copyBlock(
    const Index3D& block_idx,
    const Index3DDeviceHashMapType<VoxelBlock<InputVoxelType>>& block_hash,
    const OutputCellType& default_value,
    const ConversionOperatorType& conversion_op,
    GPU3DGridView<OutputCellType>& grid) {
  using BlockType = VoxelBlock<InputVoxelType>;

  // A single thread accesses the hash
  __shared__ BlockType* block_ptr;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    block_ptr = getBlockPtr(block_hash, block_idx);
  }
  __syncthreads();

  // The voxel addressed by this thread
  // NOTE(alexmillane): Note the x,y,z reversal in the voxel index for
  // coallessed access.
  const Index3D voxel_idx = {threadIdx.z, threadIdx.y, threadIdx.x};
  constexpr int kVoxelsPerSide = BlockType::kVoxelsPerSide;

  // Get a pointer to the output voxel
  const Index3D global_voxel_idx = block_idx * kVoxelsPerSide + voxel_idx;
  if (!grid.isInsideGrid(global_voxel_idx)) {
    // We indexed outside the grid.
    return;
  }
  OutputCellType& voxel = grid(global_voxel_idx);

  // If we could look up the block,
  if (block_ptr) {
    // Do the copy over.
    voxel = conversion_op(
        block_ptr->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()]);
  } else {
    // Otherwise set the default value.
    voxel = default_value;
  }
}

// Call Requirements
// - #ThreadBlocks: A 3D grid equal to the size of the AABB in blocks.
// - #Threads: 3D grid of side length kVoxelsPerSide.
template <typename InputVoxelType, typename OutputCellType,
          typename ConversionOperatorType>
__global__ void extractValuesToGrid(
    Index3DDeviceHashMapType<InputVoxelType> block_hash,
    const Index3D min_index, const OutputCellType default_value,
    ConversionOperatorType conversion_op, GPU3DGridView<OutputCellType> grid) {
  // The block index
  const Index3D block_idx =
      min_index + Index3D(blockIdx.x, blockIdx.y, blockIdx.z);
  copyBlock(block_idx, block_hash, default_value, conversion_op, grid);
}

template <typename VoxelType>
struct IdentityFunctor {
  IdentityFunctor() = default;
  ~IdentityFunctor() = default;

  __device__ __inline__ VoxelType& operator()(VoxelType& voxel) const {
    return voxel;
  }
};

int robustFloor(float float_val) {
  // Function to prevent flooring integer numbers stored as
  // floating variable to the next lower integer due to rounding errors.
  constexpr float kMaxDiffToRound = 1e-4;
  int nearest_int = static_cast<int>(std::round(float_val));
  // Round to the nearest integer if closer than kMaxDiffToRound,
  // otherwise floor to the next lower integer.
  if (std::abs(float_val - nearest_int) < kMaxDiffToRound) {
    return nearest_int;
  } else {
    return static_cast<int>(std::floor(float_val));
  }
}

Index3D robustFloor(const Vector3f& float_vec) {
  Index3D int_vec;
  for (int i = 0; i < float_vec.size(); ++i) {
    int_vec[i] = robustFloor(float_vec[i]);
  }
  return int_vec;
}

template <typename InputVoxelType, typename OutputCellType,
          typename ConversionFunctor>
void voxelLayerToDenseVoxelGridInAABBAsync(
    const VoxelBlockLayer<InputVoxelType>& layer,
    const AxisAlignedBoundingBox& aabb, const OutputCellType default_value,
    const ConversionFunctor& conversion_op, Unified3DGrid<OutputCellType>* grid,
    const CudaStream& cuda_stream) {
  // Size of the output grid
  const float inv_voxel_size = 1.0f / layer.voxel_size();
  // NOTE: robustFloor aims to prevent flooring integer numbers stored as
  // floating variable to the next lower integer due to rounding errors.
  const Index3D min_in_vox = robustFloor(aabb.min() * inv_voxel_size);
  const Index3D max_in_vox = robustFloor(aabb.max() * inv_voxel_size);
  const Index3D dims_in_vox = max_in_vox - min_in_vox + Index3D::Ones();
  CHECK((dims_in_vox.array() > 0).all());
  grid->setAABBAsync(min_in_vox, dims_in_vox, cuda_stream);

  constexpr int kVoxelsPerSide = VoxelBlock<InputVoxelType>::kVoxelsPerSide;
  // Use the same integer voxel bounds as the output grid. Recomputing block
  // bounds from floats can omit an edge plane included by robustFloor().
  const auto blockIndexFromVoxelIndex = [](const Index3D& voxel_index) {
    Index3D block_index;
    for (int axis = 0; axis < 3; ++axis) {
      const int index = voxel_index[axis];
      // C++ division truncates toward zero; negative indices need floor.
      block_index[axis] =
          index / kVoxelsPerSide - (index % kVoxelsPerSide < 0 ? 1 : 0);
    }
    return block_index;
  };
  const Index3D min_block_idx = blockIndexFromVoxelIndex(min_in_vox);
  const Index3D max_block_idx = blockIndexFromVoxelIndex(max_in_vox);
  const Index3D dims_in_blox = max_block_idx - min_block_idx + Index3D::Ones();

  // Extract the values into a grid.
  const dim3 thread_blocks = {static_cast<unsigned int>(dims_in_blox.x()),
                              static_cast<unsigned int>(dims_in_blox.y()),
                              static_cast<unsigned int>(dims_in_blox.z())};
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  extractValuesToGrid<<<thread_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      layer.getGpuLayerView(cuda_stream).getHash().impl_, min_block_idx,
      default_value, conversion_op, grid->getGPUView());
}

template <typename VoxelType>
void voxelLayerToDenseVoxelGridInAABBAsync(
    const VoxelBlockLayer<VoxelType>& layer,  // NOLINT
    const AxisAlignedBoundingBox& aabb,       // NOLINT
    const VoxelType default_value,            // NOLINT
    Unified3DGrid<VoxelType>* grid,           // NOLINT
    const CudaStream& cuda_stream) {
  // Pass through to conversion function with identity
  IdentityFunctor<VoxelType> conversion_op;
  voxelLayerToDenseVoxelGridInAABBAsync(layer, aabb, default_value,
                                        conversion_op, grid, cuda_stream);
}

}  // namespace nvblox
