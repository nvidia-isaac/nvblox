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
#include <cuda_runtime.h>
#include <type_traits>  // Ensure this is included

#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/internal/appearance_getter.h"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"
#include "nvblox/mesh/internal/marching_cubes.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearance(
    const AppearanceLayerType& appearance_layer, MeshLayerType* mesh_layer) {
  updateAppearance(appearance_layer, mesh_layer->getAllBlockIndices(),
                   mesh_layer);
}

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearance(
    const AppearanceLayerType& appearance_layer,
    const std::vector<Index3D>& block_indices, MeshLayerType* mesh_layer) {
  // Default choice is GPU
  updateAppearanceGPU(appearance_layer, block_indices, mesh_layer);
}

//
/* Appearance Mesh blocks on the GPU
 *
 * Call with
 * - one ThreadBlock per VoxelBlock, GridDim 1D
 * - BlockDim 1D, any size: we implement a stridded access pattern over
 *   MeshBlock verticies
 *
 * @param: appearance_blocks: a list of appearance blocks which correspond in
 * position to mesh_blocks
 * @param: block_indices:     a list of blocks indices.
 * @param: cuda_mesh_blocks:  a list of mesh_blocks to be appearance painted.
 */
template <typename AppearanceVoxelType>
__global__ void updateAppearanceBlockByClosestVoxel(
    const VoxelBlock<AppearanceVoxelType>** appearance_blocks,
    const Index3D* block_indices, const float block_size,
    const float voxel_size,
    CudaMeshBlock<typename AppearanceVoxelType::ArrayType>* cuda_mesh_blocks) {
  // Block
  const VoxelBlock<AppearanceVoxelType>* appearance_block_ptr =
      appearance_blocks[blockIdx.x];
  const Index3D block_index = block_indices[blockIdx.x];
  CudaMeshBlock<typename AppearanceVoxelType::ArrayType> cuda_mesh_block =
      cuda_mesh_blocks[blockIdx.x];

  // The position of this block in the layer
  const Vector3f p_L_B_m = getPositionFromBlockIndex(block_size, block_index);

  // Interate through MeshBlock vertices - Stidded access pattern
  for (int i = threadIdx.x; i < cuda_mesh_block.vertices_size;
       i += blockDim.x) {
    // The position of this vertex in the layer
    const Vector3f p_L_V_m = cuda_mesh_block.vertices[i];

    // The position of this vertex in the block
    const Vector3f p_B_V_m = p_L_V_m - p_L_B_m;

    // Convert this to a voxel index
    Index3D voxel_idx_in_block = (p_B_V_m.array() / voxel_size).cast<int>();

    // NOTE(alexmillane): Here we make some assumptions.
    // - We assume that the closest voxel to p_L_V is in the AppearanceBlock
    //   co-located with the MeshBlock from which p_L_V was drawn.
    // - This is will (very?) occasionally be incorrect when mesh vertices
    //   escape block boundaries. However, making this assumption saves us any
    //   neighbor calculations.
    constexpr size_t KVoxelsPerSizeMinusOne =
        VoxelBlock<AppearanceVoxelType>::kVoxelsPerSide - 1;
    voxel_idx_in_block =
        voxel_idx_in_block.array().min(KVoxelsPerSizeMinusOne).max(0);

    // Get the appearance voxel
    const AppearanceVoxelType appearance_voxel =
        appearance_block_ptr->voxels[voxel_idx_in_block.x()]  // NOLINT
                                    [voxel_idx_in_block.y()]  // NOLINT
                                    [voxel_idx_in_block.z()];

    // Write the appearance out to global memory
    cuda_mesh_block.vertex_appearances[i] =
        AppearanceGetter<AppearanceVoxelType>::getAppearance(appearance_voxel);
  }
}

template <typename AppearanceType>
__global__ void updateAppearanceBlocksConstant(
    AppearanceType appearance,
    CudaMeshBlock<AppearanceType>* cuda_mesh_blocks) {
  // Each threadBlock operates on a single MeshBlock
  CudaMeshBlock<AppearanceType> cuda_mesh_block = cuda_mesh_blocks[blockIdx.x];
  // Interate through MeshBlock vertices - Stidded access pattern
  for (int i = threadIdx.x; i < cuda_mesh_block.vertices_size;
       i += blockDim.x) {
    cuda_mesh_block.vertex_appearances[i] = appearance;
  }
}

template <typename AppearanceType>
void updateAppearanceBlocksConstantGPU(
    const std::vector<Index3D>& block_indices, const AppearanceType& appearance,
    MeshBlockLayer<AppearanceType>* mesh_layer, CudaStream* cuda_stream) {
  using CudaMeshBlockType = CudaMeshBlock<AppearanceType>;

  CHECK_NOTNULL(mesh_layer);
  if (block_indices.size() == 0) {
    return;
  }

  // Prepare CudaMeshBlockTypes, which are effectively containers of device
  // pointers
  std::vector<CudaMeshBlockType> cuda_mesh_blocks;
  cuda_mesh_blocks.resize(block_indices.size());
  for (size_t i = 0; i < block_indices.size(); i++) {
    cuda_mesh_blocks[i] =
        CudaMeshBlockType(mesh_layer->getBlockAtIndex(block_indices[i]).get());
  }

  // Allocate
  CudaMeshBlockType* cuda_mesh_block_device_ptrs;
  checkCudaErrors(
      cudaMalloc(&cuda_mesh_block_device_ptrs,
                 cuda_mesh_blocks.size() * sizeof(CudaMeshBlockType)));

  // Host -> GPU
  checkCudaErrors(
      cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
                      cuda_mesh_blocks.size() * sizeof(CudaMeshBlockType),
                      cudaMemcpyHostToDevice, *cuda_stream));

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
  const int num_blocks = block_indices.size();
  updateAppearanceBlocksConstant<<<num_blocks, kThreadsPerBlock, 0,
                                   *cuda_stream>>>(appearance,
                                                   cuda_mesh_block_device_ptrs);
  cuda_stream->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Deallocate
  checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
}

template <typename AppearanceVoxelType>
void updateAppearanceBlockByClosestVoxelGPU(
    const VoxelBlockLayer<AppearanceVoxelType>& appearance_layer,
    const std::vector<Index3D>& block_indices,
    MeshBlockLayer<typename AppearanceVoxelType::ArrayType>* mesh_layer,
    CudaStream* cuda_stream) {
  using CudaMeshBlockType =
      CudaMeshBlock<typename AppearanceVoxelType::ArrayType>;

  CHECK_NOTNULL(mesh_layer);
  if (block_indices.size() == 0) {
    return;
  }

  // Get the locations (on device) of the appearance blocks
  // NOTE(alexmillane): This function assumes that all block_indices have been
  // checked to exist in appearance_layer.
  std::vector<const VoxelBlock<AppearanceVoxelType>*> appearance_blocks =
      getBlockPtrsFromIndices(block_indices, appearance_layer);

  // Prepare CudaMeshBlockTypes, which are effectively containers of device
  // pointers
  std::vector<CudaMeshBlockType> cuda_mesh_blocks;
  cuda_mesh_blocks.resize(block_indices.size());
  for (size_t i = 0; i < block_indices.size(); i++) {
    cuda_mesh_blocks[i] =
        CudaMeshBlockType(mesh_layer->getBlockAtIndex(block_indices[i]).get());
  }

  // Allocate
  const VoxelBlock<AppearanceVoxelType>** appearance_block_device_ptrs;
  checkCudaErrors(cudaMalloc(
      &appearance_block_device_ptrs,
      appearance_blocks.size() * sizeof(VoxelBlock<AppearanceVoxelType>*)));
  Index3D* block_indices_device_ptr;
  checkCudaErrors(cudaMalloc(&block_indices_device_ptr,
                             block_indices.size() * sizeof(Index3D)));
  CudaMeshBlockType* cuda_mesh_block_device_ptrs;
  checkCudaErrors(
      cudaMalloc(&cuda_mesh_block_device_ptrs,
                 cuda_mesh_blocks.size() * sizeof(CudaMeshBlockType)));

  // Host -> GPU transfers
  checkCudaErrors(cudaMemcpyAsync(
      appearance_block_device_ptrs, appearance_blocks.data(),
      appearance_blocks.size() * sizeof(VoxelBlock<AppearanceVoxelType>*),
      cudaMemcpyHostToDevice, *cuda_stream));
  checkCudaErrors(cudaMemcpyAsync(block_indices_device_ptr,
                                  block_indices.data(),
                                  block_indices.size() * sizeof(Index3D),
                                  cudaMemcpyHostToDevice, *cuda_stream));
  checkCudaErrors(
      cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
                      cuda_mesh_blocks.size() * sizeof(CudaMeshBlockType),
                      cudaMemcpyHostToDevice, *cuda_stream));

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
  const int num_blocks = block_indices.size();
  const float voxel_size =
      mesh_layer->block_size() / VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  updateAppearanceBlockByClosestVoxel<<<num_blocks, kThreadsPerBlock, 0,
                                        *cuda_stream>>>(
      appearance_block_device_ptrs,  // NOLINT
      block_indices_device_ptr,      // NOLINT
      mesh_layer->block_size(),      // NOLINT
      voxel_size,                    // NOLINT
      cuda_mesh_block_device_ptrs);
  cuda_stream->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Deallocate
  checkCudaErrors(cudaFree(appearance_block_device_ptrs));
  checkCudaErrors(cudaFree(block_indices_device_ptr));
  checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
}

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearanceGPU(
    const AppearanceLayerType& appearance_layer, MeshLayerType* mesh_layer) {
  updateAppearanceGPU(appearance_layer, mesh_layer->getAllBlockIndices(),
                      mesh_layer);
}

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearanceGPU(
    const AppearanceLayerType& appearance_layer,
    const std::vector<Index3D>& requested_block_indices,
    MeshLayerType* mesh_layer) {
  CHECK_NOTNULL(mesh_layer);
  CHECK_EQ(appearance_layer.block_size(), mesh_layer->block_size());

  // NOTE(alexmillane): Generally, some of the MeshBlocks which we are
  // "coloring" will not have data in the color layer. HOWEVER, for colored
  // MeshBlocks (ie with non-empty color members), the size of the colors must
  // match vertices. Therefore we "color" all requested block_indices in two
  // parts:
  // - The first part using the color layer, and
  // - the second part a constant color.

  // Check for each index, that the MeshBlock exists, and if it does
  // allocate space for color.
  std::vector<Index3D> block_indices;
  block_indices.reserve(requested_block_indices.size());
  std::for_each(requested_block_indices.begin(), requested_block_indices.end(),
                [&mesh_layer, &block_indices, this](const Index3D& block_idx) {
                  if (mesh_layer->isBlockAllocated(block_idx)) {
                    mesh_layer->getBlockAtIndex(block_idx)
                        ->expandAppearanceToMatchVerticesAsync(*cuda_stream_);
                    block_indices.push_back(block_idx);
                  }
                });

  // Split block indices into two groups, one group containing indices with
  // corresponding ColorBlocks, and one without.
  std::vector<Index3D> block_indices_in_appearance_layer;
  std::vector<Index3D> block_indices_not_in_appearance_layer;
  block_indices_in_appearance_layer.reserve(block_indices.size());
  block_indices_not_in_appearance_layer.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (appearance_layer.isBlockAllocated(block_idx)) {
      block_indices_in_appearance_layer.push_back(block_idx);
    } else {
      block_indices_not_in_appearance_layer.push_back(block_idx);
    }
  }

  updateAppearanceBlockByClosestVoxelGPU(appearance_layer,
                                         block_indices_in_appearance_layer,
                                         mesh_layer, cuda_stream_.get());
  updateAppearanceBlocksConstantGPU(
      block_indices_not_in_appearance_layer,
      AppearanceGetter<AppearanceVoxelType>::getDefaultAppearance(), mesh_layer,
      cuda_stream_.get());
}

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearanceCPU(
    const AppearanceLayerType& appearance_layer, MeshLayerType* mesh_layer,
    const CudaStream& cuda_stream) {
  updateAppearanceCPU(appearance_layer, mesh_layer->getAllBlockIndices(),
                      mesh_layer, cuda_stream);
}

template <typename AppearanceVoxelType>
void MeshIntegrator<AppearanceVoxelType>::updateAppearanceCPU(
    const AppearanceLayerType& appearance_layer,
    const std::vector<Index3D>& block_indices, MeshLayerType* mesh_layer,
    const CudaStream& cuda_stream) {
  // For each vertex just grab the closest color
  for (const Index3D& block_idx : block_indices) {
    typename MeshBlockType::Ptr block = mesh_layer->getBlockAtIndex(block_idx);
    if (block == nullptr) {
      continue;
    }
    block->vertex_appearances.resizeAsync(block->vertices.size(), cuda_stream);
    cuda_stream.synchronize();
    for (size_t i = 0; i < block->vertices.size(); i++) {
      const Vector3f& vertex = block->vertices[i];
      const AppearanceVoxelType* appearance_voxel;
      if (getVoxelAtPosition<AppearanceVoxelType>(appearance_layer, vertex,
                                                  &appearance_voxel)) {
        block->vertex_appearances[i] =
            AppearanceGetter<AppearanceVoxelType>::getAppearance(
                *appearance_voxel);
      } else {
        block->vertex_appearances[i] =
            AppearanceGetter<AppearanceVoxelType>::getDefaultAppearance();
      }
    }
  }
}

template class MeshIntegrator<ColorVoxel>;
template class MeshIntegrator<FeatureVoxel>;

}  // namespace nvblox
