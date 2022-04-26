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

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "nvblox/core/accessors.h"
#include "nvblox/core/common_names.h"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/mesh/impl/marching_cubes_table.h"
#include "nvblox/mesh/marching_cubes.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

MeshIntegrator::~MeshIntegrator() {
  if (cuda_stream_ != nullptr) {
    cudaStreamDestroy(cuda_stream_);
  }
}

bool MeshIntegrator::integrateBlocksGPU(
    const TsdfLayer& distance_layer, const std::vector<Index3D>& block_indices,
    BlockLayer<MeshBlock>* mesh_layer) {
  timing::Timer mesh_timer("mesh/gpu/integrate");
  CHECK_NOTNULL(mesh_layer);
  CHECK_NEAR(distance_layer.block_size(), mesh_layer->block_size(), 1e-4);
  if (block_indices.empty()) {
    return true;
  }

  // Initialize the stream if not done yet.
  if (cuda_stream_ == nullptr) {
    checkCudaErrors(cudaStreamCreate(&cuda_stream_));
  }

  // Figure out which of these actually contain something worth meshing.
  float block_size = distance_layer.block_size();
  float voxel_size = distance_layer.voxel_size();

  // Clear all blocks if they exist.
  for (const Index3D& block_index : block_indices) {
    MeshBlock::Ptr mesh_block = mesh_layer->getBlockAtIndex(block_index);
    if (mesh_block) {
      mesh_block->clear();
    }
  }

  // First create a list of meshable blocks.
  std::vector<Index3D> meshable_blocks;
  timing::Timer meshable_timer("mesh/gpu/get_meshable");
  getMeshableBlocksGPU(distance_layer, block_indices, 5 * voxel_size,
                       &meshable_blocks);
  meshable_timer.Stop();

  // Then get all the candidates and mesh each block.
  timing::Timer mesh_blocks_timer("mesh/gpu/mesh_blocks");

  meshBlocksGPU(distance_layer, meshable_blocks, mesh_layer);

  // TODO: optionally weld here as well.
  mesh_blocks_timer.Stop();

  return true;
}

// Kernels

// Takes in a vector of blocks, and outputs an integer true if that block is
// meshable.
// Block size MUST be voxels_per_side x voxels_per_side x voxel_per_size.
// Grid size can be anything.
__global__ void isBlockMeshableKernel(int num_blocks,
                                      const VoxelBlock<TsdfVoxel>** blocks,
                                      float cutoff_distance, float min_weight,
                                      bool* meshable) {
  dim3 voxel_index = threadIdx;
  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Get the correct voxel for this index.
    const TsdfVoxel& voxel =
        blocks[block_index]
            ->voxels[voxel_index.z][voxel_index.y][voxel_index.x];
    if (fabs(voxel.distance) <= cutoff_distance && voxel.weight >= min_weight) {
      meshable[block_index] = true;
    }
  }
}

// Takes in a set of blocks arranged in neighbor sets and their relative
// positions, then finds vertex candidates, and finally creates the output
// meshes for them.
// Block size MUST be voxels_per_side x voxels_per_side x voxel_per_size.
// Grid size can be anything.
__global__ void meshBlocksCalculateTableIndicesKernel(
    int num_blocks, const VoxelBlock<TsdfVoxel>** blocks,
    const Vector3f* block_positions, float voxel_size, float min_weight,
    marching_cubes::PerVoxelMarchingCubesResults* marching_cubes_results,
    int* mesh_block_sizes) {
  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  constexpr int kVoxelsPerBlock =
      kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;
  constexpr int kCubeNeighbors = 8;

  const dim3 voxel_index = dim3(threadIdx.z, threadIdx.y, threadIdx.x);

  const int linear_thread_idx =
      threadIdx.x +
      kVoxelsPerSide * (threadIdx.y + kVoxelsPerSide * threadIdx.z);

  // Preallocate a half voxel size.
  const Vector3f half_voxel(0.5f, 0.5f, 0.5f);

  marching_cubes::PerVoxelMarchingCubesResults marching_cubes_results_local;

  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Initialize the calculated output size for this block.
    __shared__ int mesh_block_size;
    if (linear_thread_idx == 0) {
      mesh_block_size = 0;
    }
    __syncthreads();

    // Getting the block pointer is complicated now so let's just get it.
    const VoxelBlock<TsdfVoxel>* block = blocks[block_index * kCubeNeighbors];

    // Get the linear index of the this voxel in this block
    const int vertex_neighbor_idx =
        block_index * kVoxelsPerBlock + linear_thread_idx;

    // Check all 8 neighbors.
    bool skip_voxel = false;
    for (unsigned int i = 0; i < 8; ++i) {
      Index3D corner_index(
          voxel_index.x + marching_cubes::kCornerIndexOffsets[i][0],
          voxel_index.y + marching_cubes::kCornerIndexOffsets[i][1],
          voxel_index.z + marching_cubes::kCornerIndexOffsets[i][2]);
      Index3D block_offset(0, 0, 0);
      bool search_neighbor = false;
      // Are we in bounds? If not, have to get a neighbor.
      // The neighbor should correspond to the index in neighbor blocks.
      for (int j = 0; j < 3; j++) {
        if (corner_index[j] >= kVoxelsPerSide) {
          // Here the index is too much.
          corner_index(j) -= kVoxelsPerSide;
          block_offset(j) = 1;
          search_neighbor = true;
        }
      }

      const TsdfVoxel* voxel = nullptr;
      // Don't look for neighbors for now.
      if (search_neighbor) {
        int neighbor_index =
            marching_cubes::neighborIndexFromDirection(block_offset);
        const VoxelBlock<TsdfVoxel>* neighbor_block =
            blocks[block_index * kCubeNeighbors + neighbor_index];
        if (neighbor_block == nullptr) {
          skip_voxel = true;
          break;
        }
        voxel =
            &neighbor_block
                 ->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
      } else {
        voxel =
            &block
                 ->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
      }
      // If any of the neighbors are not observed, this can't be a mesh
      // triangle.
      if (voxel->weight < min_weight) {
        skip_voxel = true;
        break;
      }

      // Calculate the position of this voxel.
      marching_cubes_results_local.vertex_sdf[i] = voxel->distance;
      marching_cubes_results_local.vertex_coords[i] =
          block_positions[block_index] +
          voxel_size * (corner_index.cast<float>() + half_voxel +
                        (kVoxelsPerSide * block_offset).cast<float>());
    }

    if (!skip_voxel) {
      // If we've made it this far, this needs to be meshed.
      marching_cubes_results_local.contains_mesh = true;

      // Calculate the index into the magic marching cubes table
      marching_cubes_results_local.marching_cubes_table_index =
          marching_cubes::calculateVertexConfiguration(
              marching_cubes_results_local.vertex_sdf);

      // Mesh this cube. This will keep track of what index we're at within
      // the cube.
      marching_cubes::calculateOutputIndex(&marching_cubes_results_local,
                                           &mesh_block_size);

      // Write out to global memory
      marching_cubes_results[vertex_neighbor_idx] =
          marching_cubes_results_local;
    }

    // Writing the shared variable block size to global memory (per block)
    __syncthreads();
    if (linear_thread_idx == 0) {
      mesh_block_sizes[block_index] = mesh_block_size;
    }
  }
}

__global__ void meshBlocksCalculateVerticesKernel(
    int num_blocks,
    const marching_cubes::PerVoxelMarchingCubesResults* marching_cubes_results,
    const int* mesh_block_sizes, CudaMeshBlock* mesh_blocks) {
  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;

  const int linear_thread_idx =
      threadIdx.x +
      kVoxelsPerSide * (threadIdx.y + kVoxelsPerSide * threadIdx.z);

  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // If this block contains a mesh
    if (mesh_block_sizes[block_index] > 0) {
      // Get the linear index of the this voxel in this block
      constexpr int kVoxelsPerBlock =
          kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;
      const int vertex_neighbor_idx =
          block_index * kVoxelsPerBlock + linear_thread_idx;

      // If this voxel contains a mesh
      if (marching_cubes_results[vertex_neighbor_idx].contains_mesh) {
        // Convert the marching cube table index into vertex coordinates
        marching_cubes::calculateVertices(
            marching_cubes_results[vertex_neighbor_idx],
            &mesh_blocks[block_index]);
      }
    }
  }
}

// Wrappers

void MeshIntegrator::getMeshableBlocksGPU(
    const TsdfLayer& distance_layer, const std::vector<Index3D>& block_indices,
    float cutoff_distance, std::vector<Index3D>* meshable_blocks) {
  CHECK_NOTNULL(meshable_blocks);
  if (block_indices.size() == 0) {
    return;
  }

  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  // One block per block, 1 thread per pixel. :)
  // Dim block can be smaller, but dim_threads must be the same.
  int dim_block = block_indices.size();
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);

  // Collect all the meshable blocks as raw pointers.
  // Get all the block pointers and positions.
  block_ptrs_host_.resize(block_indices.size());

  for (size_t i = 0; i < block_indices.size(); i++) {
    block_ptrs_host_[i] =
        distance_layer.getBlockAtIndex(block_indices[i]).get();
  }

  block_ptrs_device_ = block_ptrs_host_;

  // Allocate a device vector that holds the meshable result.
  meshable_device_.resize(block_indices.size());
  meshable_device_.setZero();

  checkCudaErrors(cudaPeekAtLastError());
  isBlockMeshableKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
      block_indices.size(), block_ptrs_device_.data(), cutoff_distance,
      min_weight_, meshable_device_.data());
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));

  meshable_host_ = meshable_device_;

  for (size_t i = 0; i < block_indices.size(); i++) {
    if (meshable_host_[i]) {
      meshable_blocks->push_back(block_indices[i]);
    }
  }
}

void MeshIntegrator::meshBlocksGPU(const TsdfLayer& distance_layer,
                                   const std::vector<Index3D>& block_indices,
                                   BlockLayer<MeshBlock>* mesh_layer) {
  if (block_indices.empty()) {
    return;
  }
  timing::Timer mesh_prep_timer("mesh/gpu/mesh_blocks/prep");
  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  constexpr int kCubeNeighbors = 8;

  // One block per block, 1 thread per voxel. :)
  // Dim block can be smaller, but dim_threads must be the same.
  int dim_block = block_indices.size();
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);

  // Get the block and voxel size.
  const float block_size = distance_layer.block_size();
  const float voxel_size = distance_layer.voxel_size();

  // Get all the block pointers and positions.
  // Block pointers are actually a 2D array of also the neighbor block pointers
  // The neighbors CAN be null so they need to be checked.
  block_ptrs_host_.resize(block_indices.size() * kCubeNeighbors);
  block_positions_host_.resize(block_indices.size());
  for (size_t i = 0; i < block_indices.size(); i++) {
    block_ptrs_host_[i * kCubeNeighbors] =
        distance_layer.getBlockAtIndex(block_indices[i]).get();
    for (size_t j = 1; j < kCubeNeighbors; j++) {
      // Get the pointers to all the neighbors as well.
      block_ptrs_host_[i * kCubeNeighbors + j] =
          distance_layer
              .getBlockAtIndex(block_indices[i] +
                               marching_cubes::directionFromNeighborIndex(j))
              .get();
    }
    block_positions_host_[i] =
        getPositionFromBlockIndex(block_size, block_indices[i]);
  }

  // Create an output mesh blocks vector..
  mesh_blocks_host_.resize(block_indices.size());

  block_ptrs_device_ = block_ptrs_host_;
  block_positions_device_ = block_positions_host_;

  // Allocate working space
  constexpr int kNumVoxelsPerBlock =
      kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;
  marching_cubes_results_device_.resize(block_indices.size() *
                                        kNumVoxelsPerBlock);
  marching_cubes_results_device_.setZero();
  mesh_block_sizes_device_.resize(block_indices.size());
  mesh_block_sizes_device_.setZero();
  mesh_prep_timer.Stop();

  // Run the first half of marching cubes and calculate:
  // - the per-vertex indexes into the magic triangle table
  // - the number of vertices in each mesh block.
  timing::Timer mesh_kernel_1_timer("mesh/gpu/mesh_blocks/kernel_table");
  meshBlocksCalculateTableIndicesKernel<<<dim_block, dim_threads, 0,
                                          cuda_stream_>>>(
      block_indices.size(), block_ptrs_device_.data(),
      block_positions_device_.data(), voxel_size, min_weight_,
      marching_cubes_results_device_.data(), mesh_block_sizes_device_.data());
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));

  mesh_kernel_1_timer.Stop();

  // Copy back the new mesh block sizes (so we can allocate space)
  timing::Timer mesh_copy_timer("mesh/gpu/mesh_blocks/copy_out");
  mesh_block_sizes_host_ = mesh_block_sizes_device_;
  mesh_copy_timer.Stop();

  // Allocate mesh blocks
  timing::Timer mesh_allocation_timer("mesh/gpu/mesh_blocks/block_allocation");
  for (size_t i = 0; i < block_indices.size(); i++) {
    const int num_vertices = mesh_block_sizes_host_[i];

    if (num_vertices > 0) {
      MeshBlock::Ptr output_block =
          mesh_layer->allocateBlockAtIndex(block_indices[i]);

      // Grow the vector with a growth factor and a minimum allocation to avoid
      // repeated reallocation
      if (num_vertices > output_block->capacity()) {
        constexpr int kMinimumMeshBlockTrianglesPerVoxel = 1;
        constexpr int kMinimumMeshBlockVertices =
            kNumVoxelsPerBlock * kMinimumMeshBlockTrianglesPerVoxel * 3;
        constexpr int kMeshBlockOverallocationFactor = 2;
        const int num_vertices_to_allocate =
            std::max(kMinimumMeshBlockVertices,
                     num_vertices * kMeshBlockOverallocationFactor);
        output_block->reserveNumberOfVertices(num_vertices_to_allocate);
      }
      output_block->resizeToNumberOfVertices(num_vertices);
      mesh_blocks_host_[i] = CudaMeshBlock(output_block.get());
    }
  }
  mesh_blocks_device_ = mesh_blocks_host_;
  mesh_allocation_timer.Stop();

  // Run the second half of marching cubes
  // - Translating the magic table indices into triangle vertices and writing
  //   them into the mesh layer.
  timing::Timer mesh_kernel_2_timer("mesh/gpu/mesh_blocks/kernel_vertices");
  meshBlocksCalculateVerticesKernel<<<dim_block, dim_threads, 0,
                                      cuda_stream_>>>(
      block_indices.size(), marching_cubes_results_device_.data(),
      mesh_block_sizes_device_.data(), mesh_blocks_device_.data());
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  mesh_kernel_2_timer.Stop();

  // Optional third stage: welding.
  if (weld_vertices_) {
    timing::Timer welding_timer("mesh/gpu/mesh_blocks/welding");
    weldVertices(block_indices, mesh_layer);
  }
}

void MeshIntegrator::weldVertices(const std::vector<Index3D>& block_indices,
                                  BlockLayer<MeshBlock>* mesh_layer) {
  for (const Index3D& index : block_indices) {
    MeshBlock::Ptr mesh_block = mesh_layer->getBlockAtIndex(index);

    if (!mesh_block || mesh_block->size() <= 3) {
      continue;
    }

    // Store a copy of the input vertices.
    input_vertices_ = mesh_block->vertices;
    input_normals_ = mesh_block->normals;

    // sort vertices to bring duplicates together
    thrust::sort(thrust::device, mesh_block->vertices.begin(),
                 mesh_block->vertices.end(), VectorCompare<Vector3f>());

    // Find unique vertices and erase redundancies. The iterator will point to
    // the new last index.
    auto iterator = thrust::unique(thrust::device, mesh_block->vertices.begin(),
                                   mesh_block->vertices.end());

    // Figure out the new size.
    size_t new_size = iterator - mesh_block->vertices.begin();
    mesh_block->vertices.resize(new_size);
    mesh_block->normals.resize(new_size);

    // Find the indices of the original triangles.
    thrust::lower_bound(thrust::device, mesh_block->vertices.begin(),
                        mesh_block->vertices.end(), input_vertices_.begin(),
                        input_vertices_.end(), mesh_block->triangles.begin(),
                        VectorCompare<Vector3f>());

    // Reshuffle the normals to match.
    thrust::scatter(thrust::device, input_normals_.begin(),
                    input_normals_.end(), mesh_block->triangles.begin(),
                    mesh_block->normals.begin());
  }
}

}  // namespace nvblox