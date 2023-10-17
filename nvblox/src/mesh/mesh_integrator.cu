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

#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/internal/cuda/marching_cubes.cuh"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"
#include "nvblox/mesh/internal/marching_cubes.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

MeshIntegrator::MeshIntegrator()
    : MeshIntegrator(std::make_shared<CudaStreamOwning>()) {}

MeshIntegrator::MeshIntegrator(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {
  // clang-format off
    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0,
                           0, 0, 1, 1, 0, 0, 1, 1,
                           0, 0, 0, 0, 1, 1, 1, 1;
  // clang-format on
}

// Return all indices that exists in layer
std::vector<Index3D> getIndicesInLayer(
    const std::vector<Index3D>& block_indices_in, const TsdfLayer& layer) {
  std::vector<Index3D> out = block_indices_in;

  auto remove_end = std::remove_if(
      out.begin(), out.end(), [&layer](const Index3D& block_index) {
        return layer.getBlockAtIndex(block_index) == nullptr;
      });
  out.erase(remove_end, out.end());
  return out;
}

bool MeshIntegrator::integrateBlocksGPU(
    const TsdfLayer& distance_layer,
    const std::vector<Index3D>& block_indices_in,
    BlockLayer<MeshBlock>* mesh_layer) {
  timing::Timer mesh_timer("mesh/gpu/integrate");
  const std::vector<Index3D> block_indices =
      getIndicesInLayer(block_indices_in, distance_layer);
  CHECK_NOTNULL(mesh_layer);
  CHECK_NEAR(distance_layer.block_size(), mesh_layer->block_size(), 1e-4);
  if (block_indices.empty()) {
    return true;
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
  getMeshableBlocksGPU(distance_layer, block_indices,
                       cutoff_distance_vox_ * voxel_size, &meshable_blocks);
  meshable_timer.Stop();

  // Then get all the candidates and mesh each block.
  timing::Timer mesh_blocks_timer("mesh/gpu/mesh_blocks");
  meshBlocksGPU(distance_layer, meshable_blocks, mesh_layer);
  mesh_blocks_timer.Stop();

  return true;
}

bool MeshIntegrator::integrateMeshFromDistanceField(
    const TsdfLayer& distance_layer, BlockLayer<MeshBlock>* mesh_layer,
    const DeviceType device_type) {
  // First, get all the blocks.
  std::vector<Index3D> block_indices = distance_layer.getAllBlockIndices();
  if (device_type == DeviceType::kCPU) {
    return integrateBlocksCPU(distance_layer, block_indices, mesh_layer);
  } else {
    return integrateBlocksGPU(distance_layer, block_indices, mesh_layer);
  }
}

bool MeshIntegrator::integrateBlocksCPU(
    const TsdfLayer& distance_layer, const std::vector<Index3D>& block_indices,
    BlockLayer<MeshBlock>* mesh_layer) {
  timing::Timer mesh_timer("mesh/integrate");
  CHECK_NOTNULL(mesh_layer);
  CHECK_NEAR(distance_layer.block_size(), mesh_layer->block_size(), 1e-4);

  // Figure out which of these actually contain something worth meshing.
  const float block_size = distance_layer.block_size();
  const float voxel_size = distance_layer.voxel_size();

  // For each block...
  for (const Index3D& block_index : block_indices) {
    // Get the block.
    VoxelBlock<TsdfVoxel>::ConstPtr block =
        distance_layer.getBlockAtIndex(block_index);

    // Check meshability - basically if this contains anything near the
    // border.
    if (!isBlockMeshable(block, voxel_size * 2)) {
      continue;
    }

    // Get all the neighbor blocks.
    std::vector<VoxelBlock<TsdfVoxel>::ConstPtr> neighbor_blocks(8);
    for (int i = 0; i < 8; i++) {
      Index3D neighbor_index =
          block_index + marching_cubes::directionFromNeighborIndex(i);
      neighbor_blocks[i] = distance_layer.getBlockAtIndex(neighbor_index);
    }

    // Get all the potential triangles:
    std::vector<marching_cubes::PerVoxelMarchingCubesResults>
        triangle_candidates;
    getTriangleCandidatesInBlock(block, neighbor_blocks, block_index,
                                 block_size, &triangle_candidates);

    if (triangle_candidates.size() <= 0) {
      continue;
    }

    // Allocate the mesh block.
    MeshBlock::Ptr mesh_block = mesh_layer->allocateBlockAtIndexAsync(block_index, *cuda_stream_);

    // Then actually calculate the triangles.
    for (const marching_cubes::PerVoxelMarchingCubesResults& candidate :
         triangle_candidates) {
      marching_cubes::meshCube(candidate, mesh_block.get());
    }
  }

  return true;
}

bool MeshIntegrator::isBlockMeshable(
    const VoxelBlock<TsdfVoxel>::ConstPtr block, float cutoff) const {
  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;

  Index3D voxel_index;
  // Iterate over all the voxels:
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        const TsdfVoxel* voxel =
            &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

        // Check if voxel distance is within the cutoff to determine if
        // there's going to be a surface boundary in this block.
        if (voxel->weight >= min_weight_ &&
            std::abs(voxel->distance) <= cutoff) {
          return true;
        }
      }
    }
  }
  return false;
}

void MeshIntegrator::getTriangleCandidatesInBlock(
    const TsdfBlock::ConstPtr block,
    const std::vector<TsdfBlock::ConstPtr>& neighbor_blocks,
    const Index3D& block_index, const float block_size,
    std::vector<marching_cubes::PerVoxelMarchingCubesResults>*
        triangle_candidates) {
  CHECK_NOTNULL(block);
  CHECK_NOTNULL(triangle_candidates);

  constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
  const float voxel_size = block_size / kVoxelsPerSide;

  Index3D voxel_index;
  // Iterate over all inside voxels.
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        // Get the position of this voxel.
        Vector3f voxel_position = getCenterPositionFromBlockIndexAndVoxelIndex(
            block_size, block_index, voxel_index);

        marching_cubes::PerVoxelMarchingCubesResults neighbors;
        // Figure out if this voxel is actually a triangle candidate.
        if (getTriangleCandidatesAroundVoxel(block, neighbor_blocks,
                                             voxel_index, voxel_position,
                                             voxel_size, &neighbors)) {
          triangle_candidates->push_back(neighbors);
        }
      }
    }
  }
}

bool MeshIntegrator::getTriangleCandidatesAroundVoxel(
    const TsdfBlock::ConstPtr block,
    const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
    const Index3D& voxel_index, const Vector3f& voxel_position,
    const float voxel_size,
    marching_cubes::PerVoxelMarchingCubesResults* neighbors) {
  DCHECK_EQ(neighbor_blocks.size(), 8);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  for (unsigned int i = 0; i < 8; ++i) {
    Index3D corner_index = voxel_index + cube_index_offsets_.col(i);

    // Are we in bounds? If not, have to get a neighbor.
    // The neighbor should correspond to the index in neighbor blocks.

    bool search_neighbor = false;
    Index3D block_offset(0, 0, 0);
    for (int j = 0; j < 3; j++) {
      if (corner_index[j] >= kVoxelsPerSide) {
        // Here the index is too much.
        corner_index(j) -= kVoxelsPerSide;
        block_offset(j) = 1;
        search_neighbor = true;
      }
    }

    const TsdfVoxel* voxel = nullptr;
    // Get the voxel either from the current block or from the corresponding
    // neighbor.
    if (search_neighbor) {
      // Find the correct neighbor block.
      int neighbor_index =
          marching_cubes::neighborIndexFromDirection(block_offset);
      if (neighbor_blocks[neighbor_index] == nullptr) {
        return false;
      }
      voxel =
          &neighbor_blocks[neighbor_index]
               ->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
    } else {
      voxel =
          &block->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
    }

    // If any of the neighbors are not observed, this can't be a mesh
    // triangle.
    if (voxel->weight < min_weight_) {
      return false;
    }
    neighbors->vertex_sdf[i] = voxel->distance;
    neighbors->vertex_coords[i] =
        voxel_position + voxel_size * cube_index_offsets_.col(i).cast<float>();
  }

  // Figure out the index if we've made it this far.
  neighbors->marching_cubes_table_index =
      marching_cubes::calculateVertexConfiguration(neighbors->vertex_sdf);

  // Index 0 & 255 contain no triangles so not worth outputting it.
  return neighbors->marching_cubes_table_index != 0 &&
         neighbors->marching_cubes_table_index != 255;
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

  block_ptrs_device_.copyFromAsync(block_ptrs_host_, *cuda_stream_);

  // Allocate a device vector that holds the meshable result.
  meshable_device_.resizeAsync(block_indices.size(), *cuda_stream_);
  meshable_device_.setZeroAsync(*cuda_stream_);

  isBlockMeshableKernel<<<dim_block, dim_threads, 0, *cuda_stream_>>>(
      block_indices.size(), block_ptrs_device_.data(), cutoff_distance,
      min_weight_, meshable_device_.data());

  checkCudaErrors(cudaPeekAtLastError());

  meshable_host_.copyFromAsync(meshable_device_, *cuda_stream_);
  cuda_stream_->synchronize();

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
  mesh_blocks_host_.setZeroAsync(*cuda_stream_);

  block_ptrs_device_.copyFromAsync(block_ptrs_host_, *cuda_stream_);
  block_positions_device_.copyFromAsync(block_positions_host_, *cuda_stream_);

  // Allocate working space
  constexpr int kNumVoxelsPerBlock =
      kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;
  marching_cubes_results_device_.resizeAsync(
      block_indices.size() * kNumVoxelsPerBlock, *cuda_stream_);
  marching_cubes_results_device_.setZeroAsync(*cuda_stream_);
  mesh_block_sizes_device_.resizeAsync(block_indices.size(), *cuda_stream_);
  mesh_block_sizes_device_.setZeroAsync(*cuda_stream_);
  mesh_prep_timer.Stop();

  // Run the first half of marching cubes and calculate:
  // - the per-vertex indexes into the magic triangle table
  // - the number of vertices in each mesh block.
  timing::Timer mesh_kernel_1_timer("mesh/gpu/mesh_blocks/kernel_table");
  meshBlocksCalculateTableIndicesKernel<<<dim_block, dim_threads, 0,
                                          *cuda_stream_>>>(
      block_indices.size(), block_ptrs_device_.data(),
      block_positions_device_.data(), voxel_size, min_weight_,
      marching_cubes_results_device_.data(), mesh_block_sizes_device_.data());
  checkCudaErrors(cudaPeekAtLastError());
  cuda_stream_->synchronize();

  mesh_kernel_1_timer.Stop();

  // Copy back the new mesh block sizes (so we can allocate space)
  timing::Timer mesh_copy_timer("mesh/gpu/mesh_blocks/copy_out");
  mesh_block_sizes_host_.copyFromAsync(mesh_block_sizes_device_, *cuda_stream_);
  cuda_stream_->synchronize();
  mesh_copy_timer.Stop();

  // Allocate mesh blocks
  timing::Timer mesh_allocation_timer("mesh/gpu/mesh_blocks/block_allocation");
  for (size_t i = 0; i < block_indices.size(); i++) {
    const size_t num_vertices = mesh_block_sizes_host_[i];

    if (num_vertices > 0) {
      MeshBlock::Ptr output_block = mesh_layer->allocateBlockAtIndexAsync(
          block_indices[i], *cuda_stream_);
      if (output_block == nullptr) {
        continue;
      }
      // Grow the vector with a growth factor and a minimum allocation to avoid
      // repeated reallocation
      if (num_vertices > output_block->capacity()) {
        constexpr size_t kMinimumMeshBlockTrianglesPerVoxel = 1;
        constexpr size_t kMinimumMeshBlockVertices =
            kNumVoxelsPerBlock * kMinimumMeshBlockTrianglesPerVoxel * 3;
        constexpr size_t kMeshBlockOverallocationFactor = 2;
        const int num_vertices_to_allocate =
            std::max(kMinimumMeshBlockVertices,
                     num_vertices * kMeshBlockOverallocationFactor);
        output_block->vertices.reserve(num_vertices_to_allocate);
        output_block->normals.reserve(num_vertices_to_allocate);
        output_block->triangles.reserve(num_vertices_to_allocate);
      }
      output_block->vertices.resize(num_vertices);
      output_block->normals.resize(num_vertices);
      output_block->triangles.resize(num_vertices);
      mesh_blocks_host_[i] = CudaMeshBlock(output_block.get());
    }
  }
  mesh_blocks_device_.copyFromAsync(mesh_blocks_host_, *cuda_stream_);
  mesh_allocation_timer.Stop();

  // Run the second half of marching cubes
  // - Translating the magic table indices into triangle vertices and writing
  //   them into the mesh layer.
  timing::Timer mesh_kernel_2_timer("mesh/gpu/mesh_blocks/kernel_vertices");
  meshBlocksCalculateVerticesKernel<<<dim_block, dim_threads, 0,
                                      *cuda_stream_>>>(
      block_indices.size(), marching_cubes_results_device_.data(),
      mesh_block_sizes_device_.data(), mesh_blocks_device_.data());
  checkCudaErrors(cudaPeekAtLastError());
  cuda_stream_->synchronize();
  mesh_kernel_2_timer.Stop();

  // Optional third stage: welding.
  if (weld_vertices_) {
    timing::Timer welding_timer("mesh/gpu/mesh_blocks/welding");
    weldVertices(&mesh_blocks_device_);

    mesh_blocks_host_.copyFromAsync(mesh_blocks_device_, *cuda_stream_);
    cuda_stream_->synchronize();

    // Set the sizes on CPU :(
    for (size_t i = 0; i < block_indices.size(); i++) {
      size_t new_size = mesh_blocks_host_[i].vertices_size;
      MeshBlock::Ptr output_block =
          mesh_layer->getBlockAtIndex(block_indices[i]);
      if (output_block == nullptr) {
        continue;
      }
      output_block->vertices.resize(new_size);
      output_block->normals.resize(new_size);
    }
  }
}

template <int kBlockThreads, int kItemsPerThread>
__global__ void weldVerticesCubKernel(CudaMeshBlock* mesh_blocks) {
  // First get the correct block for this.
  int block_index = blockIdx.x;
  CudaMeshBlock* block = &mesh_blocks[block_index];
  int num_vals = block->vertices_size;
  if (num_vals <= 0) {
    return;
  }

  // Check if there are too many vertices in the block to weld.
  // If there are give up on this block.
  if (num_vals >= kBlockThreads * kItemsPerThread) {
    // Note(alexmillane): We used to print a warning here, but it would spam the
    // console. For now we're abandoning providing feedback to the user about
    // blocks failing to weld as it's not a critical issue, the blocks just
    // retain their original number of vertices.
    return;
  }

  // Create all the storage needed for CUB operations. :)
  constexpr int kValueScale = 1000;
  typedef uint64_t VertexPositionHashValue;
  typedef int VertexIndex;
  typedef cub::BlockRadixSort<VertexPositionHashValue, kBlockThreads,
                              kItemsPerThread, VertexIndex>
      BlockRadixSortT;
  typedef cub::BlockDiscontinuity<VertexPositionHashValue, kBlockThreads>
      BlockDiscontinuityT;
  typedef cub::BlockScan<VertexIndex, kBlockThreads> BlockScanT;

  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
    typename BlockRadixSortT::TempStorage sort;
    typename BlockDiscontinuityT::TempStorage discontinuity;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  __shared__ int output_index;
  if (threadIdx.x == 0) {
    output_index = 0;
  }

  // First we create a values list which is actually the indicies.
  // Obtain this block's segment of consecutive keys (blocked across threads)
  uint64_t thread_keys[kItemsPerThread];
  Vector3f thread_values[kItemsPerThread];
  Vector3f thread_normals[kItemsPerThread];
  int thread_inds[kItemsPerThread];
  int head_flags[kItemsPerThread];
  int head_indices[kItemsPerThread];
  int thread_offset = threadIdx.x * kItemsPerThread;

  // Fill in the keys from the values.
  // I guess we can just do a for loop. kItemsPerThread should be fairly small.
  Index3DHash index_hash;
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i >= num_vals) {
      // We just pack the key with a large value.
      thread_values[i] = Vector3f::Zero();
      thread_keys[i] = SIZE_MAX;
      thread_inds[i] = -1;
    } else {
      thread_values[i] = block->vertices[thread_offset + i];
      thread_keys[i] = index_hash(Index3D(thread_values[i].x() * kValueScale,
                                          thread_values[i].y() * kValueScale,
                                          thread_values[i].z() * kValueScale));
      thread_inds[i] = block->triangles[thread_offset + i];
    }
  }

  // We then sort the values.
  __syncthreads();
  // Collectively sort the keys
  BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_inds);
  __syncthreads();
  // We remove duplicates by find when the discontinuities happen.
  BlockDiscontinuityT(temp_storage.discontinuity)
      .FlagHeads(head_flags, thread_keys, cub::Inequality());
  __syncthreads();
  // Get the indices that'll be assigned to the new unique values.
  BlockScanT(temp_storage.scan)
      .InclusiveSum<kItemsPerThread>(head_flags, head_indices);
  __syncthreads();

  // Cool now write only 1 instance of the unique entries to the output.
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i < num_vals) {
      if (head_flags[i] == 1) {
        // Get the proper value out. Cache this for in-place ops next step.
        thread_values[i] = block->vertices[thread_inds[i]];
        thread_normals[i] = block->normals[thread_inds[i]];
        atomicMax(&output_index, head_indices[i]);
      }
      // For the key of each initial vertex, we find what index it now has.
      block->triangles[thread_inds[i]] = head_indices[i] - 1;
    }
  }
  __syncthreads();

  // Have to do this twice since we do this in-place. Now actually replace
  // the values.
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i < num_vals) {
      if (head_flags[i] == 1) {
        // Get the proper value out.
        block->vertices[head_indices[i] - 1] = thread_values[i];
        block->normals[head_indices[i] - 1] = thread_normals[i];
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block->vertices_size = output_index;
  }
}

void MeshIntegrator::weldVertices(
    device_vector<CudaMeshBlock>* cuda_mesh_blocks) {
  if (cuda_mesh_blocks->size() == 0) {
    return;
  }
  // Together this should be >> the max number of vertices in the mesh.
  constexpr int kNumThreads = 128;
  constexpr int kNumItemsPerThread = 20;
  weldVerticesCubKernel<kNumThreads, kNumItemsPerThread>
      <<<cuda_mesh_blocks->size(), kNumThreads, 0, *cuda_stream_>>>(
          cuda_mesh_blocks->data());
  cuda_stream_->synchronize();
}

parameters::ParameterTreeNode MeshIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "mesh_integrator" : name_remap;
  return ParameterTreeNode(
      name, {
                ParameterTreeNode("min_weight:", min_weight_),
                ParameterTreeNode("cutoff_distance_vox:", cutoff_distance_vox_),
                ParameterTreeNode("weld_vertices:", weld_vertices_),
            });
}

}  // namespace nvblox
