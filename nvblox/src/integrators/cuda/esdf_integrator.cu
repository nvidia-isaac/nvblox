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
#include "nvblox/gpu_hash/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/cuda/gpu_indexing.cuh"
#include "nvblox/utils/timing.h"

#include "nvblox/integrators/esdf_integrator.h"

namespace nvblox {

EsdfIntegrator::~EsdfIntegrator() {
  if (cuda_stream_ != nullptr) {
    cudaStreamDestroy(cuda_stream_);
  }
}

void EsdfIntegrator::integrateBlocksOnGPU(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    EsdfLayer* esdf_layer) {
  timing::Timer esdf_timer("esdf/integrate");

  if (block_indices.empty()) {
    return;
  }

  // First, check if the stream exists. If not, create one.
  if (cuda_stream_ == nullptr) {
    checkCudaErrors(cudaStreamCreate(&cuda_stream_));
  }

  timing::Timer allocate_timer("esdf/integrate/allocate");
  // First, allocate all the destination blocks.
  allocateBlocksOnCPU(block_indices, esdf_layer);
  allocate_timer.Stop();

  timing::Timer mark_timer("esdf/integrate/mark_sites");
  // Then, mark all the sites on GPU.
  // This finds all the blocks that are eligible to be parents.
  std::vector<Index3D> blocks_with_sites;
  std::vector<Index3D> blocks_to_clear;
  markAllSitesOnGPU(tsdf_layer, block_indices, esdf_layer, &blocks_with_sites,
                    &blocks_to_clear);
  mark_timer.Stop();

  std::vector<Index3D> cleared_blocks;
  if (!blocks_to_clear.empty()) {
    timing::Timer compute_timer("esdf/integrate/clear");
    clearInvalidOnGPU(blocks_to_clear, esdf_layer, &cleared_blocks);
    std::vector<Index3D> all_clear_updated;
  }

  timing::Timer compute_timer("esdf/integrate/compute");
  // Parallel block banding on GPU.
  computeEsdfOnGPU(blocks_with_sites, esdf_layer);
  if (!cleared_blocks.empty()) {
    computeEsdfOnGPU(cleared_blocks, esdf_layer);
  }
  compute_timer.Stop();
}

void EsdfIntegrator::integrateSliceOnGPU(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    float z_min, float z_max, float z_output, EsdfLayer* esdf_layer) {
  timing::Timer esdf_timer("esdf/integrate_slice");

  if (block_indices.empty()) {
    return;
  }

  // First, check if the stream exists. If not, create one.
  if (cuda_stream_ == nullptr) {
    checkCudaErrors(cudaStreamCreate(&cuda_stream_));
  }

  timing::Timer allocate_timer("esdf/integrate_slice/allocate");
  // First, allocate all the destination blocks.
  allocateBlocksOnCPU(block_indices, esdf_layer);
  allocate_timer.Stop();

  timing::Timer mark_timer("esdf/integrate_slice/mark_sites");
  // Then, mark all the sites on GPU.
  // This finds all the blocks that are eligible to be parents.
  std::vector<Index3D> blocks_with_sites;
  std::vector<Index3D> blocks_to_clear;
  markSitesInSliceOnGPU(tsdf_layer, block_indices, z_min, z_max, z_output,
                        esdf_layer, &blocks_with_sites, &blocks_to_clear);
  mark_timer.Stop();

  std::vector<Index3D> cleared_blocks;
  if (!blocks_to_clear.empty()) {
    timing::Timer compute_timer("esdf/integrate_slice/clear");
    clearInvalidOnGPU(blocks_to_clear, esdf_layer, &cleared_blocks);
    std::vector<Index3D> all_clear_updated;
  }

  timing::Timer compute_timer("esdf/integrate_slice/compute");
  // Parallel block banding on GPU.
  computeEsdfOnGPU(blocks_with_sites, esdf_layer);

  if (!cleared_blocks.empty()) {
    computeEsdfOnGPU(cleared_blocks, esdf_layer);
  }
  compute_timer.Stop();
}

__device__ void clearVoxelDevice(EsdfVoxel* voxel,
                                 float max_squared_distance_vox) {
  voxel->parent_direction.setZero();
  voxel->squared_distance_vox = max_squared_distance_vox;
}

// Takes in a vector of blocks, and outputs an integer true if that block is
// meshable.
// Block size MUST be voxels_per_side x voxels_per_side x voxel_per_size.
// Grid size can be anything.
__global__ void markAllSitesKernel(int num_blocks,
                                   const TsdfBlock** tsdf_blocks,
                                   EsdfBlock** esdf_blocks,
                                   float max_site_distance_m, float min_weight,
                                   float max_squared_distance_vox,
                                   bool* updated, bool* to_clear) {
  dim3 voxel_index = threadIdx;
  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Get the correct voxel for this index.
    const TsdfVoxel* tsdf_voxel =
        &tsdf_blocks[block_index]
             ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
    EsdfVoxel* esdf_voxel =
        &esdf_blocks[block_index]
             ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
    if (tsdf_voxel->weight >= min_weight) {
      // Mark as inside if the voxel distance is negative.
      bool is_inside = tsdf_voxel->distance <= 0.0f;
      if (esdf_voxel->is_inside && is_inside == false) {
        clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
        to_clear[block_index] = true;
      }
      esdf_voxel->is_inside = is_inside;
      if (is_inside && fabsf(tsdf_voxel->distance) <= max_site_distance_m) {
        esdf_voxel->is_site = true;
        esdf_voxel->squared_distance_vox = 0.0f;
        esdf_voxel->parent_direction.setZero();
        updated[block_index] = true;
      } else {
        if (esdf_voxel->is_site) {
          esdf_voxel->is_site = false;
          // This voxel needs to be cleared.
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
          to_clear[block_index] = true;
        } else if (!esdf_voxel->observed) {
          // This is a brand new voxel.
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
        } else if (esdf_voxel->squared_distance_vox <= 1e-4) {
          // This is an invalid voxel that should be cleared.
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
          to_clear[block_index] = true;
        }
      }
      esdf_voxel->observed = true;
    }
  }
}

// From:
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMax((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

/// Thread size MUST be 8x8x8, block size can be anything.
__global__ void markSitesInSliceKernel(
    int num_input_blocks, int num_output_blocks, const TsdfBlock** tsdf_blocks,
    EsdfBlock** esdf_blocks, int output_voxel_index, int input_min_voxel_index,
    int input_max_voxel_index, float max_site_distance_m, float min_weight,
    float max_squared_distance_vox, bool* updated, bool* cleared) {
  dim3 voxel_index = threadIdx;
  voxel_index.z = output_voxel_index;
  int layer_index = threadIdx.z;
  int num_layers = blockDim.z;

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  __shared__ EsdfVoxel new_values[kVoxelsPerSide][kVoxelsPerSide];
  __shared__ bool observed[kVoxelsPerSide][kVoxelsPerSide];
  __shared__ float min_distance[kVoxelsPerSide][kVoxelsPerSide];

  // Initialize these.
  if (layer_index == 0) {
    observed[voxel_index.x][voxel_index.y] = false;
    min_distance[voxel_index.x][voxel_index.y] = 100.0f;
  }
  __syncthreads();

  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_output_blocks;
       block_index += gridDim.x) {
    // Get the correct block for this.
    const TsdfBlock* tsdf_block =
        tsdf_blocks[block_index + num_output_blocks * layer_index];
    // There's also null pointers in there.
    if (tsdf_block != nullptr) {
      // Iterate over all of the voxels in this block.
      int start_index = 0;
      int end_index = kVoxelsPerSide;
      if (layer_index == 0) {
        start_index = input_min_voxel_index;
      }
      if (layer_index == num_layers - 1) {
        end_index = input_max_voxel_index;
      }
      for (int i = start_index; i < end_index; i++) {
        const TsdfVoxel* tsdf_voxel =
            &tsdf_block->voxels[voxel_index.x][voxel_index.y][i];
        // EsdfVoxel* new_voxel = &new_values[voxel_index.x][voxel_index.y];
        // Get the correct voxel for this index.
        if (tsdf_voxel->weight >= min_weight) {
          observed[voxel_index.x][voxel_index.y] = true;
          atomicMinFloat(&min_distance[voxel_index.x][voxel_index.y],
                         tsdf_voxel->distance);
        }
      }
    }

    // sync threads across everyone trying to update this voxel
    __syncthreads();

    // Ok now only if we're layer 0 do we compare the new and old values and
    // decide what to output.
    if (layer_index == 0) {
      EsdfVoxel* esdf_voxel =
          &esdf_blocks[block_index]
               ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];

      // Case 0: Just skip it if it's unobserved. We don't care.
      if (!observed[voxel_index.x][voxel_index.y]) {
        continue;
      }
      // Determine if the new value puts us inside or in a site.
      bool is_inside = min_distance[voxel_index.x][voxel_index.y] <= 0.0f;
      bool is_site = fabsf(min_distance[voxel_index.x][voxel_index.y]) <=
                         max_site_distance_m &&
                     is_inside;

      // First handle the case where the voxel is a site.
      if (is_site) {
        if (esdf_voxel->is_site) {
          // Ok whatever. Add to the site list.
          // Its existing values are fine.
          updated[block_index] = true;
        } else {
          // Wasn't a site before, is now.
          esdf_voxel->observed = true;
          esdf_voxel->is_site = true;
          clearVoxelDevice(esdf_voxel, 0.0f);
          updated[block_index] = true;
        }
      } else {
        // Here we have to double-check what's going on.
        // If it was a site before, and isn't anymore, we have to clear it.
        if (esdf_voxel->is_site) {
          esdf_voxel->is_site = false;
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
          cleared[block_index] = true;
        }
        // Otherwise just leave it alone unless it's brand new.
        if (!esdf_voxel->observed) {
          esdf_voxel->observed = true;
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
        } else if (esdf_voxel->is_inside != is_inside) {
          // In case the sidedness swapped, clear the voxel.
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
          cleared[block_index] = true;
        } else if (esdf_voxel->squared_distance_vox <= 0.0f) {
          // This is somehow invalidly marked as a site despite the fact
          // it shouldn't be.
          clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
          cleared[block_index] = true;
        }
      }
      // Make the sidedness match.
      esdf_voxel->is_inside = is_inside;
    }
  }
}

__device__ void sweepSingleBand(Index3D voxel_index, int sweep_axis,
                                float max_squared_distance_vox,
                                EsdfBlock* esdf_block) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  Index3D last_site;
  bool site_found;
  // Sweep sweep sweep.
  // First we sweep forward, then backwards.
  for (int i = 0; i < 2; i++) {
    last_site = Index3D::Zero();
    site_found = false;
    int direction = 1;
    int start_voxel = 0;
    int end_voxel = kVoxelsPerSide;
    if (i == 1) {
      direction = -1;
      start_voxel = kVoxelsPerSide - 1;
      end_voxel = -1;
    }

    for (voxel_index(sweep_axis) = start_voxel;
         voxel_index(sweep_axis) != end_voxel;
         voxel_index(sweep_axis) += direction) {
      EsdfVoxel* esdf_voxel =
          &esdf_block
               ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
      if (!esdf_voxel->observed) {
        continue;
      }
      // If this voxel is itself a site, then mark this for future voxels.
      if (esdf_voxel->is_site) {
        last_site = voxel_index;
        site_found = true;
      } else if (!site_found) {
        // If this voxel isn't a site but we haven't found a site yet,
        // then if this voxel is valid we set it as the site.
        if (esdf_voxel->squared_distance_vox < max_squared_distance_vox) {
          site_found = true;
          last_site = esdf_voxel->parent_direction + voxel_index;
        }
      } else {
        // If we've found the site, then should just decide what to do
        // here.
        Index3D potential_direction = last_site - voxel_index;
        float potential_distance = potential_direction.squaredNorm();
        // Either it hasn't been set at all or it's closer to the site
        // than to its current value.
        if (esdf_voxel->squared_distance_vox > potential_distance) {
          esdf_voxel->parent_direction = potential_direction;
          esdf_voxel->squared_distance_vox = potential_distance;
        } else if (esdf_voxel->squared_distance_vox <
                   max_squared_distance_vox) {
          // If the current value is a better site, then set it as a site.
          last_site = esdf_voxel->parent_direction + voxel_index;
        }
      }
    }
  }
}
__device__ bool updateSingleNeighbor(const EsdfBlock* esdf_block,
                                     const Index3D& voxel_index,
                                     const Index3D& neighbor_voxel_index,
                                     int axis, int direction,
                                     float max_squared_distance_vox,
                                     EsdfBlock* neighbor_block) {
  const EsdfVoxel* esdf_voxel =
      &esdf_block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  EsdfVoxel* neighbor_voxel =
      &neighbor_block
           ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                   [neighbor_voxel_index.z()];
  if (!esdf_voxel->observed || !neighbor_voxel->observed ||
      neighbor_voxel->is_site ||
      esdf_voxel->squared_distance_vox >= max_squared_distance_vox) {
    return false;
  }
  // Determine if we can update this.
  Eigen::Vector3i potential_direction = esdf_voxel->parent_direction;
  potential_direction(axis) -= direction;
  float potential_distance = potential_direction.squaredNorm();
  // TODO: might be some concurrency issues here, have to be a bit careful
  // on the corners/edges.
  if (neighbor_voxel->squared_distance_vox > potential_distance) {
    neighbor_voxel->parent_direction = potential_direction;
    neighbor_voxel->squared_distance_vox = potential_distance;
    return true;
  }
  return false;
}

__device__ bool clearSingleNeighbor(const EsdfBlock* esdf_block,
                                    const Index3D& voxel_index,
                                    const Index3D& neighbor_voxel_index,
                                    int axis, int direction,
                                    float max_squared_distance_vox,
                                    EsdfBlock* neighbor_block) {
  const EsdfVoxel* esdf_voxel =
      &esdf_block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  EsdfVoxel* neighbor_voxel =
      &neighbor_block
           ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                   [neighbor_voxel_index.z()];

  if (esdf_voxel->squared_distance_vox < max_squared_distance_vox ||
      !esdf_voxel->observed || neighbor_voxel->is_site ||
      neighbor_voxel->squared_distance_vox >= max_squared_distance_vox) {
    return false;
  }
  // Determine if we can update this.
  Index3D parent_voxel_dir = neighbor_voxel->parent_direction;
  if ((direction > 0 && parent_voxel_dir(axis) > 0) ||
      (direction < 0 && parent_voxel_dir(axis) < 0)) {
    return false;
  }

  clearVoxelDevice(neighbor_voxel, max_squared_distance_vox);
  return true;
}

/// Thread size MUST be 8x8xN (where N is a number of blocks up to 8), block
/// size can be anything.
__global__ void sweepBlockBandKernel(int num_blocks, EsdfBlock** esdf_blocks,
                                     float max_squared_distance_vox) {
  // We go one axis at a time, syncing threads in between.
  dim3 thread_index = threadIdx;
  thread_index.z = 0;

  for (int block_index = blockIdx.x * blockDim.z + threadIdx.z;
       block_index < num_blocks; block_index += gridDim.x * blockDim.z) {
    // For simplicity we have to have the same number of blocks in the CUDA
    // kernel call as we have actual blocks.
    EsdfBlock* esdf_block = esdf_blocks[block_index];
    Index3D voxel_index(0, thread_index.x, thread_index.y);

    // X axis done.
    sweepSingleBand(voxel_index, 0, max_squared_distance_vox, esdf_block);
    __syncthreads();

    // Y axis done.
    voxel_index << thread_index.x, 0, thread_index.y;
    sweepSingleBand(voxel_index, 1, max_squared_distance_vox, esdf_block);
    __syncthreads();

    // Z axis done.
    voxel_index << thread_index.x, thread_index.y, 0;
    sweepSingleBand(voxel_index, 2, max_squared_distance_vox, esdf_block);
    __syncthreads();
  }
}

/// Thread size MUST be 8x8xN, where N is the number of blocks processed at
/// a time, block size can be anything.
__global__ void updateLocalNeighborBandsKernel(int num_blocks, int i,
                                               EsdfBlock** esdf_blocks,
                                               int* neighbor_table,
                                               EsdfBlock** neighbor_pointers,
                                               float max_squared_distance_vox,
                                               bool* updated_neighbors) {
  // We go one axis at a time, syncing threads in between.
  dim3 thread_index = threadIdx;
  thread_index.z = 0;

  constexpr int kNumNeighbors = 6;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  for (int block_index = blockIdx.x * blockDim.z + threadIdx.z;
       block_index < num_blocks; block_index += gridDim.x * blockDim.z) {
    EsdfBlock* esdf_block = esdf_blocks[block_index];
    Index3D voxel_index;
    Index3D neighbor_voxel_index;
    // Each thread updates 1 neighbors, set by "i".
    // Get the neighbor block.
    int neighbor_index = neighbor_table[block_index * kNumNeighbors + i];
    if (neighbor_index < 0) {
      continue;
    }
    EsdfBlock* neighbor_block = neighbor_pointers[neighbor_index];
    // Now we have the neighbor block... Let's figure out which voxels we
    // should look at.
    int axis = i / 2;
    int direction = i % 2 ? -1 : 1;

    // Fill in the axes.
    if (axis == 0) {
      voxel_index << 0, thread_index.x, thread_index.y;
    } else if (axis == 1) {
      voxel_index << thread_index.x, 0, thread_index.y;
    } else if (axis == 2) {
      voxel_index << thread_index.x, thread_index.y, 0;
    }
    neighbor_voxel_index = voxel_index;
    // If we're looking backwards...
    if (direction < 0) {
      voxel_index(axis) = 0;
      neighbor_voxel_index(axis) = kVoxelsPerSide - 1;
    } else {
      voxel_index(axis) = kVoxelsPerSide - 1;
      neighbor_voxel_index(axis) = 0;
    }

    bool updated = updateSingleNeighbor(
        esdf_block, voxel_index, neighbor_voxel_index, axis, direction,
        max_squared_distance_vox, neighbor_block);
    if (updated) {
      updated_neighbors[neighbor_index] = true;
    }
  }
}

/// Thread size MUST be 8x8x8, block size can be anything.
__global__ void clearWithinBlockKernel(int num_blocks, EsdfBlock** esdf_blocks,
                                       float max_squared_distance_vox) {
  dim3 voxel_index = threadIdx;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  // Allow block size to be whatever.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Get the voxel.
    EsdfVoxel* esdf_voxel =
        &esdf_blocks[block_index]
             ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
    // Check if its parent is in the same block.
    if (!esdf_voxel->observed || esdf_voxel->is_site ||
        esdf_voxel->squared_distance_vox >= max_squared_distance_vox) {
      continue;
    }
    // Get the parent.
    Index3D parent_index =
        Index3D(voxel_index.x, voxel_index.y, voxel_index.z) +
        esdf_voxel->parent_direction;

    // Check if the voxel is within the same block.
    if (parent_index.x() < 0 || parent_index.x() >= kVoxelsPerSide ||
        parent_index.y() < 0 || parent_index.y() >= kVoxelsPerSide ||
        parent_index.z() < 0 || parent_index.z() >= kVoxelsPerSide) {
      continue;
    }

    // Ok check if the parent index is a site.
    if (!esdf_blocks[block_index]
             ->voxels[parent_index.x()][parent_index.y()][parent_index.z()]
             .is_site) {
      clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
    }
  }
}

/// Thread size MUST be 8x8x8, block size can be anything.
__global__ void clearInternalVoxelsKernel(int num_blocks,
                                          EsdfBlock** esdf_blocks,
                                          float max_squared_distance_vox) {
  dim3 voxel_index = threadIdx;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  // Allow block size to be whatever.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Get the voxel.
    EsdfVoxel* esdf_voxel =
        &esdf_blocks[block_index]
             ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
    if (!esdf_voxel->observed || esdf_voxel->is_site ||
        esdf_voxel->squared_distance_vox >= max_squared_distance_vox) {
      continue;
    }
    // Get the parent.
    Index3D parent_index =
        Index3D(voxel_index.x, voxel_index.y, voxel_index.z) +
        esdf_voxel->parent_direction;

    // Check if we're our own parent. This is definitely wrong since we're not
    // a site.
    if (esdf_voxel->parent_direction.isZero()) {
      clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
      continue;
    }

    // Get the closest index to the parent within the same block.
    // We just get the nearest neighbor.
    Index3D closest_parent(min(max(parent_index.x(), 0), kVoxelsPerSide - 1),
                           min(max(parent_index.y(), 0), kVoxelsPerSide - 1),
                           min(max(parent_index.z(), 0), kVoxelsPerSide - 1));

    // Ok check if the parent index is a site.
    // TODO: Check if we need the observed rule or not...
    const EsdfVoxel& neighbor_voxel =
        esdf_blocks[block_index]->voxels[closest_parent.x()][closest_parent.y()]
                                        [closest_parent.z()];
    if (!neighbor_voxel.observed ||
        neighbor_voxel.squared_distance_vox >= max_squared_distance_vox) {
      clearVoxelDevice(esdf_voxel, max_squared_distance_vox);
    }
  }
}

/// Thread size MUST be 8x8xN, where N is the number of blocks processed at
/// a time, block size can be anything.
__global__ void clearLocalNeighborBandsKernel(int num_blocks, int i,
                                              EsdfBlock** esdf_blocks,
                                              int* neighbor_table,
                                              EsdfBlock** neighbor_pointers,
                                              float max_squared_distance_vox,
                                              bool* updated_neighbors) {
  // We go one axis at a time, syncing threads in between.
  dim3 thread_index = threadIdx;
  thread_index.z = 0;

  constexpr int kNumNeighbors = 6;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  for (int block_index = blockIdx.x * blockDim.z + threadIdx.z;
       block_index < num_blocks; block_index += gridDim.x * blockDim.z) {
    EsdfBlock* esdf_block = esdf_blocks[block_index];
    Index3D voxel_index;
    Index3D neighbor_voxel_index;
    // Each thread updates 1 neighbors, set by "i".
    // Get the neighbor block.
    int neighbor_index = neighbor_table[block_index * kNumNeighbors + i];
    if (neighbor_index < 0) {
      continue;
    }
    EsdfBlock* neighbor_block = neighbor_pointers[neighbor_index];
    // Now we have the neighbor block... Let's figure out which voxels we
    // should look at.
    int axis = i / 2;
    int direction = i % 2 ? -1 : 1;

    // Fill in the axes.
    if (axis == 0) {
      voxel_index << 0, thread_index.x, thread_index.y;
    } else if (axis == 1) {
      voxel_index << thread_index.x, 0, thread_index.y;
    } else if (axis == 2) {
      voxel_index << thread_index.x, thread_index.y, 0;
    }
    neighbor_voxel_index = voxel_index;
    // If we're looking backwards...
    if (direction < 0) {
      voxel_index(axis) = 0;
      neighbor_voxel_index(axis) = kVoxelsPerSide - 1;
    } else {
      voxel_index(axis) = kVoxelsPerSide - 1;
      neighbor_voxel_index(axis) = 0;
    }

    bool updated = clearSingleNeighbor(
        esdf_block, voxel_index, neighbor_voxel_index, axis, direction,
        max_squared_distance_vox, neighbor_block);
    if (updated) {
      updated_neighbors[neighbor_index] = true;
    }
  }
}

void EsdfIntegrator::markAllSitesOnGPU(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    EsdfLayer* esdf_layer, std::vector<Index3D>* blocks_with_sites,
    std::vector<Index3D>* cleared_blocks) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(blocks_with_sites);

  // Caching.
  const float voxel_size = tsdf_layer.voxel_size();
  const float max_distance_vox = max_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;
  // Cache the minimum distance in metric size.
  const float max_site_distance_m = max_site_distance_vox_ * voxel_size;

  int num_blocks = block_indices.size();

  // Get all of the block pointers we need.
  tsdf_pointers_host_.resize(num_blocks);
  block_pointers_host_.resize(num_blocks);

  // Have an updated output variable as well.
  updated_blocks_device_.resize(num_blocks);
  updated_blocks_device_.setZero();
  cleared_blocks_device_.resize(num_blocks);
  cleared_blocks_device_.setZero();

  // Populate all the input vectors.
  for (size_t i = 0; i < num_blocks; i++) {
    const Index3D& block_index = block_indices[i];
    EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
    TsdfBlock::ConstPtr tsdf_block = tsdf_layer.getBlockAtIndex(block_index);

    if (!esdf_block || !tsdf_block) {
      LOG(ERROR) << "Somehow trying to update non-existent blocks!";
      continue;
    }

    tsdf_pointers_host_[i] = tsdf_block.get();
    block_pointers_host_[i] = esdf_block.get();
  }

  // Copy what we need over to the device.
  tsdf_pointers_device_ = tsdf_pointers_host_;
  block_pointers_device_ = block_pointers_host_;

  // Call the kernel.
  int dim_block = num_blocks;
  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  markAllSitesKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
      num_blocks, tsdf_pointers_device_.data(), block_pointers_device_.data(),
      max_site_distance_m, min_weight_, max_squared_distance_vox,
      updated_blocks_device_.data(), cleared_blocks_device_.data());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy out.
  updated_blocks_host_ = updated_blocks_device_;
  cleared_blocks_host_ = cleared_blocks_device_;

  // Get the output vector.
  // TODO(helen): swap this to a kernel operation.
  for (size_t i = 0; i < num_blocks; i++) {
    if (updated_blocks_host_[i]) {
      blocks_with_sites->push_back(block_indices[i]);
    }
    if (cleared_blocks_host_[i]) {
      cleared_blocks->push_back(block_indices[i]);
    }
  }
}

// 2D slice version of the markAllSites function above.
void EsdfIntegrator::markSitesInSliceOnGPU(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    float min_z, float max_z, float output_z, EsdfLayer* esdf_layer,
    std::vector<Index3D>* updated_blocks,
    std::vector<Index3D>* cleared_blocks) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(updated_blocks);
  CHECK_NOTNULL(cleared_blocks);

  // Caching.
  const float voxel_size = tsdf_layer.voxel_size();
  const float max_distance_vox = max_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;
  // Cache the minimum distance in metric size.
  const float max_site_distance_m = max_site_distance_vox_ * voxel_size;

  // We are going to subsample the block_indices.
  // We need to figure out all the output blocks, which will be a subset
  // of the input blocks. At the same time we need to get all of the stacks
  // of input blocks at all levels.
  // We are going to pull some "clever" stuff: the input block list will be
  // of length N * n_input_blocks, where "N" is the number of vertical
  // layers there could be that fall into the min z to max z range.

  // Ok first figure out how many layers we could have.
  Index3D min_block_index;
  Index3D min_voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(tsdf_layer.block_size(),
                                           Vector3f(0.0f, 0.0f, min_z),
                                           &min_block_index, &min_voxel_index);
  const int min_block_index_z = min_block_index.z();
  const int min_voxel_index_z = min_voxel_index.z();
  Index3D max_block_index;
  Index3D max_voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(tsdf_layer.block_size(),
                                           Vector3f(0.0f, 0.0f, max_z),
                                           &max_block_index, &max_voxel_index);
  const int max_block_index_z = max_block_index.z();
  const int max_voxel_index_z = max_voxel_index.z();

  // There is always at least 1 layer.
  int num_vertical_layers = max_block_index_z - min_block_index_z + 1;

  // And figure out what the index of the output voxel is.
  // std::pair<Index3D, Index3D> output_block_and_voxel_index
  Index3D output_block_index;
  Index3D output_voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(
      tsdf_layer.block_size(), Vector3f(0.0f, 0.0f, output_z),
      &output_block_index, &output_voxel_index);
  const int output_block_index_z = output_block_index.z();
  const int output_voxel_index_z = output_voxel_index.z();

  // Next get a list of all the valid input blocks.
  Index3DSet output_block_set;
  for (const Index3D& block_index : block_indices) {
    if (block_index.z() >= min_block_index_z &&
        block_index.z() <= max_block_index_z) {
      output_block_set.insert(
          Index3D(block_index.x(), block_index.y(), output_block_index_z));
    }
  }

  // Ok now we have all the indices we actually need.
  // Just have to get their pointers and we're good.
  size_t num_blocks = output_block_set.size();
  if (num_blocks == 0) {
    return;
  }

  std::vector<Index3D> input_blocks(num_blocks * num_vertical_layers);
  std::vector<Index3D> output_blocks(num_blocks);
  tsdf_pointers_host_.resize(num_blocks * num_vertical_layers);
  tsdf_pointers_host_.setZero();
  block_pointers_host_.resize(num_blocks);

  size_t i = 0;
  for (const Index3D& block_index : output_block_set) {
    // This is for the output block, which we allocate along the way.
    output_blocks[i] = block_index;
    block_pointers_host_[i] =
        esdf_layer->allocateBlockAtIndex(block_index).get();

    // Go through all the relevant input pointers:
    Index3D input_block_index = block_index;

    int j = 0;
    for (input_block_index.z() = min_block_index_z;
         input_block_index.z() <= max_block_index_z; input_block_index.z()++) {
      input_blocks[i + num_blocks * j] = input_block_index;
      // This can be null. It's fine.
      tsdf_pointers_host_[i + num_blocks * j] =
          tsdf_layer.getBlockAtIndex(input_block_index).get();
      j++;
    }
    i++;
  }

  // Copy what we need over to the device.
  tsdf_pointers_device_ = tsdf_pointers_host_;
  block_pointers_device_ = block_pointers_host_;

  // Finally, set up the updated and cleared vectors.
  updated_blocks_device_.resize(num_blocks);
  updated_blocks_device_.setZero();
  cleared_blocks_device_.resize(num_blocks);
  cleared_blocks_device_.setZero();

  // Call the kernel!
  int dim_block = num_blocks;
  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, num_vertical_layers);
  markSitesInSliceKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
      num_blocks, num_blocks, tsdf_pointers_device_.data(),
      block_pointers_device_.data(), output_voxel_index_z, min_voxel_index_z,
      max_voxel_index_z, max_site_distance_m, min_weight_,
      max_squared_distance_vox, updated_blocks_device_.data(),
      cleared_blocks_device_.data());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy out.
  updated_blocks_host_ = updated_blocks_device_;
  cleared_blocks_host_ = cleared_blocks_device_;

  // Pack the outputs. The rest of the functions should work as before.
  for (size_t i = 0; i < output_blocks.size(); i++) {
    if (updated_blocks_host_[i]) {
      updated_blocks->push_back(output_blocks[i]);
    }
    if (cleared_blocks_host_[i]) {
      cleared_blocks->push_back(output_blocks[i]);
    }
  }
}

void EsdfIntegrator::clearInvalidOnGPU(
    const std::vector<Index3D>& blocks_to_clear, EsdfLayer* esdf_layer,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(updated_blocks);

  // Caching.
  const float voxel_size = esdf_layer->voxel_size();
  const float max_distance_vox = max_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  int num_blocks = blocks_to_clear.size();
  block_pointers_host_.resize(num_blocks);

  // Have an updated output variable as well.
  updated_blocks_device_.resize(num_blocks);
  updated_blocks_device_.setZero();

  // Populate all the input vectors.
  for (size_t i = 0; i < num_blocks; i++) {
    const Index3D& block_index = blocks_to_clear[i];
    block_pointers_host_[i] = esdf_layer->getBlockAtIndex(block_index).get();
  }

  block_pointers_device_ = block_pointers_host_;

  // Alright now run a kernel to clear all the voxels within a block.
  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  clearWithinBlockKernel<<<num_blocks, dim_threads, 0, cuda_stream_>>>(
      num_blocks, block_pointers_device_.data(), max_squared_distance_vox);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Then clear all the neighbors.
  Index3DSet all_cleared_blocks;
  std::copy(blocks_to_clear.begin(), blocks_to_clear.end(),
            std::inserter(all_cleared_blocks, all_cleared_blocks.end()));

  std::vector<Index3D> clear_list = blocks_to_clear;
  std::vector<Index3D> new_clear_list;
  VLOG(3) << "Blocks to clear: " << blocks_to_clear.size();
  while (!clear_list.empty()) {
    clearBlockNeighbors(clear_list, esdf_layer, &new_clear_list);
    std::copy(new_clear_list.begin(), new_clear_list.end(),
              std::inserter(all_cleared_blocks, all_cleared_blocks.end()));
    std::swap(clear_list, new_clear_list);
    new_clear_list.clear();
    VLOG(3) << "Clear list size: " << clear_list.size();
  }

  for (const Index3D& index : all_cleared_blocks) {
    updated_blocks->push_back(index);
  }
}

void EsdfIntegrator::clearBlockNeighbors(std::vector<Index3D>& clear_list,
                                         EsdfLayer* esdf_layer,
                                         std::vector<Index3D>* new_clear_list) {
  int num_blocks = clear_list.size();

  if (num_blocks == 0) {
    return;
  }
  constexpr int kNumNeighbors = 6;
  const float voxel_size = esdf_layer->voxel_size();
  const float max_distance_vox = max_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;
  dim3 dim_threads_per_voxel(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);

  // Step 0: block pointers.
  block_pointers_host_.resize(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    const Index3D& block_index = clear_list[i];
    block_pointers_host_[i] = esdf_layer->getBlockAtIndex(block_index).get();
  }
  block_pointers_device_ = block_pointers_host_;

  // Step 0a: fix up the blocks so their neighbors are valid.
  clearInternalVoxelsKernel<<<num_blocks, dim_threads_per_voxel, 0,
                              cuda_stream_>>>(
      num_blocks, block_pointers_device_.data(), max_squared_distance_vox);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));

  // Step one: set up the neighbor table.
  std::vector<Index3D> neighbor_indices;
  neighbor_table_host_.resize(num_blocks * kNumNeighbors);
  neighbor_table_host_.setZero();
  neighbor_pointers_host_.resize(0);

  createNeighborTable(clear_list, esdf_layer, &neighbor_indices,
                      &neighbor_pointers_host_, &neighbor_table_host_);

  // Step two: run the neighbor updating kernel.
  updated_blocks_device_.resize(neighbor_indices.size());
  updated_blocks_device_.setZero();

  neighbor_pointers_device_ = neighbor_pointers_host_;
  neighbor_table_device_ = neighbor_table_host_;

  constexpr int kNumBlocksPerCudaBlock = 8;
  int dim_block = std::max(
      static_cast<int>(
          std::ceil(num_blocks / static_cast<float>(kNumBlocksPerCudaBlock))),
      1);
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kNumBlocksPerCudaBlock);
  // We have to do the neighbors one at a time basically for concurrency
  // issues.
  // No clue if the concurrency issues hold for the clearing operation.
  // But this is easier to copy-and-paste.
  for (int i = 0; i < kNumNeighbors; i++) {
    clearLocalNeighborBandsKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
        num_blocks, i, block_pointers_device_.data(),
        neighbor_table_device_.data(), neighbor_pointers_device_.data(),
        max_squared_distance_vox, updated_blocks_device_.data());
  }
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Repack into output vector.
  updated_blocks_host_ = updated_blocks_device_;
  block_pointers_host_.resize(0);

  new_clear_list->clear();
  for (size_t i = 0; i < neighbor_indices.size(); i++) {
    if (updated_blocks_host_[i]) {
      new_clear_list->push_back(neighbor_indices[i]);
      block_pointers_host_.push_back(neighbor_pointers_host_[i]);
    }
  }

  // Step three: clear any remaining voxels on the interior of the blocks
  int num_updated_blocks = new_clear_list->size();
  if (num_updated_blocks == 0) {
    return;
  }

  block_pointers_device_ = block_pointers_host_;
  clearInternalVoxelsKernel<<<num_updated_blocks, dim_threads_per_voxel, 0,
                              cuda_stream_>>>(block_pointers_device_.size(),
                                              block_pointers_device_.data(),
                                              max_squared_distance_vox);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

void EsdfIntegrator::computeEsdfOnGPU(
    const std::vector<Index3D>& blocks_with_sites, EsdfLayer* esdf_layer) {
  CHECK_NOTNULL(esdf_layer);
  // Cache everything.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  block_pointers_host_.resize(blocks_with_sites.size());
  for (size_t i = 0; i < blocks_with_sites.size(); i++) {
    block_pointers_host_[i] =
        esdf_layer->getBlockAtIndex(blocks_with_sites[i]).get();
  }

  // First we go over all of the blocks with sites.
  // We compute all the proximal sites inside the block first.
  block_pointers_device_ = block_pointers_host_;
  sweepBlockBandOnGPU(block_pointers_device_, max_squared_distance_vox);

  // Get the neighbors of all the blocks with sites.
  std::vector<Index3D> blocks_to_run = blocks_with_sites;
  std::vector<Index3D> updated_blocks;

  int i = 0;
  while (!blocks_to_run.empty()) {
    updateLocalNeighborBandsOnGPU(blocks_to_run, block_pointers_device_,
                                  max_squared_distance_vox, esdf_layer,
                                  &updated_blocks, &neighbor_pointers_device_);
    VLOG(3) << "Iteration: " << i
            << " Number of updated blocks: " << updated_blocks.size()
            << " blocks with sites: " << blocks_with_sites.size();
    i++;
    sweepBlockBandOnGPU(neighbor_pointers_device_, max_squared_distance_vox);
    blocks_to_run = std::move(updated_blocks);
    block_pointers_device_ = neighbor_pointers_device_;
  }
}

void EsdfIntegrator::sweepBlockBandOnGPU(
    device_vector<EsdfBlock*>& block_pointers, float max_squared_distance_vox) {
  if (block_pointers.empty()) {
    return;
  }
  timing::Timer sweep_timer("esdf/integrate/compute/sweep");

  // Caching.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const int num_blocks = block_pointers.size();

  // Call the kernel.
  // We do 2-dimensional sweeps in this kernel. Each thread does 3 sweeps.
  // We do 8 blocks at a time.
  constexpr int kNumBlocksPerCudaBlock = 8;
  int dim_block = std::max(
      static_cast<int>(
          std::ceil(num_blocks / static_cast<float>(kNumBlocksPerCudaBlock))),
      1);
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kNumBlocksPerCudaBlock);
  sweepBlockBandKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
      num_blocks, block_pointers.data(), max_squared_distance_vox);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

void EsdfIntegrator::updateLocalNeighborBandsOnGPU(
    const std::vector<Index3D>& block_indices,
    device_vector<EsdfBlock*>& block_pointers, float max_squared_distance_vox,
    EsdfLayer* esdf_layer, std::vector<Index3D>* updated_blocks,
    device_vector<EsdfBlock*>* updated_block_pointers) {
  if (block_indices.empty()) {
    return;
  }

  timing::Timer neighbors_timer("esdf/integrate/compute/neighbors");

  CHECK_EQ(block_indices.size(), block_pointers.size());

  constexpr int kNumNeighbors = 6;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const int num_blocks = block_pointers.size();

  timing::Timer table_timer("esdf/integrate/compute/neighbors/table");

  // This one is quite a bit more complicated.
  // For each block, we need to get its 6 neighbors.
  std::vector<Index3D> neighbor_indices;
  neighbor_table_host_.resize(num_blocks * kNumNeighbors);
  neighbor_table_host_.setZero();
  neighbor_pointers_host_.resize(0);

  createNeighborTable(block_indices, esdf_layer, &neighbor_indices,
                      &neighbor_pointers_host_, &neighbor_table_host_);

  table_timer.Stop();

  // Set up an updated map.
  updated_blocks_device_.resize(neighbor_indices.size());
  updated_blocks_device_.setZero();

  neighbor_pointers_device_ = neighbor_pointers_host_;
  neighbor_table_device_ = neighbor_table_host_;

  timing::Timer kernel_timer("esdf/integrate/compute/neighbors/kernel");

  // Ok now we have to give all this stuff to the kernel.
  // TODO(helen): you get weird-ass concurrency issues if this is not 1.
  constexpr int kNumBlocksPerCudaBlock = 8;
  int dim_block = std::max(
      static_cast<int>(
          std::ceil(num_blocks / static_cast<float>(kNumBlocksPerCudaBlock))),
      1);
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kNumBlocksPerCudaBlock);
  // We have to do the neighbors one at a time basically for concurrency
  // issues.
  for (int i = 0; i < kNumNeighbors; i++) {
    updateLocalNeighborBandsKernel<<<dim_block, dim_threads, 0, cuda_stream_>>>(
        num_blocks, i, block_pointers.data(), neighbor_table_device_.data(),
        neighbor_pointers_device_.data(), max_squared_distance_vox,
        updated_blocks_device_.data());
  }
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  kernel_timer.Stop();

  // Unpack the kernel results.
  // TODO(helen): swap this to a kernel operation.
  updated_blocks->clear();
  updated_blocks_host_ = updated_blocks_device_;
  block_pointers_host_.resize(0);

  for (size_t i = 0; i < neighbor_indices.size(); i++) {
    if (updated_blocks_host_[i]) {
      updated_blocks->push_back(neighbor_indices[i]);
      block_pointers_host_.push_back(neighbor_pointers_host_[i]);
    }
  }
  *updated_block_pointers = block_pointers_host_;
}

void EsdfIntegrator::createNeighborTable(
    const std::vector<Index3D>& block_indices, EsdfLayer* esdf_layer,
    std::vector<Index3D>* neighbor_indices,
    host_vector<EsdfBlock*>* neighbor_pointers,
    host_vector<int>* neighbor_table) {
  // TODO(helen): make this extensible to different number of neighbors.
  constexpr int kNumNeighbors = 6;
  int num_blocks = block_indices.size();

  // Hash map mapping the neighbor index to the pointers above.
  Index3DHashMapType<int>::type neighbor_map;

  // Direction Shorthand: axis = neighbor_index/2
  // direction = neighbor_index%2 ? -1 : 1
  Index3D direction = Index3D::Zero();
  for (int block_number = 0; block_number < num_blocks; block_number++) {
    const Index3D& block_index = block_indices[block_number];
    for (int neighbor_number = 0; neighbor_number < kNumNeighbors;
         neighbor_number++) {
      direction.setZero();
      // Change just one axis of the direction.
      direction(neighbor_number / 2) = neighbor_number % 2 ? -1 : 1;
      // Check if this is already in our hash.
      Index3D neighbor_index = block_index + direction;
      auto res = neighbor_map.find(neighbor_index);
      if (res != neighbor_map.end()) {
        (*neighbor_table)[block_number * kNumNeighbors + neighbor_number] =
            res->second;
      } else {
        // Doesn't exist in the neighbor list yet.
        EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(neighbor_index);
        if (esdf_block) {
          int next_index = neighbor_indices->size();
          neighbor_indices->push_back(neighbor_index);
          neighbor_pointers->push_back(esdf_block.get());
          neighbor_map[neighbor_index] = next_index;
          (*neighbor_table)[block_number * kNumNeighbors + neighbor_number] =
              next_index;
        } else {
          (*neighbor_table)[block_number * kNumNeighbors + neighbor_number] =
              -1;
          neighbor_map[neighbor_index] = -1;
        }
      }
    }
  }
  CHECK_EQ(neighbor_table->size(), kNumNeighbors * block_indices.size());
  CHECK_EQ(neighbor_indices->size(), neighbor_pointers->size());
}

}  // namespace nvblox