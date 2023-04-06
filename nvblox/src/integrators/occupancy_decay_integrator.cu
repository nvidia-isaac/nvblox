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
#include <nvblox/integrators/occupancy_decay_integrator.h>

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

__device__ float inline voxelIsDecayed(float log_odds,
                                       float free_space_decay_log_odds,
                                       float occupied_space_decay_log_odds) {
  // Check if the next decay step would change the sign of the log odds value
  // (i.e. pass 0.5 probability).
  if (log_odds >= 0) {
    return log_odds + occupied_space_decay_log_odds < 0;
  } else {
    return log_odds + free_space_decay_log_odds >= 0;
  }
}

__global__ void decayProbabilityKernel(OccupancyBlock** block_ptrs,
                                       float free_space_decay_log_odds,
                                       float occupied_space_decay_log_odds,
                                       bool* is_block_fully_decayed) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_decayed[blockIdx.x] = true;
  }
  __syncthreads();

  OccupancyVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  if (voxelIsDecayed(voxel_ptr->log_odds, free_space_decay_log_odds,
                     occupied_space_decay_log_odds)) {
    // This voxel decayed to zero log odds (0.5 occupancy probability).
    voxel_ptr->log_odds = 0.0;
    return;
  } else {
    // If one voxel in a block is not decayed, the block is not fully decayed.
    // NOTE: There could be more than one thread writing this value, but because
    // all of them write false it is no issue.
    is_block_fully_decayed[blockIdx.x] = false;
  }

  if (voxel_ptr->log_odds >= 0) {
    voxel_ptr->log_odds += occupied_space_decay_log_odds;
  } else {
    voxel_ptr->log_odds += free_space_decay_log_odds;
  }
}

OccupancyDecayIntegrator::OccupancyDecayIntegrator() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}
OccupancyDecayIntegrator::~OccupancyDecayIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void OccupancyDecayIntegrator::decay(OccupancyLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  if (layer_ptr->numAllocatedBlocks() == 0) {
    // Empty layer, nothing to do here.
    return;
  }
  decayProbability(layer_ptr);
  if (deallocate_decayed_blocks_) {
    deallocateFullyDecayedBlocks(layer_ptr);
  }
}

void OccupancyDecayIntegrator::decayProbability(OccupancyLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Expand the buffers when needed
  if (num_allocated_blocks > allocated_block_ptrs_host_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_allocated_blocks);
    allocated_block_ptrs_host_.reserve(new_size);
    allocated_block_ptrs_device_.reserve(new_size);
    block_fully_decayed_device_.reserve(new_size);
    block_fully_decayed_host_.reserve(new_size);
  }

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_ = layer_ptr->getAllBlockPointers();
  allocated_block_ptrs_device_ = allocated_block_ptrs_host_;

  // Kernel call - One ThreadBlock launched per VoxelBlock
  block_fully_decayed_device_.resize(num_allocated_blocks);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_allocated_blocks;
  decayProbabilityKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                           integration_stream_>>>(
      allocated_block_ptrs_device_.data(),  // NOLINT
      free_space_decay_log_odds_,           // NOLINT
      occupied_space_decay_log_odds_,       // NOLINT
      block_fully_decayed_device_.data());  // NOLINT
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back to host
  block_fully_decayed_host_ = block_fully_decayed_device_;

  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == num_allocated_blocks);
  CHECK(allocated_block_ptrs_device_.size() == num_allocated_blocks);
  CHECK(block_fully_decayed_device_.size() == num_allocated_blocks);
  CHECK(block_fully_decayed_host_.size() == num_allocated_blocks);
}

void OccupancyDecayIntegrator::deallocateFullyDecayedBlocks(
    OccupancyLayer* layer_ptr) {
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Get the block indices on host
  std::vector<Index3D> allocated_block_indices_host =
      layer_ptr->getAllBlockIndices();

  // Find blocks that are fully decayed
  CHECK(num_allocated_blocks == allocated_block_indices_host.size());
  CHECK(num_allocated_blocks == block_fully_decayed_host_.size());
  for (size_t i = 0; i < num_allocated_blocks; i++) {
    if (block_fully_decayed_host_[i]) {
      layer_ptr->clearBlock(allocated_block_indices_host[i]);
    }
  }
}

bool OccupancyDecayIntegrator::deallocate_decayed_blocks() {
  return deallocate_decayed_blocks_;
}
void OccupancyDecayIntegrator::deallocate_decayed_blocks(
    bool deallocate_decayed_blocks) {
  deallocate_decayed_blocks_ = deallocate_decayed_blocks;
}

float OccupancyDecayIntegrator::free_region_decay_probability() {
  return probabilityFromLogOdds(free_space_decay_log_odds_);
}
void OccupancyDecayIntegrator::free_region_decay_probability(float value) {
  CHECK(value > 0.5f && value <= 1.f)
      << "The free_region_decay_probability must be in [0.5, "
         "1.0] for the free region to decay towards 0.5 occupancy probability.";
  free_space_decay_log_odds_ = logOddsFromProbability(value);
}

float OccupancyDecayIntegrator::occupied_region_decay_probability() {
  return probabilityFromLogOdds(occupied_space_decay_log_odds_);
}
void OccupancyDecayIntegrator::occupied_region_decay_probability(float value) {
  CHECK(value >= 0.f && value < 0.5f)
      << "The occupied_region_decay_probability must be in [0.0, "
         "0.5] for the occupied region to decay towards 0.5 occupancy "
         "probability.";
  occupied_space_decay_log_odds_ = logOddsFromProbability(value);
}

}  // namespace nvblox
