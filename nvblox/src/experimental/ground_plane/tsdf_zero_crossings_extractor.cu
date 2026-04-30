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
limitationj under the License.
*/
#include <assert.h>
#include <algorithm>
#include <cmath>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/tsdf_zero_crossings_extractor.h"
#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

__global__ void computeZeroCrossingsFromAboveKernel(
    const TsdfBlock** tsdf_blocks, const TsdfBlock** tsdf_blocks_above,
    float min_tsdf_weight, const float voxel_size,
    const Index3D* block_indices_device, int max_crossings,
    Vector3f* p_L_crossings, int* p_L_crossings_count) {
  // Get the voxels for this thread-block, which could go over a VoxelBlock
  // boundary.
  const TsdfBlock* tsdf_block = tsdf_blocks[blockIdx.x];
  const TsdfBlock* tsdf_block_above = tsdf_blocks_above[blockIdx.x];

  const TsdfVoxel* voxel_above = nullptr;
  const TsdfVoxel* voxel_below = nullptr;
  if (threadIdx.z < (TsdfBlock::kVoxelsPerSide - 1)) {
    // We are within one block for both the above and below voxel.
    voxel_above =
        &tsdf_block->voxels[threadIdx.x][threadIdx.y][threadIdx.z + 1];
    voxel_below = &tsdf_block->voxels[threadIdx.x][threadIdx.y][threadIdx.z];
  }
  if (threadIdx.z == (TsdfBlock::kVoxelsPerSide - 1)) {
    // We need to check the boundary to the VoxelBlock above.
    if (tsdf_block_above == nullptr) {
      // If the block above is invalid/ No Block was initialized above the
      // current one, skip.
      return;
    }
    voxel_above = &tsdf_block_above->voxels[threadIdx.x][threadIdx.y][0];
    voxel_below = &tsdf_block->voxels[threadIdx.x][threadIdx.y][threadIdx.z];
  }

  const bool is_valid_weight_above = voxel_above->weight >= min_tsdf_weight;
  const bool is_valid_weight_below = voxel_below->weight >= min_tsdf_weight;
  if (is_valid_weight_above && is_valid_weight_below) {
    // Check for positive to negative zero-crossing.
    if (is_valid_weight_above && is_valid_weight_below) {
      // Check for positive to negative zero-crossing.
      if (voxel_above->distance > 0.0f && voxel_below->distance <= 0.0f) {
        const float block_size = voxelSizeToBlockSize(voxel_size);
        const Index3D block_idx = block_indices_device[blockIdx.x];
        const Index3D voxel_index_below(threadIdx.x, threadIdx.y, threadIdx.z);
        const Vector3f p_L_below = getCenterPositionFromBlockIndexAndVoxelIndex(
            block_size, block_idx, voxel_index_below);

        const float distance_at_vox_above = voxel_above->distance;
        const float distance_at_vox_below = voxel_below->distance;
        const float distance_m =
            (-distance_at_vox_below * voxel_size) /
            (distance_at_vox_above - distance_at_vox_below);
        const auto p_L_crossing =
            Vector3f(p_L_below.x(), p_L_below.y(), p_L_below.z() + distance_m);

        // Atomically get the next available index in the global crossing array.
        const int idx = atomicAdd(p_L_crossings_count, 1);
        // Return as we would access invalid memory.
        if (idx >= max_crossings) {
          return;
        }
        p_L_crossings[idx] = p_L_crossing;
      }
    }
  }
}

TsdfZeroCrossingsExtractor::TsdfZeroCrossingsExtractor(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

std::optional<std::vector<Vector3f>>
TsdfZeroCrossingsExtractor::computeZeroCrossingsFromAboveOnGPU(
    const TsdfLayer& tsdf_layer) {
  timing::Timer timer("ground_plane/compute_zero_crossings_from_above_on_gpu");

  // Get all block indices
  std::vector<Index3D> block_indices_host = tsdf_layer.getAllBlockIndices();
  std::vector<const TsdfBlock*> block_ptrs_host =
      tsdf_layer.getAllBlockPointers();

  const int num_blocks = block_ptrs_host.size();
  if (num_blocks == 0) {
    return std::nullopt;
  }

  resetAndAllocateCrossingBuffers();

  block_ptrs_device_.copyFromAsync(block_ptrs_host, *cuda_stream_);
  block_indices_device_.copyFromAsync(block_indices_host, *cuda_stream_);

  constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
  const dim3 threads_per_block(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);

  // Collect the respective blocks above
  std::vector<const TsdfBlock*> block_above_ptrs_host;
  for (const Index3D& idx : block_indices_host) {
    const auto block_above_ptr_host =
        tsdf_layer.getBlockAtIndex(idx + Index3D(0, 0, 1));
    if (block_above_ptr_host) {
      block_above_ptrs_host.push_back(block_above_ptr_host.get());
    } else {
      block_above_ptrs_host.push_back(nullptr);
    }
  }
  block_ptrs_above_device_.copyFromAsync(block_above_ptrs_host, *cuda_stream_);

  computeZeroCrossingsFromAboveKernel<<<num_blocks, threads_per_block, 0,
                                        *cuda_stream_>>>(
      block_ptrs_device_.data(), block_ptrs_above_device_.data(),
      min_tsdf_weight_, tsdf_layer.voxel_size(), block_indices_device_.data(),
      max_crossings_, p_L_crossings_global_device_.data(),
      p_L_crossings_count_device_.get());
  p_L_crossings_count_device_.copyToAsync(p_L_crossings_count_host_,
                                          *cuda_stream_);

  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Check if we hit the limit, resize if needed
  if (*p_L_crossings_count_host_ >= max_crossings_) {
    const int old_max_crossings = max_crossings_;
    max_crossings_ = static_cast<int>(
        std::ceil(buffer_expansion_factor_ *
                  std::max(static_cast<int>(*p_L_crossings_count_host_),
                           max_crossings_)));

    LOG(INFO) << "Zero crossing buffer limit reached (" << old_max_crossings
              << "). Resizing buffer from " << old_max_crossings << " to "
              << max_crossings_;

    // Reallocate buffers with new size
    resetAndAllocateCrossingBuffers();

    // Retry computation with larger buffer
    computeZeroCrossingsFromAboveKernel<<<num_blocks, threads_per_block, 0,
                                          *cuda_stream_>>>(
        block_ptrs_device_.data(), block_ptrs_above_device_.data(),
        min_tsdf_weight_, tsdf_layer.voxel_size(), block_indices_device_.data(),
        max_crossings_, p_L_crossings_global_device_.data(),
        p_L_crossings_count_device_.get());
    p_L_crossings_count_device_.copyToAsync(p_L_crossings_count_host_,
                                            *cuda_stream_);

    cuda_stream_->synchronize();
    checkCudaErrors(cudaPeekAtLastError());

    // Check if we still hit the limit after resize
    if (*p_L_crossings_count_host_ >= max_crossings_) {
      LOG(WARNING) << "Maximum number of crossings reached (" << max_crossings_
                   << ") even after buffer resize.";
      return std::nullopt;
    }
  }

  // Success - resize buffer to actual count and return
  p_L_crossings_global_device_.resizeAsync(*p_L_crossings_count_host_,
                                           *cuda_stream_);
  cuda_stream_->synchronize();
  timer.Stop();
  return p_L_crossings_global_device_.toVectorAsync(*cuda_stream_);
}

void TsdfZeroCrossingsExtractor::resetAndAllocateCrossingBuffers() {
  if (p_L_crossings_count_device_ == nullptr ||
      p_L_crossings_count_host_ == nullptr) {
    p_L_crossings_count_device_ = make_unified<int>(MemoryType::kDevice);
    p_L_crossings_count_host_ = make_unified<int>(MemoryType::kHost);
  }
  p_L_crossings_count_device_.setZeroAsync(*cuda_stream_);

  if (static_cast<size_t>(max_crossings_) >=
      p_L_crossings_global_device_.size()) {
    p_L_crossings_global_device_.resizeAsync(max_crossings_, *cuda_stream_);
  }
}

void TsdfZeroCrossingsExtractor::max_crossings(int max_crossings) {
  max_crossings_ = max_crossings;
}

void TsdfZeroCrossingsExtractor::min_tsdf_weight(float min_tsdf_weight) {
  min_tsdf_weight_ = min_tsdf_weight;
}

parameters::ParameterTreeNode TsdfZeroCrossingsExtractor::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "tsdf_zero_crossings_extractor" : name_remap;
  return parameters::ParameterTreeNode(
      name,
      {
          parameters::ParameterTreeNode("min_tsdf_weight:", min_tsdf_weight_),
          parameters::ParameterTreeNode("max_crossings:", max_crossings_),
      });
}

}  // namespace nvblox
