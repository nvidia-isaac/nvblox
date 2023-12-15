/*
Copyright 2023 NVIDIA CORPORATION

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
#include "nvblox/integrators/freespace_integrator.h"

#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

static_assert(TsdfBlock::kVoxelsPerSide == FreespaceBlock::kVoxelsPerSide,
              "Need same block dimensions for tsdf and freespace blocks");

// clamp an index in-place to the inclusive range defined by minval and maxval
__device__ void clamp(Index3D& index, int minval, int maxval) {
  for (int i = 0; i < 3; ++i) {
    index(i) = std::max(std::min(index(i), maxval), minval);
  }
}

// Number of padded voxels appended to each side of a block in order to
// allow for filtering. Currently only 1 is supported.
constexpr int kPaddingSize = 1;

// Class for storing a neighborhood of 3x3x3 block pointers. This allows for
// faster lookup of voxels, compared to using a hashmap.
//
// The container is intended to be stored in shared memory and is populated
// collaboratively among the threads in one block
//
// Note that all lookups functions use global block indices, there's thus no
// need to transform the indices at the call site.
template <typename VoxelType>
class BlockNeighborhood {
  using BlockType = VoxelBlock<VoxelType>;

 public:
  // Constants defining the number of block neighbors. These cannot be changed.
  static constexpr int kNumBlocks1D = 3;
  static constexpr int kNumBlocks3D =
      kNumBlocks1D * kNumBlocks1D * kNumBlocks1D;

  // Lookup a block pointer and populate a block in the 3x3x3 neighborhood.
  // Which block to populate is determined by the current threadIdx.
  //
  // @param center_block_index  Index of the center block in the neighborhood
  // @param block_hash          Hash map used to lookup the block pointers
  __device__ void populateBlock(
      const Index3D& center_block_index,
      const Index3DDeviceHashMapType<VoxelBlock<VoxelType>>& block_hash) {
    // This will be set by all threads, but that's alright since they all write
    // the same value.
    topleft_block_index_ = {center_block_index.x() - 1,
                            center_block_index.y() - 1,
                            center_block_index.z() - 1};

    // Let the first few threads populate the block pointers.
    if (threadIdx.x < kNumBlocks1D && threadIdx.y < kNumBlocks1D &&
        threadIdx.z < kNumBlocks1D) {
      const Index3D block_index_to_populate = {
          topleft_block_index_.x() + threadIdx.x,
          topleft_block_index_.y() + threadIdx.y,
          topleft_block_index_.z() + threadIdx.z};

      setBlock(block_index_to_populate,
               getBlockPtr(block_hash, block_index_to_populate));
    }
  }
  // Getters for a block given a global block_index that is part of the
  // neighborhood
  __device__ const BlockType* getBlock(const Index3D& block_index) const {
    return block_ptrs_[getLinearIndex(block_index)];
  }
  __device__ BlockType* getBlock(const Index3D& block_index) {
    return block_ptrs_[getLinearIndex(block_index)];
  }

  // Getters for a voxel given a global block_index that is part of the
  // neighborhood
  __device__ const VoxelType* getVoxel(const Index3D& block_index,
                                       const Index3D& voxel_index) const {
    const BlockType* block_ptr = getBlock(block_index);
    if (block_ptr == nullptr) {
      return nullptr;
    } else {
      return &block_ptr
                  ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
    }
  }

  __device__ void setVoxel(const Index3D& block_index,
                           const Index3D& voxel_index, const VoxelType& voxel) {
    BlockType* block_ptr = getBlock(block_index);
    if (block_ptr != nullptr) {
      block_ptr->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()] =
          voxel;
    }
  }

 private:
  // Setter for block pointer
  __device__ void setBlock(const Index3D& block_index, BlockType* block_ptr) {
    block_ptrs_[getLinearIndex(block_index)] = block_ptr;
  }

  // Convert a global block index into an 1D index in block_ptrs_;
  __device__ int getLinearIndex(const Index3D& block_index) const {
    const Index3D local_index = block_index - topleft_block_index_;

    const int linear_index = local_index.x() + kNumBlocks1D * local_index.y() +
                             kNumBlocks1D * kNumBlocks1D * local_index.z();
    assert(linear_index >= 0 && linear_index < kNumBlocks3D);
    return linear_index;
  }

  Index3D topleft_block_index_;
  BlockType* block_ptrs_[kNumBlocks3D];
};

// Stores a block of voxels with one extra voxel of padding on each side, e.g.
// if the block size is 8x8x8, this container has capacity for 10x10x10 voxels.
// The padded voxels will be populated from neighboring blocks. Intended to be
// used to simplify border handling when filtering and is suitable to be stored
// in shared memory.
//
// Note that access functions use non-padded voxel indices, i.e. there is no
// need to compensate for the padding at the call site since this is done
// internally. For example PaddedBlock::at({0,0,0}) will refer to the top-left
// corner in both the padded and non-padded case. Indices that exceeds or
// preceeds the block dimensions will return a voxel in the neighboring block,
// for example {0, -1, 0} will return a voxel from the neighbor in negative-Y
// direction. and {0, 10, 0} will refer to a block in the positive-Y direction.
template <typename VoxelType>
class PaddedBlock {
  using BlockType = VoxelBlock<VoxelType>;

 public:
  // Size definitions. Cannot be changed
  static constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
  static constexpr int kVoxelsPerSidePadded = kVoxelsPerSide + 2 * kPaddingSize;

  // Populate the voxel at voxel_index by looking it up in block_neighbors
  __device__ void populateVoxel(
      const Index3D& voxel_index, const Index3D& center_block_index,
      const BlockNeighborhood<VoxelType>& block_neighbors) {
    // We will write to this padded block index
    const Index3D target_voxel_index =
        Index3D(voxel_index.x() + kPaddingSize, voxel_index.y() + kPaddingSize,
                voxel_index.z() + kPaddingSize);

    // Determine which block/voxel we will read from.
    Index3D source_block_index = center_block_index;
    Index3D source_voxel_index = voxel_index;

    // If we're on the border we need to read the voxel from a neigboring
    // block. Adjust block_index and voxel_index accordingly
    for (int axis = 0; axis < 3; ++axis) {
      // We're at the low boundary. Decrement block index to get previous block
      // along current axis.
      if (source_voxel_index(axis) == -1) {
        --source_block_index(axis);
        source_voxel_index(axis) =
            kVoxelsPerSide - 1;  // Voxel index will wrap around to the end
      } else if (source_voxel_index(axis) == (kVoxelsPerSide)) {
        // We're at the high boundary. Increment block to get next block along
        // current axis
        ++source_block_index(axis);
        source_voxel_index(axis) = 0;  // Voxel index will wrap around to zero
      }
    }

    // Set the voxel if it exists
    const VoxelType* source_voxel_ptr =
        block_neighbors.getVoxel(source_block_index, source_voxel_index);
    if (source_voxel_ptr == nullptr) {
      // if the voxel doesn't exist  ( because the block is outside the map) we
      // duplicate an adjacent voxel insted.
      assert(!isWithinBlockBounds(source_voxel_index));
      clamp(source_voxel_index, 0, kVoxelsPerSide - 1);
      source_voxel_ptr =
          block_neighbors.getVoxel(center_block_index, source_voxel_index);
    }

    assert(source_voxel_ptr != nullptr);
    voxels_[target_voxel_index.x()][target_voxel_index.y()]
           [target_voxel_index.z()] = *source_voxel_ptr;
  }

  // Returns true if the voxel lies inside the block i.e. *not* in the padded
  // space
  __device__ bool isWithinBlockBounds(const Index3D& voxel_index) {
    return (voxel_index.x() >= 0 && voxel_index.x() < (kVoxelsPerSide) &&
            voxel_index.y() >= 0 && voxel_index.y() < (kVoxelsPerSide) &&
            voxel_index.z() >= 0 && voxel_index.z() < (kVoxelsPerSide));
  }

  // Access functions.
  __device__ VoxelType& at(const Index3D& voxel_index) {
    return voxels_[voxel_index.x() + kPaddingSize]
                  [voxel_index.y() + kPaddingSize]
                  [voxel_index.z() + kPaddingSize];
  }
  __device__ const VoxelType& at(const Index3D& voxel_index) const {
    return voxels_[voxel_index.x() + kPaddingSize]
                  [voxel_index.y() + kPaddingSize]
                  [voxel_index.z() + kPaddingSize];
  }

 private:
  VoxelType voxels_[kVoxelsPerSidePadded][kVoxelsPerSidePadded]
                   [kVoxelsPerSidePadded];
};  // namespace nvblox

// Return true if the voxel is free according to Dynablox Eq. (10) (single
// voxel)
__device__ bool isVoxelFree(const FreespaceVoxel& freespace_voxel,
                            const TsdfVoxel& tsdf_voxel, Time current_time_ms,
                            Time min_duration_since_occupied_for_freespace_ms) {
  return tsdf_voxel.weight > 1e-6 &&
         freespace_voxel.last_occupied_timestamp_ms <=
             current_time_ms - min_duration_since_occupied_for_freespace_ms;
}

// Return true if all voxels in a neighborhood are free.
__device__ bool isVoxelNeighborhoodFree(
    const Index3D& voxel_index,
    const PaddedBlock<FreespaceVoxel>& freespace_block_padded,
    const PaddedBlock<TsdfVoxel>& tsdf_block_padded, Time current_time_ms,
    Time min_duration_since_occupied_for_freespace_ms) {
  bool neighborhood_is_free = true;

  for (int x = -kPaddingSize; x <= kPaddingSize; x++) {
    for (int y = -kPaddingSize; y <= kPaddingSize; y++) {
      for (int z = -kPaddingSize; z <= kPaddingSize; z++) {
        if (x == 0 && y == 0 && z == 0) {
          continue;  // Do not add the original voxel.
        }

        const Index3D neighbor_index = voxel_index + Index3D(x, y, z);
        const TsdfVoxel& tsdf_voxel = tsdf_block_padded.at(neighbor_index);
        const FreespaceVoxel& freespace_voxel =
            freespace_block_padded.at(neighbor_index);

        neighborhood_is_free &=
            isVoxelFree(freespace_voxel, tsdf_voxel, current_time_ms,
                        min_duration_since_occupied_for_freespace_ms);
      }
    }
  }
  return neighborhood_is_free;
}

__global__ void updateFreespaceLayerKernel(
    const Index3DDeviceHashMapType<TsdfBlock> tsdf_block_hash,
    const Index3D* block_indices_to_update, float voxel_size,
    float max_tsdf_distance_for_occupancy_m,
    Time max_unobserved_to_keep_consecutive_occupancy_ms,
    Time min_duration_since_occupied_for_freespace_ms,
    Time min_consecutive_occupancy_duration_for_reset_ms,
    bool check_neighborhood, Time last_update_time_ms,
    Time current_update_time_ms,
    Index3DDeviceHashMapType<FreespaceBlock> freespace_block_hash) {
  // This kernel implements the freespace update as described in the
  // dynablox paper (https://ieeexplore.ieee.org/document/10218983).
  //
  // It consist of the following steps:
  // - Initialization of freespace voxels if seen for the first time.
  // - Update the consecutive_occupancy_duration_ms field
  // - Update the last_occupied_timestamp_ms field
  // - Check if the voxel (and all its neighbors if check_neighborhood=true)
  //   is/are free
  // - Update the is_high_confidence_freespace field
  // Every ThreadBlock works on one VoxelBlock (blockIdx.y/z should be zero)
  const Index3D block_index = block_indices_to_update[blockIdx.x];

  // Since a thread block also includes padded voxels, we obtain the
  // actual voxel index by subtracting the border size from thread indices.
  const Index3D voxel_index =
      Index3D(threadIdx.z - kPaddingSize, threadIdx.y - kPaddingSize,
              threadIdx.x - kPaddingSize);

  // Lookup all block pointers in a 3x3x3 neighborhood around block_index and
  // store them in shared memory. This saves us from excessive and expensive
  // hashtable lookups.
  __shared__ BlockNeighborhood<FreespaceVoxel> freespace_block_neighbors;
  __shared__ BlockNeighborhood<TsdfVoxel> tsdf_block_neighbors;
  freespace_block_neighbors.populateBlock(block_index, freespace_block_hash);
  tsdf_block_neighbors.populateBlock(block_index, tsdf_block_hash);
  __syncthreads();

  // Populate shared memory with voxels from the current block. We also copy an
  // additional padded voxel layerborder around the block to allow filtering at
  // the border
  __shared__ PaddedBlock<FreespaceVoxel> freespace_block_padded;
  freespace_block_padded.populateVoxel(voxel_index, block_index,
                                       freespace_block_neighbors);

  __shared__ PaddedBlock<TsdfVoxel> tsdf_block_padded;
  tsdf_block_padded.populateVoxel(voxel_index, block_index,
                                  tsdf_block_neighbors);
  __syncthreads();

  // Get the freespace voxel
  FreespaceVoxel* freespace_voxel = &freespace_block_padded.at(voxel_index);

  // Initialization of freespace
  if (freespace_voxel->last_occupied_timestamp_ms == Time(0)) {
    // All voxels are initialized to being occupied
    freespace_voxel->last_occupied_timestamp_ms = current_update_time_ms;
    freespace_voxel->consecutive_occupancy_duration_ms = Time(0);
    freespace_voxel->is_high_confidence_freespace = false;
  } else {
    // Get the corresponding tsdf voxel
    TsdfVoxel* tsdf_voxel = &tsdf_block_padded.at(voxel_index);

    // Update consecutive occupancy duration
    // Note: We use the last_occupied_timestamp_ms from the last update here to
    // start counting the consecutive_occupancy_duration_ms from 0 ms when a
    // voxel was seen occupied. Dynablox Eq. (9)
    if (current_update_time_ms - freespace_voxel->last_occupied_timestamp_ms <=
        max_unobserved_to_keep_consecutive_occupancy_ms) {
      // Voxel was occupied lately
      freespace_voxel->consecutive_occupancy_duration_ms +=
          current_update_time_ms - last_update_time_ms;
    } else {
      // We haven't seen the voxel occupied for some time
      freespace_voxel->consecutive_occupancy_duration_ms = Time(0);
    }

    // Update the last occupied timestamp
    // Dynablox Eq. (8)
    if (tsdf_voxel->distance <= max_tsdf_distance_for_occupancy_m) {
      // We are close to a surface, let's assume the voxel is occupied
      freespace_voxel->last_occupied_timestamp_ms = current_update_time_ms;
    }

    // Check if the voxel is free
    bool is_free =
        isVoxelFree(*freespace_voxel, *tsdf_voxel, current_update_time_ms,
                    min_duration_since_occupied_for_freespace_ms);

    // Synchronize here because the last_occupied_timestamp_ms field of the
    // neighboring voxels could have been updated during this kernel. This is
    // strictly only necessary if check_neighborhood=true, but syncing inside an
    // if-statement is not recommended since it might lead to a deadlock if
    // threads are diverging.
    __syncthreads();

    // Check if neighbors are free as well
    // Dynablox Eq. (10) (neighborhood)
    if (check_neighborhood && is_free &&
        tsdf_block_padded.isWithinBlockBounds(voxel_index)) {
      is_free &= isVoxelNeighborhoodFree(
          voxel_index, freespace_block_padded, tsdf_block_padded,
          current_update_time_ms, min_duration_since_occupied_for_freespace_ms);
    }

    // Update high confidence freespace
    // Dynablox Eq. (12)
    if (freespace_voxel->consecutive_occupancy_duration_ms >=
        min_consecutive_occupancy_duration_for_reset_ms) {
      // There was consecutive occupancy for some time: reset freespace
      freespace_voxel->is_high_confidence_freespace = false;
    } else {
      // Otherwise high confidence freespace is set if the voxel is free
      // and kept if it was high confidence before
      // Dynablox Eq. (11)
      freespace_voxel->is_high_confidence_freespace =
          freespace_voxel->is_high_confidence_freespace || is_free;
    }
  }

  // Copy shared mem back to global
  if (tsdf_block_padded.isWithinBlockBounds(voxel_index)) {
    freespace_block_neighbors.setVoxel(block_index, voxel_index,
                                       *freespace_voxel);
  }
}

FreespaceIntegrator::FreespaceIntegrator()
    : FreespaceIntegrator(std::make_shared<CudaStreamOwning>()) {}

FreespaceIntegrator::FreespaceIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

float FreespaceIntegrator::max_tsdf_distance_for_occupancy_m() const {
  return max_tsdf_distance_for_occupancy_m_;
}

void FreespaceIntegrator::max_tsdf_distance_for_occupancy_m(float value) {
  max_tsdf_distance_for_occupancy_m_ = value;
}

Time FreespaceIntegrator::max_unobserved_to_keep_consecutive_occupancy_ms()
    const {
  return max_unobserved_to_keep_consecutive_occupancy_ms_;
}

void FreespaceIntegrator::max_unobserved_to_keep_consecutive_occupancy_ms(
    Time value) {
  max_unobserved_to_keep_consecutive_occupancy_ms_ = value;
}

Time FreespaceIntegrator::min_duration_since_occupied_for_freespace_ms() const {
  return min_duration_since_occupied_for_freespace_ms_;
}

void FreespaceIntegrator::min_duration_since_occupied_for_freespace_ms(
    Time value) {
  min_duration_since_occupied_for_freespace_ms_ = value;
}

Time FreespaceIntegrator::min_consecutive_occupancy_duration_for_reset_ms()
    const {
  return min_consecutive_occupancy_duration_for_reset_ms_;
}

void FreespaceIntegrator::min_consecutive_occupancy_duration_for_reset_ms(
    Time value) {
  min_consecutive_occupancy_duration_for_reset_ms_ = value;
}

bool FreespaceIntegrator::check_neighborhood() const {
  return check_neighborhood_;
}

void FreespaceIntegrator::check_neighborhood(bool value) {
  check_neighborhood_ = value;
}

parameters::ParameterTreeNode FreespaceIntegrator::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "freespace_integrator" : name_remap;
  std::function<std::string(const Time&)> time_to_string = [](const Time& t) {
    return std::to_string(static_cast<int64_t>(t));
  };
  using parameters::ParameterTreeNode;
  return ParameterTreeNode(
      name,
      {
          ParameterTreeNode("max_tsdf_distance_for_occupancy_m:",
                            max_tsdf_distance_for_occupancy_m_),
          ParameterTreeNode("max_unobserved_to_keep_consecutive_occupancy_ms:",
                            max_unobserved_to_keep_consecutive_occupancy_ms_,
                            time_to_string),
          ParameterTreeNode("min_duration_since_occupied_for_freespace_ms:",
                            min_duration_since_occupied_for_freespace_ms_,
                            time_to_string),
          ParameterTreeNode("min_consecutive_occupancy_duration_for_reset_ms:",
                            min_consecutive_occupancy_duration_for_reset_ms_,
                            time_to_string),
          ParameterTreeNode("check_neighborhood:", check_neighborhood_),
      });
}

void FreespaceIntegrator::updateFreespaceLayer(
    const std::vector<Index3D>& block_indices_to_update, Time update_time_ms,
    const TsdfLayer& tsdf_layer, FreespaceLayer* freespace_layer_ptr) {
  timing::Timer integration_timer("freespace/integrate");

  // Check inputs
  CHECK_NOTNULL(freespace_layer_ptr);
  CHECK(freespace_layer_ptr->voxel_size() - tsdf_layer.voxel_size() < 1e-4)
      << "Voxel size of tsdf and freespace layer must be equal.";
  if (block_indices_to_update.empty()) {
    return;
  }
  const size_t num_block_to_update = block_indices_to_update.size();
  current_update_time_ms_ = update_time_ms;

  // Allocate missing blocks
  timing::Timer allocate_timer("freespace/integrate/allocate");
  freespace_layer_ptr->allocateBlocksAtIndices(block_indices_to_update, *cuda_stream_);
  allocate_timer.Stop();

  timing::Timer update_timer("freespace/integrate/update_blocks");

  // Expand the buffers when needed
  if (num_block_to_update > block_indices_to_update_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_block_to_update);
    block_indices_to_update_device_.reserveAsync(new_size, *cuda_stream_);
  }

  transferBlocksIndicesToDevice(block_indices_to_update, *cuda_stream_,
                                &block_indices_to_update_host_,
                                &block_indices_to_update_device_);

  // Kernel configuration:
  // - One threadBlock per VoxelBlock
  // - NxNxN threads where N is the block side-length in voxels.
  constexpr int kNumThreads1D = TsdfBlock::kVoxelsPerSide + 2 * kPaddingSize;
  const dim3 kThreadsPerBlock(kNumThreads1D, kNumThreads1D, kNumThreads1D);
  const int num_thread_blocks = num_block_to_update;

  updateFreespaceLayerKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                               *cuda_stream_>>>(
      tsdf_layer.getGpuLayerView().getHash().impl_,           // NOLINT
      block_indices_to_update_device_.data(),                 // NOLINT
      freespace_layer_ptr->voxel_size(),                      // NOLINT
      max_tsdf_distance_for_occupancy_m_,                     // NOLINT
      max_unobserved_to_keep_consecutive_occupancy_ms_,       // NOLINT
      min_duration_since_occupied_for_freespace_ms_,          // NOLINT
      min_consecutive_occupancy_duration_for_reset_ms_,       // NOLINT
      check_neighborhood_,                                    // NOLINT
      last_update_time_ms_,                                   // NOLINT
      current_update_time_ms_,                                // NOLINT
      freespace_layer_ptr->getGpuLayerView().getHash().impl_  // NOLINT
  );
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  last_update_time_ms_ = update_time_ms;
}

}  // namespace nvblox
