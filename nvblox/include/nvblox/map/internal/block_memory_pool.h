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
#pragma once

#include <stack>
#include <vector>
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/internal/block_memory_pool_params.h"

namespace nvblox {

/// Storage class for pre-allcoated blocks
///
/// Maintains a large number of blocks (unified pointers) that are pre-allocated
/// up front. This reduces the need for exepensive calls to cudaMmalloc and
/// cudaFree during runtime.
///
/// Whenever the client needs to allocate a new block, popBlock() should be
/// used which returns (and transfers ownership) of a pre-alloacated block.
///
/// Whenever the client needs to free a block, pushblock() should be used
/// which returns the block to the pool and makes it ready for re-use.
template <class BlockType>
class BlockMemoryPool {
 public:
  /// Constructor that allocates blocks
  /// @param params Parameters governing the behavior of the block memory pool.
  BlockMemoryPool(const BlockMemoryPoolParams params = BlockMemoryPoolParams());

  /// Obtain a block from the pool. Should be used instead of allocating a
  /// new block. The pool is expanded if there are no more blocks remaining.
  /// @param cuda_stream Used when allocating memory in case the buffer
  /// needs

  typename BlockType::Ptr popBlock(const CudaStream& cuda_stream);

  /// Return a block to the pool. Should be used instead of de-allocating
  /// the block.
  /// @param block  Block to push
  void pushBlock(typename BlockType::Ptr block);

  /// Return the number of bytes occupied by all blocks in the pool
  size_t numAllocatedBytes() const {
    return num_allocated_blocks_ * sizeof(BlockType);
  }

  /// Return the number of blocks allocated
  size_t numAllocatedBlocks() const { return num_allocated_blocks_; }

  /// Return the parameters
  const BlockMemoryPoolParams& params() const { return params_; }

 private:
  /// Expand the memory pool and synchronize the stream
  void expand(const size_t num_blocks_to_allocate,
              const CudaStream& cuda_stream);

  /// Container for storing the memory in the pool
  std::stack<typename BlockType::Ptr> blocks_;

  /// Container for storing blocks that should be re-initialized
  host_vector<BlockType*> recycled_blocks_;

  /// Current size of the pool
  int num_allocated_blocks_ = 0;

  /// Parameters
  BlockMemoryPoolParams params_;
};

}  // namespace nvblox

#include "nvblox/map/internal/impl/block_memory_pool_impl.h"
