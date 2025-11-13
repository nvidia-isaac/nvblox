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
#pragma once
#include "nvblox/core/internal/error_check.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {

// Kernel for copying gpu hash.
// Number of threads needed: block_hash_in.max_size()
template <typename BlockType>
__global__ void insertAllKernel(
    Index3DDeviceHashMapType<BlockType> block_hash_in,
    Index3DDeviceHashMapType<BlockType> block_hash_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // We are here iterating over the internal data array of stdgpu. Since all of
  // these elements are not necessarily a part of the hashmap, we might
  // encounter garbage and/or duplicated indices (e.g. ones remaining from
  // earler removal operations). To determine if elements are valid, we cannot
  // use the regular contains()/find()/count() functions, since they will return
  // true also for duplicated indices that are not part of the hash map.  To
  // this end, we use the occupied() function that directly looks up the
  // corresponding entry in the _occupied bitmask. The occupied() function is
  // currently exposed by means of a patch to stdgpu.
  if (idx < block_hash_in.max_size() && block_hash_in.occupied(idx)) {
    const auto pair = *(block_hash_in.begin() + idx);
    auto result = block_hash_out.insert(pair);
    NVBLOX_CHECK(result.second, "GPU hash insertion error")
  }
}

// Kernel for zeroing the gpu hash
// Number of threads needed: block_hash.max_size()
template <typename BlockType>
__global__ void setToZero(Index3DDeviceHashMapType<BlockType> block_hash) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < block_hash.max_size()) {
    memset(block_hash.begin() + idx, 0,
           sizeof(thrust::pair<Index3D, BlockType*>));
  }
}

template <typename BlockType>
void GPUHashImpl<BlockType>::initializeFromAsync(
    const GPUHashImpl<BlockType>& other, const CudaStream& cuda_stream) {
  // Hash must be empty when we're initializing from another hash
  CHECK_EQ(impl_.size(), 0);

  // Check that there's enough room to insert everything from other
  CHECK_GE(max_num_blocks_, other.impl_.size());

  constexpr int kNumThreadsPerBlock = 512;
  const int num_blocks =
      divideRoundUp(other.impl_.max_size(), kNumThreadsPerBlock);
  insertAllKernel<<<num_blocks, kNumThreadsPerBlock, 0, cuda_stream>>>(
      other.impl_, impl_);
  cuda_stream.synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Check that the sizes are equal after insertion
  CHECK_EQ(impl_.size(), other.impl_.size());
}

template <typename BlockType>
GPUHashImpl<BlockType>::GPUHashImpl(int max_num_blocks,
                                    const CudaStream& cuda_stream)
    : impl_(Index3DDeviceHashMapType<BlockType>::createDeviceObject(
          max_num_blocks)) {
  max_num_blocks_ = impl_.bucket_count();
  LOG(INFO) << "Creating a GPUHashImpl with requested capacity of "
            << max_num_blocks << " blocks. Real capacity: " << max_num_blocks_;

  // Initialize memory to zero to make compute-sanitizer's initcheck happy.
  constexpr int kNumThreadsPerBlock = 512;
  const int num_cuda_blocks =
      divideRoundUp(impl_.max_size(), kNumThreadsPerBlock);
  setToZero<<<num_cuda_blocks, kNumThreadsPerBlock, 0, cuda_stream>>>(impl_);
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename BlockType>
GPUHashImpl<BlockType>::~GPUHashImpl() {
  if (impl_.size() > 0) {
    Index3DDeviceHashMapType<BlockType>::destroyDeviceObject(impl_);
    VLOG(3) << "Destroying a GPUHashImpl";
  }
}

}  // namespace nvblox
