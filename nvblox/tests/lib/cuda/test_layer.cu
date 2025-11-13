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

#include <cuda_runtime.h>

#include "nvblox/core/hash.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/map/common_names.h"
#include "nvblox/tests/gpu_layer_utils.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {
namespace test_utils {

//  Set is_equal_out to false if the index-ptr pairs defined by block_indices
//  and block_ptrs cannot be found in the gpu hash
__global__ void hashesEqualKernel(
    const Index3DDeviceHashMapType<TsdfBlock> gpu_hash,
    const int num_block_indices, Index3D* block_indices, TsdfBlock** block_ptrs,
    int* num_inequal) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < num_block_indices) {
    const Index3D idx = block_indices[linear_index];

    auto itr = gpu_hash.find(idx);

    if (itr == gpu_hash.end() || (itr->second != block_ptrs[linear_index])) {
      printf("%i:     %p    %p\n", linear_index, itr->second,
             block_ptrs[linear_index]);
      atomicAdd(num_inequal, 1);
    }
  }
}

void checkGpuAndCpuHashesEqual(TsdfLayer& layer) {
  GPULayerView<TsdfBlock>& gpu_hash = layer.getGpuLayerView(CudaStreamOwning());

  unified_vector<Index3D> block_indices_device(MemoryType::kDevice);
  block_indices_device.copyFromAsync(layer.getAllBlockIndices(),
                                     CudaStreamOwning());

  unified_vector<TsdfBlock*> block_ptrs_device(MemoryType::kDevice);
  block_ptrs_device.copyFromAsync(layer.getAllBlockPointers(),
                                  CudaStreamOwning());

  constexpr int kNumThreadsPerBlock = 32;
  const int num_blocks = divideRoundUp(gpu_hash.size(), kNumThreadsPerBlock);

  CHECK_GE(gpu_hash.size(), block_indices_device.size());
  CHECK_GE(gpu_hash.size(), block_ptrs_device.size());

  if (num_blocks == 0) {
    return;
  }

  auto num_inequal = make_unified<int>(MemoryType::kHost, 0);

  hashesEqualKernel<<<num_blocks, kNumThreadsPerBlock>>>(
      gpu_hash.getHash().impl_, block_indices_device.size(),
      block_indices_device.data(), block_ptrs_device.data(), num_inequal.get());

  CHECK_EQ(*num_inequal, 0);
}
}  // namespace test_utils
}  // namespace nvblox
