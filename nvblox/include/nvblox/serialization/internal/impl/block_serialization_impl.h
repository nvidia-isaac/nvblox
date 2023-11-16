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

namespace nvblox {

template <typename VoxelType>
std::vector<Byte> serializeBlock(
    const unified_ptr<const VoxelBlock<VoxelType>>& block,
    const CudaStream cuda_stream) {
  size_t block_size = sizeof(block->voxels);

  std::vector<Byte> bytes;
  bytes.resize(block_size);

  checkCudaErrors(cudaMemcpyAsync(bytes.data(), (block.get())->voxels,
                                  block_size, cudaMemcpyDefault, cuda_stream));

  return bytes;
}

template <typename VoxelType>
void deserializeBlock(const std::vector<Byte>& bytes,
                      unified_ptr<VoxelBlock<VoxelType>>& block,
                      const CudaStream cuda_stream) {
  CHECK_EQ(bytes.size(), sizeof(block->voxels));

  checkCudaErrors(cudaMemcpyAsync((block.get())->voxels, bytes.data(),
                                  bytes.size(), cudaMemcpyDefault,
                                  cuda_stream));
}

}  // namespace nvblox
