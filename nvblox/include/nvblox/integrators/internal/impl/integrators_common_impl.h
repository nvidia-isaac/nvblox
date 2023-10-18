/*
Copyright 2022-2023 NVIDIA CORPORATION

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

template <typename BlockType>
__host__ std::vector<BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    BlockLayer<BlockType>* layer_ptr) {
  std::vector<BlockType*> block_ptrs;
  block_ptrs.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    typename BlockType::Ptr block_ptr = layer_ptr->getBlockAtIndex(block_index);
    CHECK(block_ptr);
    block_ptrs.push_back(block_ptr.get());
  }
  return block_ptrs;
}

template <typename BlockType>
__host__ std::vector<const BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    const BlockLayer<BlockType>& layer) {
  std::vector<const BlockType*> block_ptrs;
  block_ptrs.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    const typename BlockType::ConstPtr block_ptr =
        layer.getBlockAtIndex(block_index);
    CHECK(block_ptr);
    block_ptrs.push_back(block_ptr.get());
  }
  return block_ptrs;
}

template <typename BlockType>
void allocateBlocksWhereRequired(const std::vector<Index3D>& block_indices,
                                 BlockLayer<BlockType>* layer,
                                 const CudaStream& cuda_stream) {
  layer->allocateBlocksAtIndices(block_indices, cuda_stream);
  cuda_stream.synchronize();
}

template <class... Args>
void expandBuffersIfRequired(size_t required_min_size, CudaStream cuda_stream,
                             Args... args) {
  static_assert((std::is_pointer<Args>::value && ...));
  const bool at_least_one_vector_smaller =
      (... || (args->capacity() < required_min_size));
  if (at_least_one_vector_smaller) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * required_min_size);
    (args->reserveAsync(new_size, cuda_stream), ...);
  }
}

template <typename BlockType>
void transferBlockPointersToDevice(
    const std::vector<Index3D>& block_indices, CudaStream cuda_stream,
    BlockLayer<BlockType>* layer_ptr, host_vector<BlockType*>* block_ptrs_host,
    device_vector<BlockType*>* block_ptrs_device) {
  if (block_indices.empty()) {
    return;
  }
  // Get the device pointers associated with this indices
  const std::vector<BlockType*> block_ptrs =
      getBlockPtrsFromIndices(block_indices, layer_ptr);
  // Expand the buffers if they're too small
  const int num_blocks = block_indices.size();
  expandBuffersIfRequired(num_blocks, cuda_stream, block_ptrs_device,
                          block_ptrs_host);
  // Stage on the host pinned memory
  block_ptrs_host->copyFromAsync(block_ptrs, cuda_stream);
  // Transfer to the device
  block_ptrs_device->copyFromAsync(*block_ptrs_host, cuda_stream);
}

inline void transferBlocksIndicesToDevice(
    const std::vector<Index3D>& block_indices, CudaStream cuda_stream,
    host_vector<Index3D>* block_indices_host,
    device_vector<Index3D>* block_indices_device) {
  if (block_indices.empty()) {
    return;
  }
  // Expand the buffers if they're too small
  const int num_blocks = block_indices.size();
  expandBuffersIfRequired(num_blocks, cuda_stream, block_indices_device,
                          block_indices_host);
  // Stage on the host pinned memory
  block_indices_host->copyFromAsync(block_indices, cuda_stream);
  // Transfer to the device
  block_indices_device->copyFromAsync(*block_indices_host, cuda_stream);
}

}  // namespace nvblox
