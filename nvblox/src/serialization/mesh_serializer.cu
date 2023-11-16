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

#include "nvblox/serialization/mesh_serializer.hpp"

#include <cuda_runtime.h>
#include <string>

#include "glog/logging.h"
#include "nvblox/core/internal/error_check.h"

namespace nvblox {

// Kernel that copies several vectors into a contigous chunk of memory
//
// Number of blocks:  Must equal num_vectors.
// Number of threads: Can be any positive value but a larger number than the
//   maximum vector size will not bring any gain.
//
// @param num_vectors        Number of vectors to serialize
// @param vectors            Vectors to serialize. Size: num_vectors
// @param offsets            Output buffer offsets. Last elements contain total
//                           num elements. Size: num_vectors+1
// @param serialized_buffer  Resulting buffer. Must have capacity for all
//                           elements
template <typename T>
void __global__ SerializeVectorsKernel(const int32_t num_vectors,
                                       const T** vectors,
                                       const int32_t* offsets,
                                       T* serialized_buffer) {
  const int32_t vector_index =
      blockIdx.x;  // Which vector does this block serialize?
  const int32_t element_index_start = threadIdx.x;

  // This should not happen if the kernel was launched with
  // num_blocks=num_vectors
  if (vector_index >= num_vectors) {
    return;
  }

  const int32_t offset = offsets[vector_index];
  const int32_t num_elements = offsets[vector_index + 1] - offset;

  // If vector size is larger than number of threads, we let each thread handle
  // several elements.
  for (int32_t index = element_index_start; index < num_elements;
       index += blockDim.x) {
    serialized_buffer[offset + index] = vectors[vector_index][index];
  }
}

template <typename LayerType, typename T>
void MeshComponentSerializer<LayerType, T>::serializeAsync(
    const LayerType& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<T>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<
        const unified_vector<T>&(const typename LayerType::BlockType* block)>
        get_vector,
    const CudaStream cuda_stream) {
  if (block_indices_to_serialize.empty()) {
    return;
  }

  // Iterate over all blocks to serialize, store their data pointers and
  // offsets
  offsets_output.resize(block_indices_to_serialize.size() + 1);
  vector_ptrs_.resize(block_indices_to_serialize.size());
  int32_t total_num_elements = 0;
  int32_t max_block_size = 0;
  for (size_t i = 0; i < block_indices_to_serialize.size(); ++i) {
    const typename LayerType::BlockType* block =
        layer.getBlockAtIndex(block_indices_to_serialize[i]).get();

    offsets_output[i] = total_num_elements;
    vector_ptrs_[i] = get_vector(block).data();
    total_num_elements += get_vector(block).size();

    max_block_size = std::max<int32_t>(max_block_size, get_vector(block).size());
  }

  // We'll need the total num of elements as well so we can compute the
  // size of the last vector
  offsets_output[offsets_output.size() - 1] = total_num_elements;

  // We use thread_id to determine which vector element to copy. This
  // allow for coalesced memory transfers since all threads in one warp
  // will read from contigous memory.
  constexpr int32_t kMaxNumThreads = 1024;
  const int32_t num_threads = std::min(max_block_size, kMaxNumThreads);

  // Process one layer-block per cuda-block
  const int32_t num_blocks = block_indices_to_serialize.size();

  // Run serialization.
  serialized_output.resize(total_num_elements);
  SerializeVectorsKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      block_indices_to_serialize.size(), vector_ptrs_.data(),
      offsets_output.data(), serialized_output.data());

  checkCudaErrors(cudaPeekAtLastError());
}

std::shared_ptr<const SerializedMesh> MeshSerializer::serializeMesh(
    const MeshLayer& mesh_layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    const CudaStream cuda_stream) {
  vertex_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, serialized_mesh_->vertices,
      serialized_mesh_->vertex_block_offsets,
      [](const MeshBlock* mesh_block) -> const unified_vector<Vector3f>& {
        return mesh_block->vertices;
      },
      cuda_stream);

  color_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, serialized_mesh_->colors,
      serialized_mesh_->vertex_block_offsets,
      [](const MeshBlock* mesh_block) -> const unified_vector<Color>& {
        return mesh_block->colors;
      },
      cuda_stream);

  triangle_index_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, serialized_mesh_->triangle_indices,
      serialized_mesh_->triangle_index_block_offsets,
      [](const MeshBlock* mesh_block) -> const unified_vector<int>& {
        return mesh_block->triangles;
      },
      cuda_stream);

  // Create an unique identifier for each block.
  serialized_mesh_->block_identifiers.resize(block_indices_to_serialize.size());
  for (size_t i = 0; i < block_indices_to_serialize.size(); ++i) {
    const Index3D& block_idx = block_indices_to_serialize[i];
    serialized_mesh_->block_identifiers[i] =
        std::to_string(block_idx.x()) + "_" + std::to_string(block_idx.y()) +
        "_" + std::to_string(block_idx.z());
  }

  cuda_stream.synchronize();

  return serialized_mesh_;
}

MeshSerializer::MeshSerializer()
    : serialized_mesh_(std::make_shared<SerializedMesh>()) {}

}  // namespace nvblox
