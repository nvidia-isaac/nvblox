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

#include "nvblox/serialization/mesh_serializer_gpu.h"

#include <cuda_runtime.h>
#include <string>

#include "glog/logging.h"
#include "nvblox/core/internal/error_check.h"

namespace nvblox {

template <typename AppearanceType>
template <template <typename> class VecType>
void MeshSerializerGpu<AppearanceType>::serializeInto(
    SerializedMeshLayer<AppearanceType, VecType>* output,
    const MeshLayerType& mesh_layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    const CudaStream& cuda_stream, const bool synchronize_stream) {
  vertex_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, output->vertices,
      output->vertex_block_offsets,
      [](const MeshBlockType* mesh_block)
          -> const std::pair<const Vector3f*, int> {
        return std::make_pair(mesh_block->vertices.data(),
                              mesh_block->vertices.size());
      },
      cuda_stream);

  appearance_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, output->vertex_appearances,
      output->vertex_block_offsets,
      [](const MeshBlockType* mesh_block)
          -> const std::pair<const AppearanceType*, int> {
        return std::make_pair(mesh_block->vertex_appearances.data(),
                              mesh_block->vertex_appearances.size());
      },
      cuda_stream);

  triangle_index_serializer_.serializeAsync(
      mesh_layer, block_indices_to_serialize, output->triangle_indices,
      output->triangle_index_block_offsets,
      [](const MeshBlockType* mesh_block) -> const std::pair<const int*, int> {
        return std::make_pair(mesh_block->triangles.data(),
                              mesh_block->triangles.size());
      },
      cuda_stream);

  output->block_indices = block_indices_to_serialize;

  if (synchronize_stream) {
    cuda_stream.synchronize();
  }
}

template <typename AppearanceType>
std::shared_ptr<SerializedMeshLayer<AppearanceType>>
MeshSerializerGpu<AppearanceType>::serialize(
    const MeshLayerType& mesh_layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    const CudaStream& cuda_stream) {
  serializeInto(serialized_mesh_host_.get(), mesh_layer,
                block_indices_to_serialize, cuda_stream, true);
  return serialized_mesh_host_;
}

template <typename AppearanceType>
std::shared_ptr<
    typename MeshSerializerGpu<AppearanceType>::SerializedLayerTypeDevice>
MeshSerializerGpu<AppearanceType>::serializeToDevice(
    const MeshLayerType& mesh_layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    const CudaStream& cuda_stream, const bool synchronize_stream) {
  serializeInto(serialized_mesh_device_.get(), mesh_layer,
                block_indices_to_serialize, cuda_stream, synchronize_stream);
  return serialized_mesh_device_;
}

template <typename AppearanceType>
MeshSerializerGpu<AppearanceType>::MeshSerializerGpu()
    : serialized_mesh_host_(std::make_shared<SerializedLayerType>()),
      serialized_mesh_device_(std::make_shared<SerializedLayerTypeDevice>()) {}

// Explicit instantiations.
template class MeshSerializerGpu<Color>;
template class MeshSerializerGpu<FeatureArray>;

}  // namespace nvblox
