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

#include <memory>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/serialization/internal/serialization_gpu.h"

namespace nvblox {

/// Container for storing a serialized mesh.
///
/// @tparam AppearanceType  Per-vertex appearance (e.g. Color, FeatureArray).
/// @tparam VecType         Vector type used for the large data fields.
///                         Defaults to host_vector, preserving the original
///                         behaviour (pinned host memory, CPU-accessible).
///                         Pass device_vector to keep vertex/appearance/index
///                         data in GPU memory and avoid the PCIe round-trip.
///                         The small offset arrays always stay in host memory
///                         because they are written by CPU code.
template <typename AppearanceType,
          template <typename> class VecType = host_vector>
struct SerializedMeshLayer {
  /// Serialized mesh components (in VecType memory)
  VecType<Vector3f> vertices;
  VecType<AppearanceType> vertex_appearances;
  VecType<int> triangle_indices;

  /// Offsets for each mesh block in the output vector.
  /// Size of offsets is num_blocks+1. The first element is always
  /// zero and the last element always equals the total size of the serialized
  /// vector. The size of block n can be computed as offsets[n+1] -
  /// offsets[n].
  /// These always live in host memory — they are written by CPU code.
  host_vector<int32_t> vertex_block_offsets;
  host_vector<int32_t> triangle_index_block_offsets;

  /// Indices of serialized mesh blocks
  std::vector<Index3D> block_indices;

  // The helpers below access elements by CPU-side indexing and are only valid
  // when VecType == host_vector.

  /// Get an iterator to the given triangle block
  auto triangleBlockItr(size_t block_index) const {
    CHECK(block_index < vertex_block_offsets.size());
    return std::next(triangle_indices.begin(),
                     triangle_index_block_offsets[block_index]);
  }

  /// Get a vertex given a block index and a vertex index inside the block
  const Vector3f& getVertex(size_t block_index, size_t vertex_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return vertices[vertex_block_offsets[block_index] + vertex_index];
  }

  /// Get a appearance given a block index and a vertex index inside the block
  const AppearanceType& getAppearance(size_t block_index,
                                      size_t vertex_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return vertex_appearances[vertex_block_offsets[block_index] + vertex_index];
  }

  /// Get a Triangle index given a block index and a triangle index inside the
  /// block
  const int& getTriangleIndex(size_t block_index, size_t triangle_index) const {
    CHECK(block_index < triangle_index_block_offsets.size() - 1);
    return triangle_indices[triangle_index_block_offsets[block_index] +
                            triangle_index];
  }

  /// Helper function to get num vertices in a block
  size_t getNumVerticesInBlock(size_t block_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return vertex_block_offsets[block_index + 1] -
           vertex_block_offsets[block_index];
  }

  /// Helper function to get num triangles in a block
  size_t getNumTriangleIndicesInBlock(size_t block_index) const {
    CHECK(block_index < triangle_index_block_offsets.size() - 1);
    return triangle_index_block_offsets[block_index + 1] -
           triangle_index_block_offsets[block_index];
  }
};

using SerializedColorMeshLayer = SerializedMeshLayer<Color>;
using SerializedFeatureMeshLayer = SerializedMeshLayer<FeatureArray>;

/// Class for serialization
///
/// Mesh needs special treatment since the data int he blocks are stored
/// as struct-of-arrays rather than array-of-structs
template <typename AppearanceType>
class MeshSerializerGpu {
 public:
  MeshSerializerGpu();
  virtual ~MeshSerializerGpu() = default;
  using MeshBlockType = MeshBlock<AppearanceType>;
  using SerializedLayerType = SerializedMeshLayer<AppearanceType, host_vector>;
  using SerializedLayerTypeDevice =
      SerializedMeshLayer<AppearanceType, device_vector>;
  using MeshLayerType = MeshBlockLayer<AppearanceType>;

  /// Serialize a mesh layer into host (pinned) memory.
  ///
  /// All requested blocks will be serialized and placed in output host
  /// vectors. This implementation is more effective than issuing a memcpy
  /// for each block.
  ///
  /// @attention: Input mesh layer must be in device or unified memory
  ///
  /// @param mesh_layer                  Mesh layer to serialize
  /// @param block_indices_to_serialize  Requested block indices
  /// @param cuda_stream                 Cuda stream. Synced before return.
  std::shared_ptr<SerializedLayerType> serialize(
      const MeshLayerType& mesh_layer,
      const std::vector<Index3D>& block_indices_to_serialize,
      const CudaStream& cuda_stream);

  /// Serialize a mesh layer into device memory
  ///
  /// Like serialize(), but the large data fields (vertices, appearances,
  /// triangle_indices) are written directly to GPU memory. The stream is
  /// NOT synced before return — downstream GPU work can be queued on the
  /// same stream without stalling the CPU.
  ///
  /// @attention: Input mesh layer must be in device or unified memory
  ///
  /// @param mesh_layer                  Mesh layer to serialize
  /// @param block_indices_to_serialize  Requested block indices
  /// @param cuda_stream                 CUDA stream used for async work
  /// @param synchronize_stream          If false (default), the stream is not
  ///                                    synced.
  std::shared_ptr<SerializedLayerTypeDevice> serializeToDevice(
      const MeshLayerType& mesh_layer,
      const std::vector<Index3D>& block_indices_to_serialize,
      const CudaStream& cuda_stream, bool synchronize_stream = false);

  /// Get the serialized mesh (host variant)
  std::shared_ptr<SerializedLayerType> getSerializedLayer() const {
    return serialized_mesh_host_;
  }

  // Get the serialized mesh (device variant)
  std::shared_ptr<SerializedLayerTypeDevice> getSerializedLayerDevice() const {
    return serialized_mesh_device_;
  }

 private:
  /// Shared implementation: fills output with serialized blocks.
  /// @param synchronize_stream  If true, cuda_stream.synchronize() before
  /// return.
  template <template <typename> class VecType>
  void serializeInto(SerializedMeshLayer<AppearanceType, VecType>* output,
                     const MeshLayerType& mesh_layer,
                     const std::vector<Index3D>& block_indices_to_serialize,
                     const CudaStream& cuda_stream, bool synchronize_stream);

  LayerSerializerGpuInternal<MeshLayerType, Vector3f> vertex_serializer_;
  LayerSerializerGpuInternal<MeshLayerType, AppearanceType>
      appearance_serializer_;
  LayerSerializerGpuInternal<MeshLayerType, int> triangle_index_serializer_;

  std::shared_ptr<SerializedLayerType> serialized_mesh_host_;
  std::shared_ptr<SerializedLayerTypeDevice> serialized_mesh_device_;
};

using ColorMeshSerializerGpu = MeshSerializerGpu<Color>;
using FeatureMeshSerializerGpu = MeshSerializerGpu<FeatureArray>;

}  // namespace nvblox
