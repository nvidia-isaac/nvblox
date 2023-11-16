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

namespace nvblox {

/// Container for storing a serializeed mesh
struct SerializedMesh {
  /// Serialized mesh components
  host_vector<nvblox::Vector3f> vertices;
  host_vector<nvblox::Color> colors;
  host_vector<int> triangle_indices;

  /// Offsets for each mesh block in the output vector.
  /// Size of offsets is num_blocks+1. The first element is always
  /// zero and the last element always equals the total size of the serialized
  /// vector. The size of block n can be computed as offsets[n+1] -
  /// offsets[n]
  host_vector<int32_t> vertex_block_offsets;
  host_vector<int32_t> triangle_index_block_offsets;

  /// Unique identifier for each block that remain consistent between calls
  /// consecutive calls to the serializer.
  std::vector<std::string> block_identifiers;

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

/// Class for mesh serialization
template <typename LayerType, typename T>
class MeshComponentSerializer {
 public:
  /// Serialize one single component (vertices, colors, ...) of a mesh. Which
  /// component to serialize is determined by a provided functor.
  ///
  /// The function runs async to allow for efficent serialization of several
  /// components at once. Saved 0.6ms in serialize mesh benchmark.
  ///
  /// @param layer                       Layer to serialize
  /// @param block_indices_to_serialize  Relevant block indices in mesh layer
  /// @param serialized_output           Resulting contigous buffer
  /// @param offsets_output              Resuling offsets. See SerializedMesh
  ///                                    for detailled description
  /// @param get_vector                  Functor that returns vector pointer to
  ///                                    serialize given a mesh block
  /// @param cuda_stream                 Cuda stream. Will *not* be synced.
  void serializeAsync(const LayerType& layer,
                      const std::vector<Index3D>& block_indices_to_serialize,
                      host_vector<T>& serialized_output,
                      host_vector<int32_t>& offsets_output,
                      std::function<const unified_vector<T>&(
                          const typename LayerType::BlockType* block)>
                          get_vector,
                      const CudaStream cuda_stream);

 private:
  // Scratch data maintained by the class to avoid expensive reallocations.
  // Saved 0.8ms in serialize mesh benchmark.
  // Pinned host-vectors are used to allow for zero-copy on Jetson. Saved
  // 0.5ms in serialize mesh benchmark.

  // Pointers to vectors that should be serialized
  host_vector<const T*> vector_ptrs_;
};

/// Class for serialization
///
/// Mesh needs special tretment since the data int he blocks are stored
/// as struct-of-arrays rather than array-of-structs
class MeshSerializer final {
 public:
  MeshSerializer();

  /// Serialize a mesh layer
  ///
  /// All requested blocks will be serialized and placed in output host
  /// vectors. This implementation is more effictive than issuing a memcpy
  /// for each block.
  ///
  /// @attention: Input mesh layer must be in device or unified memory
  ///
  /// @param mesh_layer                  Mesh layer to serialize
  /// @param block_indices_to_serialize  Requested block indices
  /// @param cuda_stream                 Cuda stream
  std::shared_ptr<const SerializedMesh> serializeMesh(
      const nvblox::MeshLayer& mesh_layer,
      const std::vector<nvblox::Index3D>& block_indices_to_serialize,
      const nvblox::CudaStream cuda_stream);

  /// Get the serialized mesh
  std::shared_ptr<const SerializedMesh> getSerializedMesh() const {
    return serialized_mesh_;
  }

 private:
  MeshComponentSerializer<MeshLayer, Vector3f> vertex_serializer_;
  MeshComponentSerializer<MeshLayer, Color> color_serializer_;
  MeshComponentSerializer<MeshLayer, int> triangle_index_serializer_;

  std::shared_ptr<SerializedMesh> serialized_mesh_;
};

}  // namespace nvblox
