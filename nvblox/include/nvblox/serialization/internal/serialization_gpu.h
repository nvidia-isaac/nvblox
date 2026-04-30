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
#pragma once

#include <memory>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// Class for block layer serialization
template <typename LayerType, typename T>
class LayerSerializerGpuInternal {
 public:
  /// Serialize a block layer into a host vector.
  ///
  /// After serialization, the voxels in the block layer will be layed out in
  /// contigous memory, according to the following structure:
  ///    [ B0_v0, B0_v1, ... B0_nv, B1_v0, B1_vn, ... Bm_v0, Bm_v1, Bm_v2 ]
  /// where:
  ///    Bi_vj is the j:th voxel inside the i:th block
  ///    m: Number of blocks to serialize.
  ///    n: Number of voxels in a block.
  ///
  /// To recover the underlying block structure, an block offset vector is
  /// provided. Size of offsets is num_blocks+1. The first element is always
  /// zero and the last element always equals the total size of the serialized
  /// vector. The size of block n can be computed as offsets[n+1] -
  /// offsets[n]
  ///
  /// Empty blocks will appear still appear in the offset vector, but with zero
  /// size.
  ///
  /// Note that This class should normally not be used directly, instead it is
  /// recommended to use one of the per-layer specializations.
  ///
  ///
  /// @param layer                       Layer to serialize
  /// @param block_indices_to_serialize  Relevant block indices in mesh layer
  /// @param serialized_output           Resulting contiguous buffer
  /// @param offsets_output              Resulting offsets
  /// @param get_data_and_size           Functor that given a block, returns a
  ///                                    (ptr, size) pair
  //                                     where ptr is a pointer to the beginning
  //                                     of the block and size is the number of
  //                                     elements that should be serialized
  /// @param cuda_stream                 Cuda stream. Will *not* be synced.
  ///
  /// @tparam OutputVector  Vector type for serialized_output. Defaults to
  ///                       host_vector. Pass device_vector to keep the result
  ///                       in GPU memory and avoid the PCIe round-trip.
  template <template <typename> class OutputVector = host_vector>
  void serializeAsync(const LayerType& layer,
                      const std::vector<Index3D>& block_indices_to_serialize,
                      OutputVector<T>& serialized_output,
                      host_vector<int32_t>& offsets_output,
                      std::function<std::pair<const T*, int>(
                          const typename LayerType::BlockType* block)>
                          get_data_and_size,
                      const CudaStream& cuda_stream);

 private:
  // Scratch data. Pointers to vectors that should be serialized
  host_vector<const T*> vector_ptrs_;
};

}  // namespace nvblox
