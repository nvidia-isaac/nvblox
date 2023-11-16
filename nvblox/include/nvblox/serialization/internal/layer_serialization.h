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
#include <string>
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/layer.h"
#include "nvblox/serialization/internal/block_serialization.h"
#include "nvblox/serialization/internal/layer_type_register.h"

namespace nvblox {

// ------------------- Serialization ----------------------
// Base template.
template <typename LayerType>
LayerParameterStruct serializeLayerParameters(const LayerType& layer);

template <typename LayerType>
std::vector<Index3D> getLayerDataIndices(const LayerType& layer);

template <typename LayerType>
std::vector<Byte> serializeLayerDataAtIndex(const LayerType& layer,
                                            const Index3D& index,
                                            const CudaStream cuda_stream);

// Block specializations
template <typename VoxelType>
LayerParameterStruct serializeLayerParameters(
    const VoxelBlockLayer<VoxelType>& layer);

template <typename VoxelType>
std::vector<Index3D> getLayerDataIndices(
    const VoxelBlockLayer<VoxelType>& layer);

template <typename VoxelType>
std::vector<Byte> serializeLayerDataAtIndex(
    const VoxelBlockLayer<VoxelType>& layer, const Index3D& index,
    const CudaStream cuda_stream);

// ------------------- Deserialization ----------------------

template <typename LayerType>
struct GetLayerType {
  using type = LayerType;
};

template <typename VoxelType>
struct GetLayerType<VoxelBlockLayer<VoxelType>> {
  using type = VoxelType;
};

// Base templates
// Deserialize Layer Parameters base template is too ambiguous so cannot be
// present here.

template <typename LayerType>
void addDataToLayer(const Index3D& index, const std::vector<Byte>& data,
                    LayerType* layer);

// Block specializations
template <typename VoxelType>
std::unique_ptr<VoxelBlockLayer<VoxelType>> deserializeLayerParameters(
    MemoryType memory_type, const LayerParameterStruct& params);

template <typename VoxelType>
void addDataToLayer(const Index3D& index, const std::vector<Byte>& data,
                    VoxelBlockLayer<VoxelType>* layer);

}  // namespace nvblox

#include "nvblox/serialization/internal/impl/layer_serialization_impl.h"
