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

#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/blox.h"

namespace nvblox {

// ------------------- Serialization ----------------------
// Base template.
template <typename BlockType>
std::vector<Byte> serializeBlock(const unified_ptr<BlockType>& block,
                                 const CudaStream cuda_stream);

// Voxel specializations
template <typename VoxelType>
std::vector<Byte> serializeBlock(
    const unified_ptr<const VoxelBlock<VoxelType>>& block,
    const CudaStream cuda_stream);

// ------------------- Deserialization ----------------------
template <typename BlockType>
void deserializeBlock(const std::vector<Byte>& bytes,
                      unified_ptr<BlockType>& block,
                      const CudaStream cuda_stream);

template <typename VoxelType>
void deserializeBlock(const std::vector<Byte>& bytes,
                      unified_ptr<VoxelBlock<VoxelType>>& block,
                      const CudaStream cuda_stream);

}  // namespace nvblox

#include "nvblox/serialization/internal/impl/block_serialization_impl.h"
