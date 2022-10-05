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

#include "nvblox/core/blox.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"

namespace nvblox {

// ------------------- Serialization ----------------------
// Base template.
template <typename BlockType>
std::vector<Byte> serializeBlock(const unified_ptr<BlockType>& block);

// Voxel specializations
template <typename VoxelType>
std::vector<Byte> serializeBlock(
    const unified_ptr<const VoxelBlock<VoxelType>>& block);

// ------------------- Deserialization ----------------------
template <typename BlockType>
void deserializeBlock(const std::vector<Byte>& bytes,
                      unified_ptr<BlockType>& block);

template <typename VoxelType>
void deserializeBlock(const std::vector<Byte>& bytes,
                      unified_ptr<VoxelBlock<VoxelType>>& block);

}  // namespace nvblox

#include "nvblox/serialization/impl/block_serialization_impl.h"
