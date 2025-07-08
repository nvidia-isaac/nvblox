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

#include "nvblox/utils/params.h"

namespace nvblox {

constexpr Param<MemoryType>::Description kMemoryTypeParamDesc{
    "memory_type", MemoryType::kDevice, "Type of memory used by the pool."};
constexpr Param<int>::Description kNumPreallocatedBlocksParamDesc{
    "num_preallocated_blocks", 2048,
    "The number of blocks allocated in each layer during startup. A larger "
    "number will reduce the amount of memory allocations during startup."};
constexpr Param<float>::Description kExpansionFactor{
    "expansion_factor", 2.F,
    "Block memory pool expansion factor. When expanding the block pool, this "
    "number is multiplied with the current number of allocated blocks to get "
    "the new size. A value less than one means that only the bare-minimum "
    "amount of blocks will be allocated."};

struct BlockMemoryPoolParams {
  BlockMemoryPoolParams() = default;

  /// Construct with default args and custom memory type
  BlockMemoryPoolParams(const MemoryType _memory_type) {
    memory_type = _memory_type;
  }
  Param<MemoryType> memory_type{kMemoryTypeParamDesc};
  Param<int> num_preallocated_blocks{kNumPreallocatedBlocksParamDesc};
  Param<float> expansion_factor{kExpansionFactor};
};

} // namespace nvblox
