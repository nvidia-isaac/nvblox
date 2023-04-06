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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

#include <stdgpu/cstddef.h>
#include <stdgpu/unordered_map.cuh>

#include "nvblox/utils/logging.h"

#include "nvblox/core/hash.h"
#include "nvblox/core/types.h"

namespace nvblox {

template <typename BlockType>
using IndexBlockPair = thrust::pair<Index3D, BlockType*>;

template <typename BlockType>
using ConstIndexBlockPair = thrust::pair<const Index3D, BlockType*>;

template <typename BlockType>
using Index3DDeviceHashMapType =
    stdgpu::unordered_map<Index3D, BlockType*, Index3DHash,
                          std::equal_to<Index3D>>;

template <typename BlockType>
class GPUHashImpl {
 public:
  GPUHashImpl() = default;
  GPUHashImpl(int max_num_blocks);
  ~GPUHashImpl();

  size_t size() const { return static_cast<size_t>(max_num_blocks_); }

  stdgpu::index_t max_num_blocks_;
  Index3DDeviceHashMapType<BlockType> impl_;
};

}  // namespace nvblox

#include "nvblox/gpu_hash/internal/cuda/impl/gpu_hash_interface_impl.cuh"
