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
#include <stdgpu/unordered_set.cuh>

#include "nvblox/core/hash.h"
#include "nvblox/core/types.h"

namespace nvblox {

typedef class stdgpu::unordered_set<Index3D, Index3DHash,
                                    std::equal_to<Index3D>>
    Index3DDeviceSet_t;

struct Index3DDeviceSet {
  Index3DDeviceSet(size_t size);
  ~Index3DDeviceSet();

  Index3DDeviceSet_t set;
};

// Copies the contents of a Index3DDeviceSet to an std::vector.
void copySetToVector(const Index3DDeviceSet_t& set, std::vector<Index3D>* vec);

}  // namespace nvblox
