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
#include "nvblox/core/unified_vector.h"

namespace nvblox {

typedef class stdgpu::unordered_set<Index3D, Index3DHash,
                                    std::equal_to<Index3D>>
    Index3DDeviceSetType;

struct Index3DDeviceSet {
  Index3DDeviceSet(size_t size);
  ~Index3DDeviceSet();

  /// Clear and resize operations which allow reusing a single set object.
  /// If resizing to a smaller size, will *NOT* delete the existing set.
  /// Clears the contents of the set in any case.
  void resize(size_t size);

  /// Clear the contents of the set.
  void clear();

  Index3DDeviceSetType set;
};

// Copies the contents of a Index3DDeviceSet to an std::vector.
void copySetToVector(const Index3DDeviceSetType& set,
                     std::vector<Index3D>* vec);
void copySetToDeviceVectorAsync(const Index3DDeviceSetType& set,
                                device_vector<Index3D>* vec,
                                const CudaStream cuda_stream);

}  // namespace nvblox
