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
#include <typeindex>
#include <unordered_map>

#include <nvblox/core/layer.h>
#include <nvblox/core/types.h>

#include "nvblox/experiments/cake_common.h"

namespace nvblox {
namespace experiments {

class LayerCakeDynamic {
 public:
  LayerCakeDynamic(float voxel_size, MemoryType memory_type)
      : voxel_size_(voxel_size), memory_type_(memory_type) {}

  template <typename LayerType>
  LayerType* add();

  template <typename LayerType>
  LayerType* getPtr();

  template <typename LayerType>
  bool exists() const;

  // Factory accepting a list of LayerTypes
  template <typename... LayerTypes>
  static LayerCakeDynamic create(float voxel_size, MemoryType memory_type);

 private:
  // Params
  const float voxel_size_;
  const MemoryType memory_type_;

  // NOTE(alexmillane): Could move to multimap if we want more than one layer
  //                    with the same type
  std::unordered_map<std::type_index, std::unique_ptr<BaseLayer>> layers_;
};

}  // namespace experiments
}  // namespace nvblox

#include "nvblox/experiments/impl/cake_dynamic_impl.h"
