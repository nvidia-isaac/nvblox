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

namespace nvblox {
namespace experiments {

template <class... LayerTypes>
class LayerCakeStatic {
 public:
  LayerCakeStatic(float voxel_size, MemoryType memory_type)
      : layers_(std::move(
            LayerTypes(sizeArgumentFromVoxelSize<LayerTypes>(voxel_size),
                       memory_type))...) {}

  // Access by LayerType.
  // This requires that the list of types contained in this LayerCakeStatic are
  // unique with respect to one another.
  template <typename LayerType>
  LayerType* getPtr();
  template <typename LayerType>
  const LayerType& get() const;

  template <typename LayerType>
  int count() const;

 private:
  std::tuple<LayerTypes...> layers_;
};

}  // namespace experiments
}  // namespace nvblox

#include "nvblox/experiments/impl/cake_static_impl.h"
