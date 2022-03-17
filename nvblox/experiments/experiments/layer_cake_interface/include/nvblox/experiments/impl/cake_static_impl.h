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

#include "nvblox/experiments/cake_common.h"

namespace nvblox {
namespace experiments {

template <class... LayerTypes>
template <typename LayerType>
LayerType* LayerCakeStatic<LayerTypes...>::getPtr() {
  static_assert(count_type_occurrence<LayerType, LayerTypes...>::value > 0,
                "LayerCake does not contain requested layer");
  static_assert(count_type_occurrence<LayerType, LayerTypes...>::value < 2,
                "To get a Layer by type, the LayerCake must only contain "
                "only a single matching LayerType");
  return &std::get<LayerType>(layers_);
}

template <class... LayerTypes>
template <typename LayerType>
const LayerType& LayerCakeStatic<LayerTypes...>::get() const {
  static_assert(count_type_occurrence<LayerType, LayerTypes...>::value > 0,
                "LayerCake does not contain requested layer");
  static_assert(count_type_occurrence<LayerType, LayerTypes...>::value < 2,
                "To get a Layer by type, the LayerCake must only contain "
                "only a single matching LayerType");
  return std::get<LayerType>(layers_);
}

template <class... LayerTypes>
template <typename LayerType>
int LayerCakeStatic<LayerTypes...>::count() const {
  return count_type_occurrence<LayerType, LayerTypes...>::value;
}

}  // namespace experiments
}  // namespace nvblox
