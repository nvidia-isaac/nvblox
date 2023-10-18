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

#include "nvblox/core/variadic_template_tools.h"

namespace nvblox {

template <typename LayerType>
LayerType* LayerCake::add(MemoryType memory_type) {
  if (layers_.count(typeid(LayerType)) == 0) {
    // Allocate
    CHECK_GT(voxel_size_, 0.0f);
    auto layer_ptr = std::make_unique<LayerType>(
        sizeArgumentFromVoxelSize<LayerType>(voxel_size_), memory_type);
    LayerType* return_ptr = layer_ptr.get();
    // Store (as BaseLayer ptr)
    layers_.emplace(std::type_index(typeid(LayerType)), std::move(layer_ptr));
    LOG(INFO) << "Adding Layer with type: " << typeid(LayerType).name()
              << ", voxel_size: " << voxel_size_
              << ", and memory_type: " << toString(memory_type)
              << " to LayerCake.";
    return return_ptr;
  } else {
    LOG(WARNING) << "Request to add a LayerType that's already in the cake. "
                    "Currently we only support single layers of each "
                    "LayerType. So we did nothing.";
    return nullptr;
  }
}

template <typename LayerType>
const LayerType* LayerCake::getConstPtr() const {
  auto it = layers_.find(std::type_index(typeid(LayerType)));
  if (it != layers_.end()) {
    const BaseLayer* base_ptr = it->second.get();
    const LayerType* ptr = dynamic_cast<const LayerType*>(base_ptr);
    CHECK_NOTNULL(ptr);
    return ptr;
  } else {
    LOG(WARNING) << "Request for a LayerType: " << typeid(LayerType).name()
                 << " which is not in the cake.";
    return nullptr;
  }
}

template <typename LayerType>
LayerType* LayerCake::getPtr() {
  auto it = layers_.find(std::type_index(typeid(LayerType)));
  if (it != layers_.end()) {
    BaseLayer* base_ptr = it->second.get();
    LayerType* ptr = dynamic_cast<LayerType*>(base_ptr);
    CHECK_NOTNULL(ptr);
    return ptr;
  } else {
    LOG(WARNING) << "Request for a LayerType: " << typeid(LayerType).name()
                 << " which is not in the cake.";
    return nullptr;
  }
}

template <typename LayerType>
const LayerType& LayerCake::get() const {
  const LayerType* ptr = getConstPtr<LayerType>();
  CHECK_NOTNULL(ptr);
  return *ptr;
}

template <typename LayerType>
bool LayerCake::exists() const {
  const auto it = layers_.find(std::type_index(typeid(LayerType)));
  return (it != layers_.end());
}

template <typename... LayerTypes>
LayerCake LayerCake::create(float voxel_size, MemoryType memory_type) {
  static_assert(unique_types<LayerTypes...>::value,
                "At the moment we only support LayerCakes containing unique "
                "LayerTypes.");
  LayerCake cake(voxel_size);
  BaseLayer* ignored[] = {cake.add<LayerTypes>(memory_type)...};
  static_cast<void>(ignored);
  return cake;
}

template <typename... LayerTypes, typename... MemoryTypes>
LayerCake LayerCake::create(float voxel_size, MemoryTypes... memory_types) {
  static_assert(unique_types<LayerTypes...>::value,
                "At the moment we only support LayerCakes containing unique "
                "LayerTypes.");
  LayerCake cake(voxel_size);
  BaseLayer* ignored[] = {cake.add<LayerTypes>(memory_types)...};
  static_cast<void>(ignored);
  return cake;
}

void LayerCake::insert(const std::type_index& type_index,
                       std::unique_ptr<BaseLayer>&& layer) {
  layers_.emplace(type_index, std::move(layer));
}

}  // namespace nvblox
