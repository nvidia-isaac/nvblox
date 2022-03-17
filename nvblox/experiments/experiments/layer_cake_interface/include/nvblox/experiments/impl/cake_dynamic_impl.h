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

template <typename LayerType>
LayerType* LayerCakeDynamic::add() {
  if (layers_.count(typeid(LayerType)) == 0) {
    // Allocate
    auto layer_ptr = std::make_unique<LayerType>(
        sizeArgumentFromVoxelSize<LayerType>(voxel_size_), memory_type_);
    LayerType* return_ptr = layer_ptr.get();
    // Store (as BaseLayer ptr)
    layers_.emplace(std::type_index(typeid(LayerType)), std::move(layer_ptr));
    LOG(INFO) << "Adding Layer with type: " << typeid(LayerType).name()
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
LayerType* LayerCakeDynamic::getPtr() {
  const auto it = layers_.find(std::type_index(typeid(LayerType)));
  if (it != layers_.end()) {
    BaseLayer* base_ptr = it->second.get();
    LayerType* ptr = dynamic_cast<LayerType*>(base_ptr);
    CHECK_NOTNULL(ptr);
    return ptr;
  } else {
    LOG(WARNING) << "Request for a LayerType which is not in the cake.";
    return nullptr;
  }
}

template <typename LayerType>
bool LayerCakeDynamic::exists() const {
  const auto it = layers_.find(std::type_index(typeid(LayerType)));
  return (it != layers_.end());
}

template <typename... LayerTypes>
LayerCakeDynamic LayerCakeDynamic::create(float voxel_size,
                                          MemoryType memory_type) {
  static_assert(unique_types<LayerTypes...>::value,
                "At the moment we only support LayerCakes containing unique "
                "LayerTypes.");
  LayerCakeDynamic cake(voxel_size, memory_type);
  BaseLayer* ignored[] = {cake.add<LayerTypes>()...};
  return cake;
}

}  // namespace experiments
}  // namespace nvblox
