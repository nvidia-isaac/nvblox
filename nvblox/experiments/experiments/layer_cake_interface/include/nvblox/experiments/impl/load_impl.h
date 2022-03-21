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
namespace io {

template <typename LayerType>
bool loadLayer(const std::string& filename, LayerType* layer) {
  LOG(INFO) << "Loading layer of type: " << typeid(LayerType).name();
  // TODO: Actually load something.
  return true;
}

template <class... LayerTypes>
struct Loader;

template <class LayerType>
struct Loader<LayerType> {
  static bool load(const std::string& filename, LayerCakeDynamic* cake) {
    if (cake->exists<LayerType>()) {
      LOG(WARNING) << "Not loading layer. A layer of type: "
                   << typeid(LayerType).name()
                   << " already exists in the cake.";
      return false;
    }
    return loadLayer<LayerType>(filename, cake->add<LayerType>());
  }
};

template <class LayerType, class... LayerTypesRest>
struct Loader<LayerType, LayerTypesRest...> {
  static bool load(const std::string& filename, LayerCakeDynamic* cake) {
    return Loader<LayerType>::load(filename, cake) &&
           Loader<LayerTypesRest...>::load(filename, cake);
  }
};

template <typename... LayerTypes>
bool load(const std::string& filename, LayerCakeDynamic* cake) {
  return Loader<LayerTypes...>::load(filename, cake);
}

}  // io
}  // experiments
}  // nvblox
