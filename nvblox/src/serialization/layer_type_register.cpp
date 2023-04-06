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
#include "nvblox/serialization/internal/layer_type_register.h"

namespace nvblox {

// You have to do this so stuff is created I guess.
std::unordered_map<std::string, LayerSerializationFunctions>
    LayerTypeRegister::layer_function_index_;
std::unordered_map<std::type_index, std::string>
    LayerTypeRegister::layer_name_index_;

// Register a type and how to construct one.
void LayerTypeRegister::registerType(
    const std::string& type_name, const std::type_index& type_index,
    const LayerSerializationFunctions& serialization_functions) {
  layer_name_index_.emplace(type_index, type_name);
  layer_function_index_.emplace(type_name, serialization_functions);
}

// Create a layer of a given type.
std::unique_ptr<BaseLayer> LayerTypeRegister::createLayer(
    const std::string& type_name, MemoryType memory_type,
    const LayerParameterStruct& layer_params) {
  LayerSerializationFunctions::ConstructLayerFunction func_pointer =
      LayerTypeRegister::getSerializationFunctions(type_name).construct_layer;

  if (func_pointer == nullptr) {
    return std::unique_ptr<BaseLayer>();
  }

  return func_pointer(memory_type, layer_params);
}

// Get the struct with all the functions for this layer.
LayerSerializationFunctions LayerTypeRegister::getSerializationFunctions(
    const std::string& type_name) {
  auto it = layer_function_index_.find(type_name);
  if (it == layer_function_index_.end()) {
    return LayerSerializationFunctions();
  }
  return it->second;
}

// Get the name of the layer based on its type index.
std::string LayerTypeRegister::getLayerName(const std::type_index& type_index) {
  auto it = layer_name_index_.find(type_index);
  if (it == layer_name_index_.end()) {
    LOG(WARNING) << "Layer not found, valid options: ";
    for (const auto kv : layer_name_index_) {
      LOG(WARNING) << kv.first.name() << " : " << kv.second;
    }
    return std::string();
  }
  return it->second;
}

std::type_index LayerTypeRegister::getLayerTypeIndex(
    const std::string& type_name) {
  // Find the type info of the string.
  auto it = std::find_if(
      layer_name_index_.begin(), layer_name_index_.end(),
      [&type_name](const std::pair<std::type_index, std::string>& p) {
        return p.second == type_name;
      });

  if (it == layer_name_index_.end()) {
    return std::type_index(typeid(nullptr));
  }

  return it->first;
}

}  // namespace nvblox