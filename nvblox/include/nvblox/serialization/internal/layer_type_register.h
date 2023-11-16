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

#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>

#include "nvblox/core/types.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// Struct of various layer parameters of different types.
struct LayerParameterStruct {
  std::map<std::string, std::string> string_params;
  std::map<std::string, int> int_params;
  std::map<std::string, float> float_params;
};

/// Struct holding callbacks for serialization functions for various layer
/// types.
struct LayerSerializationFunctions {
  // Serialization functions.
  typedef std::function<LayerParameterStruct(const BaseLayer*)>
      SerializeLayerParametersFunction;
  typedef std::function<std::vector<Index3D>(const BaseLayer*)>
      GetLayerDataIndicesFunction;
  typedef std::function<std::vector<Byte>(const BaseLayer*, const Index3D&,
                                          const CudaStream cuda_stream)>
      SerializeLayerDataFunction;

  // Deserialization functions.
  typedef std::function<std::unique_ptr<BaseLayer>(MemoryType,
                                                   const LayerParameterStruct&)>
      ConstructLayerFunction;
  typedef std::function<void(const Index3D&, const std::vector<Byte>&,
                             BaseLayer*, const CudaStream cuda_stream)>
      AddDataToLayerFunction;

  // The 5 functions that have to be defined to serialize and deserialize a
  // layer type.
  SerializeLayerParametersFunction serialize_params;
  GetLayerDataIndicesFunction get_data_indices;
  SerializeLayerDataFunction serialize_data;

  ConstructLayerFunction construct_layer;
  AddDataToLayerFunction add_data;
};

/// A class that allows registering a layer type to be used for serialization.
class LayerTypeRegister {
 public:
  /// This is the function all classes need to implement. type_index is the
  /// *derived* type. This is not what's stored in the index as it is *not*
  /// portable, but used by the LayerCake later on.

  // No constructor!
  LayerTypeRegister() = delete;

  /// Register a type and how to construct one.
  static void registerType(
      const std::string& type_name, const std::type_index& type_index,
      const LayerSerializationFunctions& serialization_functions);

  /// Create a layer of a given type.
  static std::unique_ptr<BaseLayer> createLayer(
      const std::string& type_name, MemoryType memory_type,
      const LayerParameterStruct& layer_params);

  /// Get the struct of function callbacks.
  static LayerSerializationFunctions getSerializationFunctions(
      const std::string& type_name);

  /// Get the name of the layer based on its type index.
  static std::string getLayerName(const std::type_index& type_index);

  /// Look up the type index based on a layer name.
  static std::type_index getLayerTypeIndex(const std::string& type_name);

 private:
  static std::unordered_map<std::string, LayerSerializationFunctions>
      layer_function_index_;
  static std::unordered_map<std::type_index, std::string> layer_name_index_;
};

}  // namespace nvblox
