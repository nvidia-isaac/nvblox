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

#include <typeindex>
#include <typeinfo>

namespace nvblox {

// Bind default functions for the given types.
template <typename LayerType>
LayerSerializationFunctions bindDefaultFunctions() {
  LayerSerializationFunctions layer_functions;
  LayerSerializationFunctions::SerializeLayerParametersFunction
      lambda_serialize_params = [](const BaseLayer* base_layer) {
        const LayerType& layer = *dynamic_cast<const LayerType*>(base_layer);

        return serializeLayerParameters(layer);
      };
  layer_functions.serialize_params = lambda_serialize_params;

  LayerSerializationFunctions::GetLayerDataIndicesFunction
      lambda_get_data_indices = [](const BaseLayer* base_layer) {
        const LayerType& layer = *dynamic_cast<const LayerType*>(base_layer);

        return getLayerDataIndices(layer);
      };
  layer_functions.get_data_indices = lambda_get_data_indices;

  LayerSerializationFunctions::SerializeLayerDataFunction
      lambda_serialize_data = [](const BaseLayer* base_layer,
                                 const Index3D& index,
                                 const CudaStream cuda_stream) {
        const LayerType& layer = *dynamic_cast<const LayerType*>(base_layer);

        return serializeLayerDataAtIndex(layer, index, cuda_stream);
      };
  layer_functions.serialize_data = lambda_serialize_data;

  LayerSerializationFunctions::ConstructLayerFunction lambda_construct_layer =
      [](MemoryType memory_type, const LayerParameterStruct& layer_params) {
        std::unique_ptr<LayerType> layer =
            deserializeLayerParameters<typename GetLayerType<LayerType>::type>(
                memory_type, layer_params);
        return layer;
      };
  layer_functions.construct_layer = lambda_construct_layer;

  LayerSerializationFunctions::AddDataToLayerFunction lambda_add_data =
      [](const Index3D& index, const std::vector<Byte>& data,
         BaseLayer* base_layer, const CudaStream cuda_stream) {
        LayerType* layer = dynamic_cast<LayerType*>(base_layer);

        addDataToLayer(index, data, layer, cuda_stream);
      };
  layer_functions.add_data = lambda_add_data;

  return layer_functions;
}

void registerCommonTypes() {
  LayerTypeRegister::registerType("tsdf_layer", typeid(TsdfLayer),
                                  bindDefaultFunctions<TsdfLayer>());
  LayerTypeRegister::registerType("esdf_layer", typeid(EsdfLayer),
                                  bindDefaultFunctions<EsdfLayer>());
  LayerTypeRegister::registerType("color_layer", typeid(ColorLayer),
                                  bindDefaultFunctions<ColorLayer>());
  LayerTypeRegister::registerType("occupancy_layer", typeid(OccupancyLayer),
                                  bindDefaultFunctions<OccupancyLayer>());
}

}  // namespace nvblox
