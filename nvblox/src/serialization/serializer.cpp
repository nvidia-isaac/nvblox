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
#include <stdio.h>
#include <vector>
#include "nvblox/utils/logging.h"

#include "nvblox/serialization/internal/serializer.h"

namespace nvblox {

Serializer::Serializer() {
  // Common types are automatically registered.
  registerCommonTypes();
}

/// Open a particular file, default to open for reading.
Serializer::Serializer(const std::string& filename,
                       std::ios_base::openmode openmode) {
  sqlite_.open(filename, openmode);
}

/// Is the current file valid?
bool Serializer::valid() const { return sqlite_.valid(); }

/// Open a file for reading (std::ios::in) or writing (std::ios::out) or both.
bool Serializer::open(const std::string& filename,
                      std::ios_base::openmode openmode) {
  return sqlite_.open(filename, openmode);
}

LayerCake Serializer::loadLayerCake(MemoryType memory_type,
                                    const CudaStream cuda_stream) {
  // Get all the layers that are in here.
  std::vector<std::string> layer_names;
  getLayerNames(&layer_names);

  using TypeIndexAndLayerPtr =
      std::pair<std::type_index, std::unique_ptr<BaseLayer>>;

  std::vector<TypeIndexAndLayerPtr> layers;
  float voxel_size = 0.0f;
  for (const std::string& layer_name : layer_names) {
    // Get all of the functions we need.
    LayerSerializationFunctions layer_functions =
        LayerTypeRegister::getSerializationFunctions(layer_name);

    // Check if the functions that we need are set. Otherwise this layer cannot
    // be deserialized.
    if (layer_functions.construct_layer == nullptr ||
        layer_functions.add_data == nullptr) {
      LOG(WARNING) << "Do not have deserialization functions for " << layer_name
                   << ".";
      continue;
    }

    // Find the type_index for this layer.
    std::type_index type_index =
        LayerTypeRegister::getLayerTypeIndex(layer_name);

    LayerParameterStruct layer_params;
    getLayerParameters(layer_name, &layer_params);

    // Create the layer object.
    std::unique_ptr<BaseLayer> layer;
    layer = layer_functions.construct_layer(memory_type, layer_params);

    // Populate the layer with bloxxx.
    std::vector<Index3D> data_indices;
    getDataIndices(layer_name, &data_indices);

    for (const Index3D& index : data_indices) {
      std::vector<Byte> data;
      getDataAtIndex(layer_name, index, &data);
      layer_functions.add_data(index, data, layer.get(), cuda_stream);
    }

    // If the layer has a block size, then set the layer cake to the correct
    // setting.
    auto it = layer_params.float_params.find("block_size");
    if (it != layer_params.float_params.end()) {
      // Update the layer cake voxel size.
      // NOTE: we check that all loaded layers have the same voxel size.
      const float new_voxel_size =
          it->second / VoxelBlock<bool>::kVoxelsPerSide;
      CHECK(voxel_size == 0.0f || voxel_size == new_voxel_size);
      voxel_size = new_voxel_size;
    }

    // Finally put it into the layer cake.
    layers.push_back({type_index, std::move(layer)});
  }

  LayerCake cake(voxel_size);
  for (auto&& type_index_lay_pair : layers) {
    cake.insert(type_index_lay_pair.first,
                std::move(type_index_lay_pair.second));
  }

  return cake;
}

bool Serializer::writeLayerCake(const LayerCake& cake,
                                const CudaStream cuda_stream) {
  const std::unordered_map<std::type_index, std::unique_ptr<BaseLayer>>&
      layer_map = cake.get_layers();

  // For each layer, figure out the type, then serialize the parameters, then
  // finally the data.
  for (auto it = layer_map.begin(); it != layer_map.end(); it++) {
    // First get the name of the layer.
    std::string layer_name = LayerTypeRegister::getLayerName(it->first);
    if (layer_name.empty()) {
      LOG(ERROR) << "Unrecognized layer type, can't serialize: "
                 << it->first.name();
      continue;
    }

    // Get all of the functions we need.
    LayerSerializationFunctions layer_functions =
        LayerTypeRegister::getSerializationFunctions(layer_name);

    // Check if we know how to serialize this layer type.
    if (layer_functions.serialize_params == nullptr ||
        layer_functions.get_data_indices == nullptr ||
        layer_functions.serialize_data == nullptr) {
      continue;
    }

    // Create the layer tables.
    createLayerTables(layer_name);

    // Populate the layer metadata table.
    LayerParameterStruct param_struct =
        layer_functions.serialize_params(it->second.get());
    param_struct.string_params["type"] = layer_name;
    setLayerParameters(layer_name, param_struct);

    // Populate the blocks.
    // Get a list of all the blocks.
    std::vector<Index3D> data_indices =
        layer_functions.get_data_indices(it->second.get());

    // Batching these into a transaction is needed for performance reasons.
    sqlite_.runStatement("BEGIN TRANSACTION;");
    for (const Index3D& index : data_indices) {
      // Get the byte string for this data.
      std::vector<Byte> data_bytes =
          layer_functions.serialize_data(it->second.get(), index, cuda_stream);

      addLayerData(layer_name, index, data_bytes);
    }
    sqlite_.runStatement("END TRANSACTION;");
  }

  return true;
}

/// Close the file.
bool Serializer::close() { return sqlite_.close(); }

// --------- horrible SQL below this line ----------

// Get the name of the blocks layer table.
std::string Serializer::layerMetadataTableName(
    const std::string& layer_name) const {
  return layer_name + "_metadata";
}
// Get the name of the layer metadatable table.
std::string Serializer::layerDataTableName(
    const std::string& layer_name) const {
  return layer_name + "_data";
}

bool Serializer::createLayerTables(const std::string& layer_name) {
  // First the metadata table.
  std::string sql_meta_table = "CREATE TABLE " +
                               layerMetadataTableName(layer_name) +
                               "(param_name TEXT PRIMARY KEY UNIQUE NOT NULL,"
                               "value_string TEXT,"
                               "value_int INT,"
                               "value_float FLOAT);";

  std::string sql_blocks_table = "CREATE TABLE " +
                                 layerDataTableName(layer_name) +
                                 "(index_x INT NOT NULL,"
                                 "index_y INT NOT NULL,"
                                 "index_z INT NOT NULL,"
                                 "data BLOB,"
                                 "PRIMARY KEY(index_x, index_y, index_z));";

  bool retval = true;
  retval = sqlite_.runStatement(sql_meta_table);
  retval &= sqlite_.runStatement(sql_blocks_table);

  return retval;
}

// Serialize layer parameters.
bool Serializer::setLayerParameters(const std::string& layer_name,
                                    const LayerParameterStruct& layer_params) {
  for (auto kv : layer_params.string_params) {
    setLayerParameterString(layer_name, kv.first, kv.second);
  }
  for (auto kv : layer_params.int_params) {
    setLayerParameterInt(layer_name, kv.first, kv.second);
  }
  for (auto kv : layer_params.float_params) {
    setLayerParameterFloat(layer_name, kv.first, kv.second);
  }
  return true;
}

// Deserialize layer parameters.
bool Serializer::getLayerParameters(const std::string& layer_name,
                                    LayerParameterStruct* layer_params) {
  // Get all the parameter names.
  std::vector<std::string> string_param_names;
  getParameterNamesString(layer_name, &string_param_names);
  for (const std::string& param_name : string_param_names) {
    std::string param_value;
    getLayerParameterString(layer_name, param_name, &param_value);
    layer_params->string_params.emplace(param_name, param_value);
  }

  std::vector<std::string> int_param_names;
  getParameterNamesInt(layer_name, &int_param_names);
  for (const std::string& param_name : int_param_names) {
    int param_value;
    getLayerParameterInt(layer_name, param_name, &param_value);
    layer_params->int_params.emplace(param_name, param_value);
  }

  std::vector<std::string> float_param_names;
  getParameterNamesFloat(layer_name, &float_param_names);
  for (const std::string& param_name : float_param_names) {
    float param_value;
    getLayerParameterFloat(layer_name, param_name, &param_value);
    layer_params->float_params.emplace(param_name, param_value);
  }
  return true;
}

bool Serializer::setLayerParameterString(const std::string& layer_name,
                                         const std::string& param_name,
                                         const std::string& param_value) {
  std::string sql_statement = "INSERT INTO " +
                              layerMetadataTableName(layer_name) +
                              " (param_name, value_string) VALUES('" +
                              param_name + "','" + param_value + "');";
  return sqlite_.runStatement(sql_statement);
}

bool Serializer::setLayerParameterInt(const std::string& layer_name,
                                      const std::string& param_name,
                                      int param_value) {
  std::string sql_statement =
      "INSERT INTO " + layerMetadataTableName(layer_name) +
      " (param_name, value_int) VALUES ('" + param_name + "','" +
      std::to_string(param_value) + "');";
  return sqlite_.runStatement(sql_statement);
}

bool Serializer::setLayerParameterFloat(const std::string& layer_name,
                                        const std::string& param_name,
                                        float param_value) {
  std::string sql_statement =
      "INSERT INTO " + layerMetadataTableName(layer_name) +
      " (param_name, value_float) VALUES ('" + param_name + "','" +
      std::to_string(param_value) + "');";
  return sqlite_.runStatement(sql_statement);
}

bool Serializer::getParameterNamesString(
    const std::string& layer_name, std::vector<std::string>* param_names) {
  std::string sql_statement = "SELECT param_name FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE value_string IS NOT NULL;";
  return sqlite_.runMultipleQueryString(sql_statement, param_names);
}

bool Serializer::getParameterNamesInt(const std::string& layer_name,
                                      std::vector<std::string>* param_names) {
  std::string sql_statement = "SELECT param_name FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE value_int IS NOT NULL;";
  return sqlite_.runMultipleQueryString(sql_statement, param_names);
}

bool Serializer::getParameterNamesFloat(const std::string& layer_name,
                                        std::vector<std::string>* param_names) {
  std::string sql_statement = "SELECT param_name FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE value_float IS NOT NULL;";
  return sqlite_.runMultipleQueryString(sql_statement, param_names);
}

bool Serializer::getLayerParameterString(const std::string& layer_name,
                                         const std::string& param_name,
                                         std::string* param_value) {
  std::string sql_statement = "SELECT value_string FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE param_name = '" + param_name + "';";
  return sqlite_.runSingleQueryString(sql_statement, param_value);
}

bool Serializer::getLayerParameterInt(const std::string& layer_name,
                                      const std::string& param_name,
                                      int* param_value) {
  std::string sql_statement = "SELECT value_int FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE param_name = '" + param_name + "';";
  return sqlite_.runSingleQueryInt(sql_statement, param_value);
}

bool Serializer::getLayerParameterFloat(const std::string& layer_name,
                                        const std::string& param_name,
                                        float* param_value) {
  std::string sql_statement = "SELECT value_float FROM " +
                              layerMetadataTableName(layer_name) +
                              " WHERE param_name = '" + param_name + "';";
  return sqlite_.runSingleQueryFloat(sql_statement, param_value);
}

bool Serializer::addLayerData(const std::string& layer_name,
                              const Index3D& index,
                              const std::vector<Byte>& data) {
  std::string sql_statement = "INSERT INTO " + layerDataTableName(layer_name) +
                              " (index_x, index_y, index_z, data) VALUES (" +
                              std::to_string(index.x()) + "," +
                              std::to_string(index.y()) + "," +
                              std::to_string(index.z()) + ",?)";
  return sqlite_.runStatementWithBlob(sql_statement, data);
}

bool Serializer::getLayerNames(std::vector<std::string>* layer_names) {
  std::string suffix = "_metadata";
  std::string sql_statement =
      "SELECT name FROM sqlite_master WHERE "
      "type='table' AND name NOT LIKE 'sqlite_%' AND name LIKE '%" +
      suffix + "';";

  bool retval = sqlite_.runMultipleQueryString(sql_statement, layer_names);

  // Strip out the suffix.
  for (std::string& layer_name : *layer_names) {
    layer_name.replace(layer_name.find(suffix), suffix.length(), "");
  }

  return retval;
}

bool Serializer::getDataIndices(const std::string& layer_name,
                                std::vector<Index3D>* indices) {
  std::string sql_statement = "SELECT index_x,index_y,index_z FROM " +
                              layerDataTableName(layer_name) + ";";
  return sqlite_.runMultipleQueryIndex3D(sql_statement, indices);
}

bool Serializer::getDataAtIndex(const std::string& layer_name,
                                const Index3D& index, std::vector<Byte>* data) {
  std::string sql_statement = "SELECT data FROM " +
                              layerDataTableName(layer_name) +
                              " WHERE index_x=" + std::to_string(index.x()) +
                              " AND index_y=" + std::to_string(index.y()) +
                              " AND index_z=" + std::to_string(index.z()) + ";";

  return sqlite_.runSingleQueryBlob(sql_statement, data);
}

}  // namespace nvblox
