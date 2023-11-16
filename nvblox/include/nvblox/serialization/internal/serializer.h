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

#include <ios>
#include <vector>

#include "nvblox/map/layer_cake.h"
#include "nvblox/serialization/internal/common_types.h"
#include "nvblox/serialization/internal/layer_type_register.h"
#include "nvblox/serialization/internal/sqlite_database.h"

namespace nvblox {

/// Class to serialize and read a layer cake from an SQLite database.
class Serializer {
 public:
  /// Default constructor for invalid file. Must call open before using.
  Serializer();
  virtual ~Serializer() = default;

  /// Open a particular file.
  Serializer(const std::string& filename, std::ios_base::openmode openmode);

  /// Is the current file valid? This means that we have successfully opened
  /// a valid database file to write to or read from.
  bool valid() const;

  /// Open a file for reading (std::ios::in) or writing (std::ios::out) or both.
  /// std::ios::out & std::ios::trunc will cause the file to be truncated.
  bool open(const std::string& filename,
            std::ios_base::openmode openmode = std::ios::in);

  /// Load a layer cake from the opened file of a given memory type.
  LayerCake loadLayerCake(MemoryType memory_type, const CudaStream cuda_stream);

  /// Write out a layer cake to the opened file, return success.
  bool writeLayerCake(const LayerCake& cake, const CudaStream cuda_stream);

  /// Close the file.
  bool close();

  // ======= Below use only if you know WTF you're doing! =====================

  /// Create a layer table & metadata table.
  bool createLayerTables(const std::string& layer_name);

  /// Serialize layer parameters.
  bool setLayerParameters(const std::string& layer_name,
                          const LayerParameterStruct& layer_params);

  /// Deserialize layer parameters.
  bool getLayerParameters(const std::string& layer_name,
                          LayerParameterStruct* layer_params);

  /// Write a new data parameter to the table.
  bool addLayerData(const std::string& layer_name, const Index3D& index,
                    const std::vector<Byte>& data);

 private:
  // Set layer parameters in the database.
  bool setLayerParameterString(const std::string& layer_name,
                               const std::string& param_name,
                               const std::string& param_value);
  bool setLayerParameterInt(const std::string& layer_name,
                            const std::string& param_name, int param_value);
  bool setLayerParameterFloat(const std::string& layer_name,
                              const std::string& param_name, float param_value);

  // Get the names of all the paramters from the database.
  bool getParameterNamesString(const std::string& layer_name,
                               std::vector<std::string>* param_names);
  bool getParameterNamesInt(const std::string& layer_name,
                            std::vector<std::string>* param_names);
  bool getParameterNamesFloat(const std::string& layer_name,
                              std::vector<std::string>* param_names);

  // Get layer parameters from the database.
  bool getLayerParameterString(const std::string& layer_name,
                               const std::string& param_name,
                               std::string* param_value);
  bool getLayerParameterInt(const std::string& layer_name,
                            const std::string& param_name, int* param_value);
  bool getLayerParameterFloat(const std::string& layer_name,
                              const std::string& param_name,
                              float* param_value);

  bool getDataAtIndex(const std::string& layer_name, const Index3D& index,
                      std::vector<Byte>* data);

  bool getLayerNames(std::vector<std::string>* layer_names);
  bool getDataIndices(const std::string& layer_name,
                      std::vector<Index3D>* indices);

  // Get the name of the data layer table.
  std::string layerDataTableName(const std::string& layer_name) const;
  // Get the name of the layer metadatable table.
  std::string layerMetadataTableName(const std::string& layer_name) const;

  SqliteDatabase sqlite_;
};

}  // namespace nvblox
