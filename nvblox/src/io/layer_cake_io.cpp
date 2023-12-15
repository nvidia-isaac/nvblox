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
#include "nvblox/utils/logging.h"

#include "nvblox/serialization/internal/serializer.h"

namespace nvblox {
namespace io {

bool writeLayerCakeToFile(const std::string& filename, const LayerCake& cake,
                          const CudaStream cuda_stream) {
  registerCommonTypes();

  // Truncate and overwrite by default.
  Serializer serializer(filename, std::ios::out | std::ios::trunc);

  if (!serializer.valid()) {
    LOG(ERROR) << "Could not open file for writing: " << filename;
    return false;
  }

  bool status = serializer.writeLayerCake(cake, cuda_stream);
  serializer.close();
  return status;
}

LayerCake loadLayerCakeFromFile(const std::string& filename,
                                MemoryType memory_type) {
  registerCommonTypes();

  Serializer serializer(filename, std::ios::in);

  if (!serializer.valid()) {
    LOG(ERROR) << "Could not open file for reading: " << filename;
    return LayerCake();
  }

  LayerCake cake = serializer.loadLayerCake(memory_type, CudaStreamOwning());
  serializer.close();
  return cake;
}

}  // namespace io
}  // namespace nvblox
