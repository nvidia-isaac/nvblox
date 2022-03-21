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

#include "nvblox/core/blox.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/unified_ptr.h"

#include "nvblox/experiments/load.h"

namespace nvblox {
namespace experiments {

struct UserDefinedBlock {
  typedef nvblox::unified_ptr<UserDefinedBlock> Ptr;
  typedef nvblox::unified_ptr<const UserDefinedBlock> ConstPtr;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<UserDefinedBlock>(memory_type);
  }

  float data = 0.0f;
};

using UserDefinedLayer = BlockLayer<UserDefinedBlock>;

// Custom load function for UserDefinedBlock
namespace io {

template <>
bool loadLayer<experiments::UserDefinedLayer>(
    const std::string& filename, experiments::UserDefinedLayer* layer);

}  // namespace io
}  // namespace experiments
}  // namespace nvblox