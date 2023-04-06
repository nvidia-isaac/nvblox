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

#include "nvblox/map/common_names.h"
#include "nvblox/serialization/internal/block_serialization.h"
#include "nvblox/serialization/internal/layer_serialization.h"
#include "nvblox/serialization/internal/layer_type_register.h"

namespace nvblox {

/// Bind default functions for the given types.
template <typename LayerType>
LayerSerializationFunctions bindDefaultFunctions();

/// Registers all the common layer types for serialization. Must be called
/// before you serialize or de-serialize anything.
inline void registerCommonTypes();

}  // namespace nvblox

#include "nvblox/serialization/internal/impl/common_types_impl.h"
