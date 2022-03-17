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

// Unique types
template <class... Types>
struct unique_types;

// Count occurances
template <class Tsearch, class... Types>
struct count_type_occurrence;

}  // namespace experiments
}  // namespace nvblox

#include "nvblox/experiments/impl/cake_common_impl.h"
