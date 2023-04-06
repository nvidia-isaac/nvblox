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
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"

namespace nvblox {
namespace test_utils {

void fillVectorWithConstant(float value, unified_vector<float>* vec_ptr);
void fillVectorWithConstant(int value, unified_vector<int>* vec_ptr);

void fillWithConstant(float value, size_t num_elems, float* vec_ptr);
void fillWithConstant(int value, size_t num_elems, int* vec_ptr);

bool checkVectorAllConstant(const unified_vector<float>& vec_ptr, float value);
bool checkVectorAllConstant(const unified_vector<int>& vec_ptr, int value);

bool checkAllConstant(const float* vec_ptr, float value, size_t num_elems);
bool checkAllConstant(const int* vec_ptr, int value, size_t num_elems);

void addOneToAllGPU(unified_vector<int>* vec_ptr);

}  // namespace test_utils
}  // namespace nvblox
