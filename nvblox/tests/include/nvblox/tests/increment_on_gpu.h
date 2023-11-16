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

#include "nvblox/core/cuda_stream.h"

namespace nvblox {
namespace test_utils {

void incrementOnGPU(int* number_ptr);
void incrementOnStream(int* number_ptr, CudaStream* cuda_stream_ptr);

void incrementOnGPU(const int num_elelments, int* numbers_ptr);

}  // namespace test_utils
}  // namespace nvblox