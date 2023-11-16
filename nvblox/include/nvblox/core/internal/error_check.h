/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include <cuda_runtime.h>
#include <npp.h>

namespace nvblox {

void check_cuda(cudaError_t result, char const* const func,
                const char* const file, int const line);

#define checkCudaErrors(val) nvblox::check_cuda((val), #val, __FILE__, __LINE__)

void check_npp(NppStatus status, char const* const func, const char* const file,
               int const line);

#define checkNppErrors(val) nvblox::check_npp((val), #val, __FILE__, __LINE__)

}  // namespace nvblox
