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
#include "nvblox/core/internal/error_check.h"

#include <iostream>

namespace nvblox {

void check_cuda(cudaError_t result, char const* const func,
                const char* const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func
              << "'. Error string: " << cudaGetErrorString(result) << ".\n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

void check_npp(NppStatus result, char const* const func, const char* const file,
               int const line) {
  if (result) {
    std::cerr << "NPP error = " << result << " at " << file << ":" << line
              << " '" << func << ".\n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

}  // namespace nvblox
