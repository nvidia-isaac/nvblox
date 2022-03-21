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
#include "nvblox/experiments/cached_allocator.h"

#include <thrust/system/cuda/vector.h>

#include <glog/logging.h>

namespace nvblox {
namespace experiments {

CachedAllocator::~CachedAllocator() { free(); }

char* CachedAllocator::allocate(std::ptrdiff_t num_bytes) {
  CHECK(!in_use_) << "This allocator is only for single consumers.";

  // If there's enough memory,
  if (num_bytes <= num_bytes_allocated_) {
    // don't allocate anything and return the block we have.
  } else {
    try {
      // Free old block
      if (memory_ != nullptr) {
        thrust::cuda::free(thrust::cuda::pointer<char>(memory_));
      }

      // allocate memory and convert cuda::pointer to raw pointer
      memory_ = thrust::cuda::malloc<char>(num_bytes).get();
      num_bytes_allocated_ = num_bytes;
    } catch (std::runtime_error& e) {
      throw;
    }
  }

  in_use_ = true;
  return memory_;
}

void CachedAllocator::deallocate(char* ptr, size_t n) {
  in_use_ = false;
}

void CachedAllocator::free() {
  thrust::cuda::free(thrust::cuda::pointer<char>(memory_));
  num_bytes_allocated_ = 0;
  in_use_ = false;
  memory_ = nullptr;
}

}  // namespace experiments
}  // namespace nvblox
