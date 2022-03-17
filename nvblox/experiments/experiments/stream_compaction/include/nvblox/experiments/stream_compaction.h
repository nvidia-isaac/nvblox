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
#include "nvblox/core/unified_vector.h"

#include "nvblox/experiments/cached_allocator.h"

namespace nvblox {
namespace experiments {

class StreamCompactor {
 public:
  StreamCompactor() {}

  int streamCompactionOnGPU(const device_vector<int>& data,
                            const device_vector<bool>& stencil,
                            device_vector<int>* compact_data_ptr,
                            bool use_cached_allocator = true);

 private:
  CachedAllocator cached_allocator_;
};

}  // namespace experiments
}  // namespace nvblox
