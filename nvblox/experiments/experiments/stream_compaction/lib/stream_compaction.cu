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
#include "nvblox/experiments/stream_compaction.h"

#include <thrust/copy.h>

#include "nvblox/utils/timing.h"

namespace nvblox {
namespace experiments {

int StreamCompactor::streamCompactionOnGPU(const device_vector<int>& data,
                                           const device_vector<bool>& stencil,
                                           device_vector<int>* compact_data_ptr,
                                           bool use_cached_allocator) {
  int* output_end_it;
  if (!use_cached_allocator) {
    output_end_it = thrust::copy_if(thrust::device,             // NOLINT
                                    data.data(),                // NOLINT
                                    data.data() + data.size(),  // NOLINT
                                    stencil.data(),             // NOLINT
                                    compact_data_ptr->data(),   // NOLINT
                                    thrust::identity<bool>());

  } else {
    output_end_it =
        thrust::copy_if(thrust::cuda::par(cached_allocator_),  // NOLINT
                        data.data(),                           // NOLINT
                        data.data() + data.size(),             // NOLINT
                        stencil.data(),                        // NOLINT
                        compact_data_ptr->data(),              // NOLINT
                        thrust::identity<bool>());
  }

  const int num_true = output_end_it - data.data();
  return num_true;
}

}  // namespace experiments
}  // namespace nvblox
