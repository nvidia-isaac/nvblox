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
#include "nvblox/gpu_hash/internal/cuda/gpu_set.cuh"

namespace nvblox {

Index3DDeviceSet::Index3DDeviceSet(size_t size) {
  set = Index3DDeviceSetType::createDeviceObject(static_cast<int64_t>(size));
}
Index3DDeviceSet::~Index3DDeviceSet() {
  Index3DDeviceSetType::destroyDeviceObject(set);
}

void Index3DDeviceSet::resize(size_t size) {
  if (set.size() >= static_cast<int32_t>(size)) {
    clear();
  } else {
    Index3DDeviceSetType::destroyDeviceObject(set);
    set = Index3DDeviceSetType::createDeviceObject(static_cast<int64_t>(size));
  }
}

void Index3DDeviceSet::clear() { set.clear(); }

void copySetToVector(const Index3DDeviceSetType& set,
                     std::vector<Index3D>* vec) {
  vec->resize(set.size());
  auto set_iter = set.device_range();
  thrust::copy_n(set_iter.begin(), set_iter.size(), vec->begin());
}

void copySetToDeviceVectorAsync(const Index3DDeviceSetType& set,
                                device_vector<Index3D>* vec,
                                const CudaStream cuda_stream) {
  vec->resizeAsync(set.size(), cuda_stream);
  auto set_iter = set.device_range();
  thrust::copy_n(thrust::device, set_iter.begin(), set_iter.size(),
                 vec->begin());
}

}  // namespace nvblox