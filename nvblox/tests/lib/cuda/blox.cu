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
#include "nvblox/tests/blox.h"

#include "nvblox/gpu_hash/internal/cuda/impl/gpu_layer_view_impl.cuh"

namespace nvblox {

template class GPULayerView<IndexBlock>;
template class GPULayerView<BooleanBlock>;
template class GPULayerView<FloatVoxelBlock>;
template class GPULayerView<InitializationTestVoxelBlock>;

}  // namespace nvblox
