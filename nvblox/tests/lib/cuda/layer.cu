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
#include "nvblox/map/internal/cuda/impl/layer_impl.cuh"
#include "nvblox/map/layer.h"

#include "nvblox/tests/voxels.h"

namespace nvblox {

template void VoxelBlockLayer<FloatVoxel>::getVoxelsGPU(
    const device_vector<Vector3f>& positions_L,
    device_vector<FloatVoxel>* voxels_ptr,
    device_vector<bool>* success_flags_ptr) const;

}  // namespace nvblox
