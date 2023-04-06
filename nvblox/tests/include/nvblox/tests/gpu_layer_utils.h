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
#pragma once

#include <vector>

#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/map/common_names.h"

namespace nvblox {
namespace test_utils {

// Returns a vector of flags indicating if the query indices exist.
std::vector<bool> getContainsFlags(const GPULayerView<TsdfBlock>& gpu_layer,
                                   const std::vector<Index3D>& indices);

// Retrieves Voxels on the GPU and brings them back to CPU.
std::pair<std::vector<TsdfVoxel>, std::vector<bool>> getVoxelsAtPositionsOnGPU(
    const GPULayerView<TsdfBlock>& gpu_layer,
    const std::vector<Vector3f>& p_L_vec);

}  // namespace test_utils
}  // namespace nvblox
