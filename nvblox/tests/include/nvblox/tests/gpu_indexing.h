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
#include "nvblox/core/types.h"

namespace nvblox {
namespace test_utils {

void getBlockAndVoxelIndexFromPositionInLayerOnGPU(const float block_size,
                                                   const Vector3f& position,
                                                   Index3D* block_idx,
                                                   Index3D* voxel_idx);

void getBlockAndVoxelIndexFromPositionInLayerOnGPU(
    const float block_size, const std::vector<Vector3f>& positions,
    std::vector<Index3D>* block_indices, std::vector<Index3D>* voxel_indices);

}  // namespace test_utils
}  // namespace nvblox