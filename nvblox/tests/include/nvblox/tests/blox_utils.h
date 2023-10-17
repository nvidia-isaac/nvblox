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

#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/tests/blox.h"

namespace nvblox {
namespace test_utils {

// Fills a TsdfBlock such that the voxels distance and weight values are their
// linear index (as a float)
void setTsdfBlockVoxelsInSequence(TsdfBlock::Ptr block);
void setFloatingBlockVoxelsInSequence(FloatVoxelBlock::Ptr block);
void setTsdfBlockVoxelsConstant(const float distance, TsdfBlock::Ptr block);

bool checkBlockAllConstant(const TsdfBlock::Ptr block, TsdfVoxel voxel_cpu);
bool checkBlockAllConstant(const InitializationTestVoxelBlock::Ptr block,
                           InitializationTestVoxel voxel_cpu);
bool checkBlockAllConstant(const ColorBlock::Ptr block, ColorVoxel voxel_cpu);

}  // namespace test_utils
}  // namespace nvblox
