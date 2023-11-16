/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

using TsdfBlock = VoxelBlock<TsdfVoxel>;
using TsdfLayer = VoxelBlockLayer<TsdfVoxel>;
using FreespaceBlock = VoxelBlock<FreespaceVoxel>;
using FreespaceLayer = VoxelBlockLayer<FreespaceVoxel>;
using OccupancyBlock = VoxelBlock<OccupancyVoxel>;
using OccupancyLayer = VoxelBlockLayer<OccupancyVoxel>;
using EsdfBlock = VoxelBlock<EsdfVoxel>;
using EsdfLayer = VoxelBlockLayer<EsdfVoxel>;
using ColorBlock = VoxelBlock<ColorVoxel>;
using ColorLayer = VoxelBlockLayer<ColorVoxel>;
using MeshLayer = BlockLayer<MeshBlock>;

}  // namespace nvblox
