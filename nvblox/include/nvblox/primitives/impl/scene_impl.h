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

---- Original voxblox license, which this file is heavily based on: ----
Copyright (c) 2016, ETHZ ASL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of voxblox nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "nvblox/core/accessors.h"
#include "nvblox/core/bounding_boxes.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
namespace primitives {

template <>
inline void Scene::setVoxel(float dist, TsdfVoxel* voxel) const {
  voxel->distance = dist;
  voxel->weight = 1.0f;  // Just to make sure it gets visualized/meshed/etc.
}

template <typename VoxelType>
void Scene::generateSdfFromScene(float max_dist,
                                 VoxelBlockLayer<VoxelType>* layer) const {
  CHECK(layer->memory_type() == MemoryType::kUnified)
      << "For scene generation the layer must be CPU accessible "
         "(MemoryType::kUnified).";
  timing::Timer sim_timer("primitives/generate_sdf");

  CHECK_NOTNULL(layer);

  float block_size = layer->block_size();

  // First allocate all the blocks within the AABB.
  std::vector<Index3D> block_indices =
      getBlockIndicesTouchedByBoundingBox(block_size, aabb_);

  for (const Index3D& block_index : block_indices) {
    layer->allocateBlockAtIndex(block_index);
  }

  // Iterate over every voxel in the layer and compute its distance to all
  // objects.
  auto lambda = [&block_size, &max_dist, this](const Index3D& block_index,
                                               const Index3D& voxel_index,
                                               VoxelType* voxel) {
    const Vector3f position = getCenterPostionFromBlockIndexAndVoxelIndex(
        block_size, block_index, voxel_index);

    if (!aabb_.contains(position)) {
      return;
    }

    // Iterate over all objects and get distances to this thing.
    // Only computes up to max_distance away from the voxel (to reduce amount
    // of ray casting).
    float voxel_dist = getSignedDistanceToPoint(position, max_dist);

    // Then update the thing.
    // Also truncate the distance *inside* the obstacle, which is not truncated
    // by the previous function.
    voxel_dist = std::max(voxel_dist, -max_dist);
    setVoxel<VoxelType>(voxel_dist, voxel);
  };

  // Call above lambda on every voxel in the layer.
  callFunctionOnAllVoxels<VoxelType>(layer, lambda);
}

}  // namespace primitives
}  // namespace nvblox