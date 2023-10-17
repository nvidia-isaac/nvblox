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

#include "nvblox/core/indexing.h"
#include "nvblox/core/log_odds.h"
#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/voxels.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
namespace primitives {

template <>
inline void Scene::setVoxel(float value, TsdfVoxel* voxel) const {
  voxel->distance = value;
  voxel->weight = 1.0f;  // Just to make sure it gets visualized/meshed/etc.
}

template <>
inline void Scene::setVoxel(float value, OccupancyVoxel* voxel) const {
  voxel->log_odds = logOddsFromProbability(value);
}

template <>
inline void Scene::setVoxel(float value, FreespaceVoxel* voxel) const {
  // We assume the value is 1.0f if an object is inside the voxel and 0.0f if
  // an object is outside the voxel (see getVoxelGroundTruthValue).
  voxel->is_high_confidence_freespace = std::abs(value) <= 1e-4;
}

template <>
inline float Scene::getVoxelGroundTruthValue<TsdfVoxel>(
    const Vector3f& position, float max_dist, float) const {
  // Iterate over all objects and get distances to this thing.
  // Only computes up to max_distance away from the voxel (to reduce amount
  // of ray casting).
  float distance = getSignedDistanceToPoint(position, max_dist);
  // Also truncate the distance *inside* the obstacle, which is not truncated
  // by the previous function.
  return std::max(distance, -max_dist);
}

template <typename VoxelType>
inline float Scene::getVoxelGroundTruthValue(const Vector3f& position,
                                             float max_dist,
                                             float voxel_size) const {
  // This template function is used both for Freespace and Tsdf voxels.
  const float min_distance_to_object =
      getSignedDistanceToPoint(position, max_dist);
  const float voxel_body_diagonal = sqrt(3.0) * voxel_size;
  // Note: Using the body diagonal to check whether the object lies
  // inside the voxel is an approximation.
  if (min_distance_to_object <= (voxel_body_diagonal / 2.0)) {
    return 1.0f;  // object inside the voxel
  } else {
    return 0.0f;  // object outside voxel
  }
}

template <typename VoxelType>
void Scene::generateLayerFromScene(float max_dist,
                                   VoxelBlockLayer<VoxelType>* layer) const {
  CHECK(layer->memory_type() != MemoryType::kDevice)
      << "For scene generation the layer must be CPU accessible "
         "(MemoryType::kUnified or MemoryType::kHost).";

  CHECK_NOTNULL(layer);

  const float block_size = layer->block_size();
  const float voxel_size = layer->voxel_size();

  // First allocate all the blocks within the AABB.
  std::vector<Index3D> block_indices =
      getBlockIndicesTouchedByBoundingBox(block_size, aabb_);

  for (const Index3D& block_index : block_indices) {
    layer->allocateBlockAtIndex(block_index);
  }

  // Iterate over every voxel in the layer and compute its distance to all
  // objects.
  auto lambda = [&block_size, &max_dist, &voxel_size, this](
                    const Index3D& block_index, const Index3D& voxel_index,
                    VoxelType* voxel) {
    const Vector3f position = getCenterPositionFromBlockIndexAndVoxelIndex(
        block_size, block_index, voxel_index);

    if (!aabb_.contains(position)) {
      return;
    }

    // Get the ground truth value for this voxel and update it
    float gt_value =
        getVoxelGroundTruthValue<VoxelType>(position, max_dist, voxel_size);
    setVoxel<VoxelType>(gt_value, voxel);
  };

  // Call above lambda on every voxel in the layer.
  callFunctionOnAllVoxels<VoxelType>(layer, lambda);
}

}  // namespace primitives
}  // namespace nvblox
