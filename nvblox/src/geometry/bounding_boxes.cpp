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
#include "nvblox/geometry/bounding_boxes.h"

namespace nvblox {

AxisAlignedBoundingBox getAABBOfBlocks(const float block_size,
                                       const std::vector<Index3D>& blocks) {
  AxisAlignedBoundingBox aabb;
  for (const Index3D& idx : blocks) {
    aabb = aabb.merged(getAABBOfBlock(block_size, idx));
  }
  return aabb;
}

AxisAlignedBoundingBox getAABBOfObservedVoxels(const EsdfLayer& layer) {
  AxisAlignedBoundingBox aabb;
  auto lambda = [&aabb, &layer](const Index3D& block_index,
                                const Index3D& voxel_index,
                                const EsdfVoxel* voxel) -> void {
    if (voxel->observed) {
      Vector3f p = getPositionFromBlockIndexAndVoxelIndex(
          layer.block_size(), block_index, voxel_index);
      aabb.extend(p);
      p += layer.block_size() * Vector3f::Ones();
      aabb.extend(p);
    }
  };
  callFunctionOnAllVoxels<EsdfVoxel>(layer, lambda);
  return aabb;
}

AxisAlignedBoundingBox getAABBOfObservedVoxels(const TsdfLayer& layer,
                                               const float min_weight) {
  AxisAlignedBoundingBox aabb;
  auto lambda = [&aabb, &layer, min_weight](const Index3D& block_index,
                                            const Index3D& voxel_index,
                                            const TsdfVoxel* voxel) -> void {
    if (voxel->weight > min_weight) {
      Vector3f p = getPositionFromBlockIndexAndVoxelIndex(
          layer.block_size(), block_index, voxel_index);
      aabb.extend(p);
      p += layer.block_size() * Vector3f::Ones();
      aabb.extend(p);
    }
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer, lambda);
  return aabb;
}

AxisAlignedBoundingBox getAABBOfObservedVoxels(const ColorLayer& layer,
                                               const float min_weight) {
  AxisAlignedBoundingBox aabb;
  auto lambda = [&aabb, &layer, min_weight](const Index3D& block_index,
                                            const Index3D& voxel_index,
                                            const ColorVoxel* voxel) -> void {
    if (voxel->weight > min_weight) {
      Vector3f p = getPositionFromBlockIndexAndVoxelIndex(
          layer.block_size(), block_index, voxel_index);
      aabb.extend(p);
      p += layer.block_size() * Vector3f::Ones();
      aabb.extend(p);
    }
  };
  callFunctionOnAllVoxels<ColorVoxel>(layer, lambda);
  return aabb;
}

}  // namespace nvblox