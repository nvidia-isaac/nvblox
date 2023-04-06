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

#include "nvblox/geometry/bounding_spheres.h"

namespace nvblox {
namespace internal {

std::vector<Index3D> getBlocksWithRadius(
    const std::vector<Index3D>& input_blocks, float block_size,
    const Vector3f& center, float radius,
    std::function<bool(float, float)> comparison) {
  // Go through all the blocks, get their distances to the center.
  std::vector<Index3D> output_blocks;
  for (const Index3D& block_index : input_blocks) {
    AxisAlignedBoundingBox box = getAABBOfBlock(block_size, block_index);
    if (comparison(box.exteriorDistance(center), radius)) {
      output_blocks.push_back(block_index);
    }
  }
  return output_blocks;
}

}  // namespace internal

std::vector<Index3D> getBlocksWithinRadius(
    const std::vector<Index3D>& input_blocks, float block_size,
    const Vector3f& center, float radius) {
  return internal::getBlocksWithRadius(input_blocks, block_size, center, radius,
                                       std::less<float>());
}

std::vector<Index3D> getBlocksOutsideRadius(
    const std::vector<Index3D>& input_blocks, float block_size,
    const Vector3f& center, float radius) {
  return internal::getBlocksWithRadius(input_blocks, block_size, center, radius,
                                       std::greater<float>());
}

std::vector<Index3D> getBlocksWithinRadiusOfAABB(
    const std::vector<Index3D>& input_blocks, float block_size,
    const AxisAlignedBoundingBox& aabb, float radius) {
  // Go through all the blocks, get their distances to the center.
  std::vector<Index3D> output_blocks;
  for (const Index3D& block_index : input_blocks) {
    // Check if it's close to the AABB.
    AxisAlignedBoundingBox box = getAABBOfBlock(block_size, block_index);
    if (box.exteriorDistance(aabb) > radius) {
      continue;
    }

    // Add to the output blocks.
    output_blocks.push_back(block_index);
  }
  return output_blocks;
}

}  // namespace nvblox