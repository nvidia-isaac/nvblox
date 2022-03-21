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
#include "nvblox/integrators/frustum.h"

#include "nvblox/core/bounding_boxes.h"

namespace nvblox {

std::vector<Index3D> FrustumCalculator::getBlocksInView(
    const Transform& T_L_C, const Camera& camera, const float block_size,
    const float max_distance) {
  CHECK_GT(max_distance, 0.0f);

  // View frustum
  const Frustum frustum = camera.getViewFrustum(T_L_C, 0.0f, max_distance);

  // Coarse bound: AABB
  const AxisAlignedBoundingBox aabb_L = frustum.getAABB();
  std::vector<Index3D> block_indices_in_aabb =
      getBlockIndicesTouchedByBoundingBox(block_size, aabb_L);

  // Tight bound: View frustum
  std::vector<Index3D> block_indices_in_frustum;
  for (const Index3D& block_index : block_indices_in_aabb) {
    const AxisAlignedBoundingBox& aabb_block =
        getAABBOfBlock(block_size, block_index);
    if (frustum.isAABBInView(aabb_block)) {
      block_indices_in_frustum.push_back(block_index);
    }
  }
  return block_indices_in_frustum;
}

std::vector<Index3D> FrustumCalculator::getBlocksInImageView(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float block_size, const float truncation_distance_m,
    const float max_integration_distance_m) {
  float min_depth, max_depth;
  std::tie(min_depth, max_depth) = image::minmaxGPU(depth_frame);
  float max_depth_plus_trunc = max_depth + truncation_distance_m;
  if (max_integration_distance_m > 0.0f) {
    max_depth_plus_trunc =
        std::min<float>(max_depth_plus_trunc, max_integration_distance_m);
  }
  return getBlocksInView(T_L_C, camera, block_size, max_depth_plus_trunc);
}

}  // namespace nvblox