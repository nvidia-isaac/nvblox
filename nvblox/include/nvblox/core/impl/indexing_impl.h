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

#include <glog/logging.h>

#include "nvblox/core/blox.h"

namespace nvblox {

// TODO: this is badly done now for non-Eigen types.
/// Input position: metric (world) units with respect to the layer origin.
/// Output index: voxel index relative to the block origin.
Index3D getVoxelIndexFromPositionInLayer(const float block_size,
                                         const Vector3f& position) {
  constexpr float kVoxelsPerSideInv = 1.0f / VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = block_size * kVoxelsPerSideInv;

  Index3D global_voxel_index =
      (position / voxel_size).array().floor().cast<int>();
  Index3D voxel_index = global_voxel_index.unaryExpr([&](int x) {
    return static_cast<int>(x % VoxelBlock<bool>::kVoxelsPerSide);
  });

  // Double-check that we get reasonable indices out.
  DCHECK((voxel_index.array() >= 0).all() &&
         (voxel_index.array() < VoxelBlock<bool>::kVoxelsPerSide).all());

  return voxel_index;
}

Index3D getBlockIndexFromPositionInLayer(const float block_size,
                                         const Vector3f& position) {
  Eigen::Vector3i index = (position / block_size).array().floor().cast<int>();
  return Index3D(index.x(), index.y(), index.z());
}

void getBlockAndVoxelIndexFromPositionInLayer(const float block_size,
                                              const Vector3f& position,
                                              Index3D* block_idx,
                                              Index3D* voxel_idx) {
  constexpr int kVoxelsPerSideMinusOne = VoxelBlock<bool>::kVoxelsPerSide - 1;
  constexpr float kVoxelsPerSideInv = 1.0f / VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size_inv = 1.0 / (block_size * kVoxelsPerSideInv);
  *block_idx = (position / block_size).array().floor().cast<int>();
  *voxel_idx =
      ((position - block_size * block_idx->cast<float>()) * voxel_size_inv)
          .array()
          .cast<int>()
          .min(kVoxelsPerSideMinusOne);
}

Vector3f getPositionFromBlockIndexAndVoxelIndex(const float block_size,
                                                const Index3D& block_index,
                                                const Index3D& voxel_index) {
  constexpr float kVoxelsPerSideInv = 1.0f / VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = block_size * kVoxelsPerSideInv;

  return Vector3f(block_size * block_index.cast<float>() +
                  voxel_size * voxel_index.cast<float>());
}

Vector3f getPositionFromBlockIndex(const float block_size,
                                   const Index3D& block_index) {
  // This is pretty trivial, huh.
  return Vector3f(block_size * block_index.cast<float>());
}

Vector3f getCenterPostionFromBlockIndexAndVoxelIndex(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index) {
  constexpr float kHalfVoxelsPerSideInv =
      0.5f / VoxelBlock<bool>::kVoxelsPerSide;
  const float half_voxel_size = block_size * kHalfVoxelsPerSideInv;

  return getPositionFromBlockIndexAndVoxelIndex(block_size, block_index,
                                                voxel_index) +
         Vector3f(half_voxel_size, half_voxel_size, half_voxel_size);
}

}  // namespace nvblox
