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
#include "nvblox/tests/projective_tsdf_integrator_cpu.h"

#include "nvblox/interpolation/interpolation_2d.h"

namespace nvblox {

template <typename SensorType>
void ProjectiveTsdfIntegratorCPU::updateBlocks(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const SensorType& camera,
    const float truncation_distance_m, TsdfLayer* layer) {
  CHECK(layer->memory_type() == MemoryType::kUnified)
      << "For CPU-based interpolation, the layer must be CPU accessible (ie "
         "MemoryType::kUnified).";

  // Update each block requested using the depth map.
  Transform T_C_L = T_L_C.inverse();
  for (const Index3D& block_index : block_indices) {
    VoxelBlock<TsdfVoxel>::Ptr block_ptr = layer->getBlockAtIndex(block_index);

    Index3D voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < TsdfBlock::kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < TsdfBlock::kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < TsdfBlock::kVoxelsPerSide;
             voxel_index.z()++) {
          Vector3f p_L = getCenterPositionFromBlockIndexAndVoxelIndex(
              layer->block_size(), block_index, voxel_index);

          // Convert the p_L to a p_C
          Vector3f p_C = T_C_L * p_L;
          Vector2f u_C;
          if (!camera.project(p_C, &u_C)) {
            // If the voxel isn't in the frame, next voxel plz
            continue;
          }

          float depth = -1.0;
          if (!interpolation::interpolate2DClosest(depth_frame, u_C, &depth)) {
            continue;
          }

          DCHECK_GE(depth, 0.0f);

          TsdfVoxel& voxel =
              block_ptr
                  ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

          // TODO: double-check this distance.
          const float voxel_distance = depth - p_C.z();

          // If we're behind the negative truncation distance, just continue.
          if (voxel_distance < -truncation_distance_m) {
            continue;
          }

          const float measurement_weight = 1.0f;
          const float fused_distance = (voxel_distance * measurement_weight +
                                        voxel.distance * voxel.weight) /
                                       (measurement_weight + voxel.weight);

          voxel.distance =
              fused_distance > 0.0f
                  ? std::min(truncation_distance_m, fused_distance)
                  : std::max(-truncation_distance_m, fused_distance);
          voxel.weight =
              std::min(measurement_weight + voxel.weight, max_weight_);
        }
      }
    }
  }
}

}  // namespace nvblox
