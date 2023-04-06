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

namespace nvblox {
namespace interpolation {

template <typename VoxelType>
void interpolateOnCPU(const std::vector<Vector3f>& points_L,
                      const VoxelBlockLayer<VoxelType>& layer,
                      std::vector<float>* distances_ptr,
                      std::vector<bool>* success_flags_ptr) {
  CHECK_NOTNULL(distances_ptr);
  CHECK_NOTNULL(success_flags_ptr);
  CHECK(layer.memory_type() == MemoryType::kUnified)
      << "For CPU-based interpolation, the layer must be CPU accessible (ie "
         "MemoryType::kUnified).";
  distances_ptr->reserve(points_L.size());
  success_flags_ptr->reserve(points_L.size());
  for (const Vector3f& p_L : points_L) {
    float distance;
    success_flags_ptr->push_back(interpolateOnCPU(p_L, layer, &distance));
    distances_ptr->push_back(distance);
  }
}

namespace internal {

// Interpolate a layer's voxel's members using a member-access function
template <typename VoxelType>
using GetMemberFunctionType = std::function<float(const VoxelType& voxel)>;
template <typename VoxelType>
using VoxelValidFunctionType = std::function<bool(const VoxelType& voxel)>;

template <typename VoxelType>
bool interpolateMemberOnCPU(const Vector3f& p_L,
                            const VoxelBlockLayer<VoxelType>& layer,
                            GetMemberFunctionType<VoxelType> get_member,
                            VoxelValidFunctionType<VoxelType> is_valid,
                            float* result_ptr);

Eigen::Matrix<float, 8, 1> getQVector3D(const Vector3f& p_offset_in_voxels_L);

template <typename VoxelType>
bool getSurroundingVoxels3D(const Eigen::Vector3f p_L,
                            const VoxelBlockLayer<VoxelType>& layer,
                            VoxelValidFunctionType<VoxelType> is_valid,
                            std::array<VoxelType, 8>* voxels_ptr,
                            Vector3f* p_offset_in_voxels_L_ptr = nullptr) {
  CHECK_NOTNULL(voxels_ptr);
  // Get the low-side voxel: ie the voxel whose midpoint lies on the low side of
  // the query point
  // NOTE(alexmillane): We implement this by finding the block
  // and voxel coordinates of a point half a voxel-width lower than the query
  // point.
  const float voxel_size = layer.voxel_size();
  const float half_voxel_size = voxel_size * 0.5f;
  Index3D low_block_idx;
  Index3D low_voxel_idx;
  getBlockAndVoxelIndexFromPositionInLayer(layer.block_size(),
                                           p_L.array() - half_voxel_size,
                                           &low_block_idx, &low_voxel_idx);

  // Get offset between bottom-left-corner voxel and the point
  if (p_offset_in_voxels_L_ptr != nullptr) {
    const Vector3f p_corner_L =
        getPositionFromBlockIndexAndVoxelIndex(layer.block_size(),
                                               low_block_idx, low_voxel_idx) +
        half_voxel_size * Vector3f::Ones();
    *p_offset_in_voxels_L_ptr = (p_L - p_corner_L) / voxel_size;
  }

  // Get the 8 voxels surrounding this point
  int linear_idx = 0;
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        Index3D voxel_idx = low_voxel_idx + Index3D(x, y, z);
        Index3D block_idx = low_block_idx;

        // Move into a neighbouring block(s) if required
        if ((voxel_idx.array() == VoxelBlock<bool>::kVoxelsPerSide).any()) {
          // Increment block index in dimensions which have hit the limits
          const Eigen::Array<bool, 3, 1> limits_hit_flags =
              (voxel_idx.array() == VoxelBlock<bool>::kVoxelsPerSide);
          block_idx += limits_hit_flags.matrix().cast<int>();
          // Reset voxel index to zero in dimensions in which the block idx was
          // incremented
          voxel_idx = limits_hit_flags.select(0, voxel_idx);
        }

        // Get the voxel
        const typename VoxelBlock<VoxelType>::ConstPtr block_ptr =
            layer.getBlockAtIndex(block_idx);
        if (block_ptr == nullptr) {
          return false;
        }
        const VoxelType& voxel =
            block_ptr->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];

        // Check voxel valid
        if (!is_valid(voxel)) {
          return false;
        }
        // Write
        (*voxels_ptr)[linear_idx] = voxel;
        ++linear_idx;
      }
    }
  }
  return true;
}

template <typename VoxelType>
bool interpolateMemberOnCPU(const Vector3f& p_L,
                            const VoxelBlockLayer<VoxelType>& layer,
                            GetMemberFunctionType<VoxelType> get_member,
                            VoxelValidFunctionType<VoxelType> is_valid,
                            float* result_ptr) {
  std::array<VoxelType, 8> voxels;

  Eigen::Vector3f p_offset_L;
  if (!internal::getSurroundingVoxels3D(p_L, layer, is_valid, &voxels,
                                        &p_offset_L)) {
    return false;
  }

  // Express interpolation as matrix mulitplication from paper:
  // (http://spie.org/samples/PM159.pdf)
  Eigen::Matrix<float, 8, 1> member_vector;
  for (int i = 0; i < 8; i++) {
    member_vector[i] = get_member(voxels[i]);
  }
  const Eigen::Matrix<float, 1, 8> q_vector =
      internal::getQVector3D(p_offset_L);
  // clang-format off
  static const Eigen::Matrix<float,8,8> interpolation_table =
      (Eigen::Matrix<float,8,8>() <<
        1,  0,  0,  0,  0,  0,  0,  0,
       -1,  0,  0,  0,  1,  0,  0,  0,
       -1,  0,  1,  0,  0,  0,  0,  0,
       -1,  1,  0,  0,  0,  0,  0,  0,
        1,  0, -1,  0, -1,  0,  1,  0,
        1, -1, -1,  1,  0,  0,  0,  0,
        1, -1,  0,  0, -1,  1,  0,  0,
       -1,  1,  1, -1,  1, -1, -1,  1
       )
          .finished();
  // clang-format on
  *result_ptr = q_vector * (interpolation_table * member_vector);
  return true;
}

}  // namespace internal
}  // namespace interpolation
}  // namespace nvblox
