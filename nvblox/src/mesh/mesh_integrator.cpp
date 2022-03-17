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
#include "nvblox/utils/timing.h"

#include "nvblox/mesh/mesh_integrator.h"

namespace nvblox {

MeshIntegrator::MeshIntegrator() {
  // clang-format off
    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0,
                           0, 0, 1, 1, 0, 0, 1, 1,
                           0, 0, 0, 0, 1, 1, 1, 1;
  // clang-format on
}

bool MeshIntegrator::integrateMeshFromDistanceField(
    const TsdfLayer& distance_layer, BlockLayer<MeshBlock>* mesh_layer,
    const DeviceType device_type) {
  // First, get all the blocks.
  std::vector<Index3D> block_indices = distance_layer.getAllBlockIndices();
  if (device_type == DeviceType::kCPU) {
    return integrateBlocksCPU(distance_layer, block_indices, mesh_layer);
  } else {
    return integrateBlocksGPU(distance_layer, block_indices, mesh_layer);
  }
}

bool MeshIntegrator::integrateBlocksCPU(
    const TsdfLayer& distance_layer, const std::vector<Index3D>& block_indices,
    BlockLayer<MeshBlock>* mesh_layer) {
  timing::Timer mesh_timer("mesh/integrate");
  CHECK_NOTNULL(mesh_layer);
  CHECK_NEAR(distance_layer.block_size(), mesh_layer->block_size(), 1e-4);

  // Figure out which of these actually contain something worth meshing.
  const float block_size = distance_layer.block_size();
  const float voxel_size = distance_layer.voxel_size();

  // For each block...
  for (const Index3D& block_index : block_indices) {
    // Get the block.
    VoxelBlock<TsdfVoxel>::ConstPtr block =
        distance_layer.getBlockAtIndex(block_index);

    // Check meshability - basically if this contains anything near the
    // border.
    if (!isBlockMeshable(block, voxel_size * 2)) {
      continue;
    }

    // Get all the neighbor blocks.
    std::vector<VoxelBlock<TsdfVoxel>::ConstPtr> neighbor_blocks(8);
    for (int i = 0; i < 8; i++) {
      Index3D neighbor_index =
          block_index + marching_cubes::directionFromNeighborIndex(i);
      neighbor_blocks[i] = distance_layer.getBlockAtIndex(neighbor_index);
    }

    // Get all the potential triangles:
    std::vector<marching_cubes::PerVoxelMarchingCubesResults>
        triangle_candidates;
    getTriangleCandidatesInBlock(block, neighbor_blocks, block_index,
                                 block_size, &triangle_candidates);

    if (triangle_candidates.size() <= 0) {
      continue;
    }

    // Allocate the mesh block.
    MeshBlock::Ptr mesh_block = mesh_layer->allocateBlockAtIndex(block_index);

    // Then actually calculate the triangles.
    for (const marching_cubes::PerVoxelMarchingCubesResults& candidate :
         triangle_candidates) {
      marching_cubes::meshCube(candidate, mesh_block.get());
    }
  }

  return true;
}

bool MeshIntegrator::isBlockMeshable(
    const VoxelBlock<TsdfVoxel>::ConstPtr block, float cutoff) const {
  constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;

  Index3D voxel_index;
  // Iterate over all the voxels:
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        const TsdfVoxel* voxel =
            &block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

        // Check if voxel distance is within the cutoff to determine if
        // there's going to be a surface boundary in this block.
        if (voxel->weight >= min_weight_ &&
            std::abs(voxel->distance) <= cutoff) {
          return true;
        }
      }
    }
  }
  return false;
}

void MeshIntegrator::getTriangleCandidatesInBlock(
    const TsdfBlock::ConstPtr block,
    const std::vector<TsdfBlock::ConstPtr>& neighbor_blocks,
    const Index3D& block_index, const float block_size,
    std::vector<marching_cubes::PerVoxelMarchingCubesResults>*
        triangle_candidates) {
  CHECK_NOTNULL(block);
  CHECK_NOTNULL(triangle_candidates);

  constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
  const float voxel_size = block_size / kVoxelsPerSide;

  Index3D voxel_index;
  // Iterate over all inside voxels.
  for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
       voxel_index.x()++) {
    for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
         voxel_index.y()++) {
      for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
           voxel_index.z()++) {
        // Get the position of this voxel.
        Vector3f voxel_position = getCenterPostionFromBlockIndexAndVoxelIndex(
            block_size, block_index, voxel_index);

        marching_cubes::PerVoxelMarchingCubesResults neighbors;
        // Figure out if this voxel is actually a triangle candidate.
        if (getTriangleCandidatesAroundVoxel(block, neighbor_blocks,
                                             voxel_index, voxel_position,
                                             voxel_size, &neighbors)) {
          triangle_candidates->push_back(neighbors);
        }
      }
    }
  }
}

bool MeshIntegrator::getTriangleCandidatesAroundVoxel(
    const TsdfBlock::ConstPtr block,
    const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
    const Index3D& voxel_index, const Vector3f& voxel_position,
    const float voxel_size,
    marching_cubes::PerVoxelMarchingCubesResults* neighbors) {
  DCHECK_EQ(neighbor_blocks.size(), 8);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  for (unsigned int i = 0; i < 8; ++i) {
    Index3D corner_index = voxel_index + cube_index_offsets_.col(i);

    // Are we in bounds? If not, have to get a neighbor.
    // The neighbor should correspond to the index in neighbor blocks.

    bool search_neighbor = false;
    Index3D block_offset(0, 0, 0);
    for (int j = 0; j < 3; j++) {
      if (corner_index[j] >= kVoxelsPerSide) {
        // Here the index is too much.
        corner_index(j) -= kVoxelsPerSide;
        block_offset(j) = 1;
        search_neighbor = true;
      }
    }

    const TsdfVoxel* voxel = nullptr;
    // Get the voxel either from the current block or from the corresponding
    // neighbor.
    if (search_neighbor) {
      // Find the correct neighbor block.
      int neighbor_index =
          marching_cubes::neighborIndexFromDirection(block_offset);
      if (neighbor_blocks[neighbor_index] == nullptr) {
        return false;
      }
      voxel =
          &neighbor_blocks[neighbor_index]
               ->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
    } else {
      voxel =
          &block->voxels[corner_index.x()][corner_index.y()][corner_index.z()];
    }

    // If any of the neighbors are not observed, this can't be a mesh
    // triangle.
    if (voxel->weight < min_weight_) {
      return false;
    }
    neighbors->vertex_sdf[i] = voxel->distance;
    neighbors->vertex_coords[i] =
        voxel_position + voxel_size * cube_index_offsets_.col(i).cast<float>();
  }

  // Figure out the index if we've made it this far.
  neighbors->marching_cubes_table_index =
      marching_cubes::calculateVertexConfiguration(neighbors->vertex_sdf);

  // Index 0 & 255 contain no triangles so not worth outputting it.
  return neighbors->marching_cubes_table_index != 0 &&
         neighbors->marching_cubes_table_index != 255;
}

}  // namespace nvblox