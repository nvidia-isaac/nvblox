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

#include "nvblox/tests/esdf_integrator_cpu.h"

namespace nvblox {

void EsdfIntegratorCPU::integrateBlocks(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    EsdfLayer* esdf_layer) {
  timing::Timer esdf_timer("esdf/integrate");

  timing::Timer allocate_timer("esdf/integrate/allocate");
  // First, allocate all the destination blocks.
  allocateBlocksOnCPU(block_indices, esdf_layer);
  allocate_timer.Stop();

  timing::Timer mark_timer("esdf/integrate/mark_sites");
  // Then, mark all the sites on CPU.
  // This finds all the blocks that are eligible to be parents.
  std::vector<Index3D> blocks_with_sites;
  std::vector<Index3D> blocks_to_clear;
  markAllSitesOnCPU(tsdf_layer, block_indices, esdf_layer, &blocks_with_sites,
                    &blocks_to_clear);
  mark_timer.Stop();

  timing::Timer clear_timer("esdf/integrate/clear_invalid");
  std::vector<Index3D> cleared_blocks;

  clearInvalidOnCPU(blocks_to_clear, esdf_layer, &cleared_blocks);

  VLOG(3) << "Invalid blocks cleared: " << cleared_blocks.size();
  clear_timer.Stop();

  timing::Timer compute_timer("esdf/integrate/compute");
  // Parallel block banding on CPU.
  // First we call all the blocks with sites to propagate the distances out
  // from the sites.
  computeEsdfOnCPU(blocks_with_sites, esdf_layer);
  // In case some blocks were cleared (otherwise this is a no-op), we also need
  // to make sure any cleared blocks get their values updated. This simply
  // ensures that their values are up-to-date.
  // TODO(helen): check if we can just roll cleared blocks into the list with
  // blocks with sites and only call it once. Compare how the speed varies.
  computeEsdfOnCPU(cleared_blocks, esdf_layer);
  compute_timer.Stop();
}

void EsdfIntegratorCPU::markAllSitesOnCPU(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    EsdfLayer* esdf_layer, std::vector<Index3D>* blocks_with_sites,
    std::vector<Index3D>* blocks_to_clear) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(blocks_with_sites);

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = tsdf_layer.block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  // Cache the minimum distance in metric size.
  const float max_site_distance_m = max_tsdf_site_distance_vox_ * voxel_size;
  int num_observed = 0;
  for (const Index3D& block_index : block_indices) {
    EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
    TsdfBlock::ConstPtr tsdf_block = tsdf_layer.getBlockAtIndex(block_index);

    if (!esdf_block || !tsdf_block) {
      LOG(ERROR) << "Somehow trying to update non-existent blocks!";
      continue;
    }

    bool block_has_sites = false;
    bool has_observed = false;
    bool block_to_clear = false;
    // Iterate over all the voxels:
    Index3D voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          // Get the voxel and call the callback on it:
          const TsdfVoxel* tsdf_voxel =
              &tsdf_block
                   ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          EsdfVoxel* esdf_voxel =
              &esdf_block
                   ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          if (tsdf_voxel->weight >= tsdf_min_weight_) {
            esdf_voxel->observed = true;
            // Mark as inside if the voxel distance is negative.
            has_observed = true;
            bool now_inside = tsdf_voxel->distance <= 0;
            if (esdf_voxel->is_inside && now_inside == false) {
              block_to_clear = true;
              esdf_voxel->squared_distance_vox = max_squared_distance_vox;
              esdf_voxel->parent_direction.setZero();
            }
            esdf_voxel->is_inside = now_inside;
            if (now_inside &&
                std::abs(tsdf_voxel->distance) <= max_site_distance_m) {
              esdf_voxel->is_site = true;
              esdf_voxel->squared_distance_vox = 0.0f;
              esdf_voxel->parent_direction.setZero();
              block_has_sites = true;
            } else {
              if (esdf_voxel->is_site) {
                esdf_voxel->is_site = false;
                block_to_clear = true;
                // This block needs to be cleared since something became an
                // un-site.
                esdf_voxel->squared_distance_vox = max_squared_distance_vox;
                esdf_voxel->parent_direction.setZero();
                // Other case is a brand new voxel.
              } else if (esdf_voxel->squared_distance_vox <= 0.0f) {
                esdf_voxel->squared_distance_vox = max_squared_distance_vox;
                esdf_voxel->parent_direction.setZero();
              }
            }
          }
        }
      }
    }
    if (block_has_sites) {
      blocks_with_sites->push_back(block_index);
    }
    if (block_to_clear) {
      blocks_to_clear->push_back(block_index);
    }
    if (has_observed) {
      num_observed++;
    }
  }
}

void EsdfIntegratorCPU::clearInvalidOnCPU(
    const std::vector<Index3D>& blocks_to_clear, EsdfLayer* esdf_layer,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(updated_blocks);

  updated_blocks->clear();
  Index3DSet all_cleared_blocks;

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  Index3DSet neighbor_clearing_set;

  int cleared_voxels = 0;

  // Ok so the goal is to basically check the neighbors along the edges,
  // and then continue sweeping if necessary.

  // Assumptions: all blocks with sites are correct now (they got updated
  // earlier). If a block contains an invalid site, it is also present along the
  // edge.
  for (const Index3D& block_index : blocks_to_clear) {
    VLOG(3) << "Clearing WITHIN block: " << block_index.transpose();

    EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
    if (!esdf_block) {
      continue;
    }
    bool any_cleared = false;
    // First we just reset all the voxels within the block that point to
    // non-sites.
    Index3D voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          EsdfVoxel* esdf_voxel =
              &esdf_block
                   ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          if (!esdf_voxel->observed || esdf_voxel->is_site ||
              esdf_voxel->squared_distance_vox >= max_squared_distance_vox) {
            continue;
          }

          // Get the parent.
          Index3D parent_index = voxel_index + esdf_voxel->parent_direction;

          // Check if the voxel is within the same block.
          if (parent_index.x() < 0 || parent_index.x() >= kVoxelsPerSide ||
              parent_index.y() < 0 || parent_index.y() >= kVoxelsPerSide ||
              parent_index.z() < 0 || parent_index.z() >= kVoxelsPerSide) {
            continue;
          }

          // Ok check if the parent index is a site.
          if (!esdf_block
                   ->voxels[parent_index.x()][parent_index.y()]
                           [parent_index.z()]
                   .is_site) {
            clearVoxel(esdf_voxel, max_squared_distance_vox);
            cleared_voxels++;
            any_cleared = true;
          }
        }
      }
    }

    // Iterate over all the neighbors.
    // If any of the neighbors reach the edge in clearing, we need to continue
    // updating neighbors.
    if (any_cleared) {
      neighbor_clearing_set.insert(block_index);
      all_cleared_blocks.insert(block_index);
    }
  }
  // Ok after we're done with everything, also update any neighbors that need
  // clearing.
  Index3DSet new_neighbor_clearing_set;

  while (!neighbor_clearing_set.empty()) {
    for (const Index3D& block_index : neighbor_clearing_set) {
      VLOG(3) << "Clearing block: " << block_index.transpose();
      clearNeighborsOnCPU(block_index, esdf_layer, &new_neighbor_clearing_set);
      all_cleared_blocks.insert(block_index);
    }
    VLOG(3) << "Num neighbors to clear: " << new_neighbor_clearing_set.size();
    std::swap(new_neighbor_clearing_set, neighbor_clearing_set);
    new_neighbor_clearing_set.clear();
  }

  VLOG(3) << "Updated voxels by clearing: " << cleared_voxels;

  std::copy(all_cleared_blocks.begin(), all_cleared_blocks.end(),
            std::back_inserter(*updated_blocks));
}

// This function was written for testing -- rather than incrementally searching
// for cleared blocks from an initial seed lsit, this iterates over the entire
// map and checks the parent of every single voxel. This cannot be parallelized
// without having the whole hash table on GPU, which is why we went with the
// other method for parallelization.
void EsdfIntegratorCPU::clearAllInvalidOnCPU(
    EsdfLayer* esdf_layer, std::vector<Index3D>* updated_blocks) {
  // Go through all blocks in the map, clearing any that point to an invalid
  // parent.
  std::vector<Index3D> block_indices = esdf_layer->getAllBlockIndices();

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  Index3DSet neighbor_clearing_set;

  int cleared_voxels = 0;

  // Ok so the goal is to basically check the neighbors along the edges,
  // and then continue sweeping if necessary.

  // Assumptions: all blocks with sites are correct now (they got updated
  // earlier). If a block contains an invalid site, it is also present along the
  // edge.
  for (const Index3D& block_index : block_indices) {
    EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
    if (!esdf_block) {
      continue;
    }
    bool any_cleared = false;

    Index3D voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          EsdfVoxel* esdf_voxel =
              &esdf_block
                   ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          if (!esdf_voxel->observed || esdf_voxel->is_site) {
            continue;
          }

          // Get the parent.
          Index3D parent_index = voxel_index + esdf_voxel->parent_direction;

          if (esdf_voxel->parent_direction == Index3D::Zero()) {
            continue;
          }

          // Check if the voxel is within the same block.
          if (parent_index.x() < 0 || parent_index.x() >= kVoxelsPerSide ||
              parent_index.y() < 0 || parent_index.y() >= kVoxelsPerSide ||
              parent_index.z() < 0 || parent_index.z() >= kVoxelsPerSide) {
            // Then we need to get the block index.
            Index3D neighbor_block_index = block_index;

            // Find the parent index.
            while (parent_index.x() >= kVoxelsPerSide) {
              parent_index.x() -= kVoxelsPerSide;
              neighbor_block_index.x()++;
            }
            while (parent_index.y() >= kVoxelsPerSide) {
              parent_index.y() -= kVoxelsPerSide;
              neighbor_block_index.y()++;
            }
            while (parent_index.z() >= kVoxelsPerSide) {
              parent_index.z() -= kVoxelsPerSide;
              neighbor_block_index.z()++;
            }

            while (parent_index.x() < 0) {
              parent_index.x() += kVoxelsPerSide;
              neighbor_block_index.x()--;
            }
            while (parent_index.y() < 0) {
              parent_index.y() += kVoxelsPerSide;
              neighbor_block_index.y()--;
            }
            while (parent_index.z() < 0) {
              parent_index.z() += kVoxelsPerSide;
              neighbor_block_index.z()--;
            }

            EsdfBlock::ConstPtr neighbor_block =
                esdf_layer->getBlockAtIndex(neighbor_block_index);
            const EsdfVoxel* neighbor_voxel =
                &neighbor_block->voxels[parent_index.x()][parent_index.y()]
                                       [parent_index.z()];
            if (!neighbor_voxel->is_site) {
              clearVoxel(esdf_voxel, max_squared_distance_vox);
              cleared_voxels++;
              any_cleared = true;
            }
          } else {
            // Ok check if the parent index is a site.
            if (!esdf_block
                     ->voxels[parent_index.x()][parent_index.y()]
                             [parent_index.z()]
                     .is_site) {
              clearVoxel(esdf_voxel, max_squared_distance_vox);

              cleared_voxels++;
              any_cleared = true;
            }
          }
        }
      }
    }
    if (any_cleared) {
      updated_blocks->push_back(block_index);
    }
  }
  VLOG(3) << "Cleared voxels in batch: " << cleared_voxels;
}

void EsdfIntegratorCPU::clearNeighborsOnCPU(const Index3D& block_index,
                                            EsdfLayer* esdf_layer,
                                            Index3DSet* neighbor_clearing_set) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(neighbor_clearing_set);

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;
  constexpr int kNumSweepDir = 3;

  int cleared_voxels = 0;

  EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
  if (!esdf_block) {
    LOG(ERROR) << "For some reason trying to clear an unallocated block.";
    return;
  }

  Index3D neighbor_index = block_index;
  for (int axis_to_sweep = 0; axis_to_sweep < kNumSweepDir; axis_to_sweep++) {
    for (int direction = -1; direction <= 1; direction += 2) {
      // Get this neighbor.
      bool block_updated = false;
      neighbor_index = block_index;
      neighbor_index(axis_to_sweep) += direction;

      EsdfBlock::Ptr neighbor_block =
          esdf_layer->getBlockAtIndex(neighbor_index);
      if (!neighbor_block) {
        continue;
      }

      VLOG(3) << "Block index: " << block_index.transpose()
              << " Neighbor index: " << neighbor_index.transpose()
              << " axis: " << axis_to_sweep << " dir: " << direction;

      // Iterate over the axes per neighbor.
      // First we look negative.
      int voxel_index = 0;
      int opposite_voxel_index = kVoxelsPerSide;
      // Then we look positive.
      if (direction > 0) {
        voxel_index = kVoxelsPerSide - 1;
        opposite_voxel_index = -1;
      }

      Index3D index = Index3D::Zero();
      index(axis_to_sweep) = voxel_index;
      int first_axis = 0, second_axis = 1;
      getSweepAxes(axis_to_sweep, &first_axis, &second_axis);
      for (index(first_axis) = 0; index(first_axis) < kVoxelsPerSide;
           index(first_axis)++) {
        for (index(second_axis) = 0; index(second_axis) < kVoxelsPerSide;
             index(second_axis)++) {
          // This is the closest ESDF voxel in our block to this guy.
          const EsdfVoxel* esdf_voxel =
              &esdf_block->voxels[index.x()][index.y()][index.z()];

          // If this isn't a cleared voxel, don't bother.
          if (/*!esdf_voxel->observed || */
              esdf_voxel->parent_direction != Index3D::Zero()) {
            continue;
          }

          for (int increment = 1; increment <= kVoxelsPerSide; increment++) {
            // Basically we just want to check if the edges' neighbors point
            // to something valid.
            Index3D neighbor_voxel_index = index;
            // Select the opposite voxel index.
            neighbor_voxel_index(axis_to_sweep) =
                opposite_voxel_index + increment * direction;
            EsdfVoxel* neighbor_voxel =
                &neighbor_block->voxels[neighbor_voxel_index.x()]
                                       [neighbor_voxel_index.y()]
                                       [neighbor_voxel_index.z()];

            // If either this was never set or is a site itself, we don't
            // care.
            if (!neighbor_voxel->observed || neighbor_voxel->is_site ||
                neighbor_voxel->squared_distance_vox >=
                    max_squared_distance_vox) {
              VLOG(3) << "Skipping because the voxel isn't updateable.";
              continue;
            }
            VLOG(3) << "Block index: " << block_index.transpose()
                    << " voxel index: " << index.transpose()
                    << " Neighbor voxel index: "
                    << neighbor_voxel_index.transpose()
                    << " Distance: " << neighbor_voxel->squared_distance_vox
                    << " Parent direction: "
                    << neighbor_voxel->parent_direction.transpose()
                    << " Sweep axis: " << axis_to_sweep
                    << " direction: " << direction
                    << " increment: " << increment;

            // If the direction of this neighbor isn't pointing in our
            // direction, we also don't care.
            Index3D parent_voxel_dir = neighbor_voxel->parent_direction;
            if ((direction > 0 && parent_voxel_dir(axis_to_sweep) >= 0) ||
                (direction < 0 && parent_voxel_dir(axis_to_sweep) <= 0)) {
              VLOG(3) << "Skipping because direction doesn't match. Parent "
                         "voxel dir: "
                      << parent_voxel_dir(axis_to_sweep)
                      << " Axis: " << axis_to_sweep
                      << " direction: " << direction;
              break;
            }

            // If it is, and the matching ESDF voxel parent doesn't match,
            // then it needs to be cleared.
            // Actually we can just clear it if the neighbor is cleared.
            clearVoxel(neighbor_voxel, max_squared_distance_vox);
            block_updated = true;
            cleared_voxels++;
            VLOG(3) << "Clearing voxel.";

            // We need to continue sweeping in this direction to clear the
            // rest of the band. Luckily this is already in the structure of
            // the for loop.
          }
        }
      }
      if (block_updated) {
        neighbor_clearing_set->insert(neighbor_index);
      }
    }
  }
  VLOG(3) << "Updated neighbor voxels by clearing: " << cleared_voxels
          << " updated blocks: " << neighbor_clearing_set->size();
}

void EsdfIntegratorCPU::computeEsdfOnCPU(
    const std::vector<Index3D>& blocks_with_sites, EsdfLayer* esdf_layer) {
  CHECK_NOTNULL(esdf_layer);

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;

  // First we go over all of the blocks with sites.
  // We compute all the proximal sites inside the block first.
  for (const Index3D& block_index : blocks_with_sites) {
    EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
    for (int i = 0; i < 3; i++) {
      sweepBlockBandOnCPU(i, voxel_size, esdf_block.get());
    }
  }

  std::vector<Index3D> updated_blocks;
  std::vector<Index3D> blocks_to_run = blocks_with_sites;
  int i = 0;
  do {
    updated_blocks.clear();
    // We then propagate the edges of these updated blocks to their immediate
    // neighbors.
    for (const Index3D& block_index : blocks_to_run) {
      updateLocalNeighborBandsOnCPU(block_index, esdf_layer, &updated_blocks);
    }

    // Then we update the inside values of these neighbors.
    for (const Index3D& block_index : updated_blocks) {
      EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
      for (int axis_idx = 0; axis_idx < 3; axis_idx++) {
        sweepBlockBandOnCPU(axis_idx, voxel_size, esdf_block.get());
      }
    }
    Index3DSet updated_set(updated_blocks.begin(), updated_blocks.end());
    blocks_to_run.assign(updated_set.begin(), updated_set.end());
    VLOG(3) << "Update " << i << " updated blocks: " << blocks_to_run.size();
    i++;
    // We continue updating in a brushfire pattern until no more blocks are
    // getting updated.
  } while (!updated_blocks.empty());
}

void EsdfIntegratorCPU::updateLocalNeighborBandsOnCPU(
    const Index3D& block_index, EsdfLayer* esdf_layer,
    std::vector<Index3D>* updated_blocks) {
  constexpr int kNumSweepDir = 3;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  // Get the center block.
  EsdfBlock::Ptr esdf_block = esdf_layer->getBlockAtIndex(block_index);
  if (!esdf_block) {
    return;
  }

  // Iterate over all the neighbors.
  Index3D neighbor_index = block_index;
  for (int axis_to_sweep = 0; axis_to_sweep < kNumSweepDir; axis_to_sweep++) {
    for (int direction = -1; direction <= 1; direction += 2) {
      // Get this neighbor.
      bool block_updated = false;
      neighbor_index = block_index;
      neighbor_index(axis_to_sweep) += direction;

      EsdfBlock::Ptr neighbor_block =
          esdf_layer->getBlockAtIndex(neighbor_index);
      if (!neighbor_block) {
        continue;
      }

      // Iterate over the axes per neighbor.
      // First we look negative.
      int voxel_index = 0;
      int opposite_voxel_index = kVoxelsPerSide - 1;
      // Then we look positive.
      if (direction > 0) {
        voxel_index = opposite_voxel_index;
        opposite_voxel_index = 0;
      }

      Index3D index = Index3D::Zero();
      index(axis_to_sweep) = voxel_index;
      int first_axis = 0, second_axis = 1;
      getSweepAxes(axis_to_sweep, &first_axis, &second_axis);
      for (index(first_axis) = 0; index(first_axis) < kVoxelsPerSide;
           index(first_axis)++) {
        for (index(second_axis) = 0; index(second_axis) < kVoxelsPerSide;
             index(second_axis)++) {
          // This is the bottom interface. So pixel 0 of our block should be
          // pixel kVoxelsPerSide-1 of the search block.
          const EsdfVoxel* esdf_voxel =
              &esdf_block->voxels[index.x()][index.y()][index.z()];
          Index3D neighbor_voxel_index = index;
          // Select the opposite voxel index.
          neighbor_voxel_index(axis_to_sweep) = opposite_voxel_index;
          EsdfVoxel* neighbor_voxel =
              &neighbor_block
                   ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                           [neighbor_voxel_index.z()];

          if (!esdf_voxel->observed || !neighbor_voxel->observed ||
              neighbor_voxel->is_site ||
              esdf_voxel->squared_distance_vox >= max_squared_distance_vox) {
            continue;
          }
          // Determine if we can update this.
          Eigen::Vector3i potential_direction = esdf_voxel->parent_direction;
          potential_direction(axis_to_sweep) += -direction;
          float potential_distance = potential_direction.squaredNorm();
          if (neighbor_voxel->squared_distance_vox > potential_distance) {
            neighbor_voxel->parent_direction = potential_direction;
            neighbor_voxel->squared_distance_vox = potential_distance;
            block_updated = true;
          }
        }
      }
      if (block_updated) {
        updated_blocks->push_back(neighbor_index);
      }
    }
  }
}

void EsdfIntegratorCPU::propagateEdgesOnCPU(int axis_to_sweep, float voxel_size,
                                            EsdfBlock* esdf_block) {
  CHECK_NOTNULL(esdf_block);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  // Select the axes to increment.
  int first_axis = 0, second_axis = 1;
  getSweepAxes(axis_to_sweep, &first_axis, &second_axis);

  // Just need to sweep once through.
  Index3D index = Index3D::Zero();
  for (index(first_axis) = 0; index(first_axis) < kVoxelsPerSide;
       index(first_axis)++) {
    for (index(second_axis) = 0; index(second_axis) < kVoxelsPerSide;
         index(second_axis)++) {
      // Get the min and max of the band.
      Index3D bottom_index = index;
      bottom_index(axis_to_sweep) = 0;
      Index3D top_index = index;
      top_index(axis_to_sweep) = kVoxelsPerSide - 1;
      EsdfVoxel* bottom_voxel =
          &esdf_block
               ->voxels[bottom_index.x()][bottom_index.y()][bottom_index.z()];
      EsdfVoxel* top_voxel =
          &esdf_block->voxels[top_index.x()][top_index.y()][top_index.z()];
      Eigen::Vector3i bottom_parent_relative = bottom_voxel->parent_direction;
      bottom_parent_relative += bottom_index;
      Eigen::Vector3i top_parent_relative = top_voxel->parent_direction;
      top_parent_relative += top_index;

      // Check if they're actually valid, lol.
      bool top_valid = top_voxel->observed && top_voxel->squared_distance_vox <
                                                  max_squared_distance_vox;
      bool bottom_valid =
          bottom_voxel->observed &&
          bottom_voxel->squared_distance_vox < max_squared_distance_vox;

      for (index(axis_to_sweep) = 0; index(axis_to_sweep) < kVoxelsPerSide;
           index(axis_to_sweep)++) {
        EsdfVoxel* esdf_voxel =
            &esdf_block->voxels[index.x()][index.y()][index.z()];
        if (!esdf_voxel->observed || esdf_voxel->is_site) {
          continue;
        }
        // Check if we should update this with either the top or the bottom
        // voxel.
        if (bottom_valid) {
          Eigen::Vector3i potential_direction = bottom_parent_relative - index;
          float potential_distance = potential_direction.squaredNorm();
          if (esdf_voxel->squared_distance_vox > potential_distance) {
            esdf_voxel->parent_direction = potential_direction;
            esdf_voxel->squared_distance_vox = potential_distance;
          }
        }
        if (top_valid) {
          Eigen::Vector3i potential_direction = top_parent_relative - index;
          float potential_distance = potential_direction.squaredNorm();
          if (esdf_voxel->squared_distance_vox > potential_distance) {
            esdf_voxel->parent_direction = potential_direction;
            esdf_voxel->squared_distance_vox = potential_distance;
          }
        }
      }
    }
  }
}

void EsdfIntegratorCPU::sweepBlockBandOnCPU(int axis_to_sweep, float voxel_size,
                                            EsdfBlock* esdf_block) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float max_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_distance_vox = max_distance_vox * max_distance_vox;

  // Select the axes to increment.
  int first_axis = 0, second_axis = 1;
  getSweepAxes(axis_to_sweep, &first_axis, &second_axis);

  Index3D index = Index3D::Zero();

  // Keep track of if we've seen any sites so far.
  // The secondary pass has "fake" sites; i.e., sites outside of the block.
  Index3D last_site = Index3D::Zero();
  bool site_found = false;
  for (index(first_axis) = 0; index(first_axis) < kVoxelsPerSide;
       index(first_axis)++) {
    for (index(second_axis) = 0; index(second_axis) < kVoxelsPerSide;
         index(second_axis)++) {
      // First we sweep forward, then backwards.
      for (int i = 0; i < 2; i++) {
        last_site = Index3D::Zero();
        site_found = false;
        int direction = 1;
        int start_voxel = 0;
        int end_voxel = kVoxelsPerSide;
        if (i == 1) {
          direction = -1;
          start_voxel = kVoxelsPerSide - 1;
          end_voxel = -1;
        }
        for (index(axis_to_sweep) = start_voxel;
             index(axis_to_sweep) != end_voxel;
             index(axis_to_sweep) += direction) {
          EsdfVoxel* esdf_voxel =
              &esdf_block->voxels[index.x()][index.y()][index.z()];
          if (!esdf_voxel->observed) {
            continue;
          }
          // If this voxel is itself a site, then mark this for future voxels.
          if (esdf_voxel->is_site) {
            last_site = index;
            site_found = true;
          } else if (!site_found) {
            // If this voxel isn't a site but we haven't found a site yet,
            // then if this voxel is valid we set it as the site.
            if (esdf_voxel->squared_distance_vox < max_squared_distance_vox) {
              site_found = true;
              last_site = esdf_voxel->parent_direction + index;
            }
          } else {
            // If we've found the site, then should just decide what to do
            // here.
            Index3D potential_direction = last_site - index;
            float potential_distance = potential_direction.squaredNorm();
            // Either it hasn't been set at all or it's closer to the site
            // than to its current value.
            if (esdf_voxel->squared_distance_vox > potential_distance) {
              esdf_voxel->parent_direction = potential_direction;
              esdf_voxel->squared_distance_vox = potential_distance;
            } else if (esdf_voxel->squared_distance_vox <
                       max_squared_distance_vox) {
              // If the current value is a better site, then set it as a site.
              last_site = esdf_voxel->parent_direction + index;
            }
          }
        }
      }
    }
  }
}

void EsdfIntegratorCPU::clearVoxel(EsdfVoxel* voxel,
                                   float max_squared_distance_vox) {
  voxel->parent_direction.setZero();
  voxel->squared_distance_vox = max_squared_distance_vox;
}

void EsdfIntegratorCPU::getSweepAxes(int axis_to_sweep, int* first_axis,
                                     int* second_axis) const {
  // Pick an order of the axes to sweep.
  switch (axis_to_sweep) {
    case 0:
      *first_axis = 1;
      *second_axis = 2;
      break;
    case 1:
      *first_axis = 0;
      *second_axis = 2;
      break;
    case 2:
      *first_axis = 0;
      *second_axis = 1;
      break;
  }
}

}  // namespace nvblox
