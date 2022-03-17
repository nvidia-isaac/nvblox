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

#include "nvblox/core/blox.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/core/voxels.h"

namespace nvblox {

class EsdfIntegrator {
 public:
  // Just takes care of the cuda stream.
  ~EsdfIntegrator();

  // Integrate the entire layer.
  void integrateLayer(const TsdfLayer& tsdf_layer, EsdfLayer* esdf_layer);

  // Integrate specific blocks in a layer.
  void integrateBlocks(const TsdfLayer& tsdf_layer,
                       const std::vector<Index3D>& block_indices,
                       EsdfLayer* esdf_layer);

  // Separate CPU and GPU methods.
  void integrateBlocksOnCPU(const TsdfLayer& tsdf_layer,
                            const std::vector<Index3D>& block_indices,
                            EsdfLayer* esdf_layer);
  void integrateBlocksOnGPU(const TsdfLayer& tsdf_layer,
                            const std::vector<Index3D>& block_indices,
                            EsdfLayer* esdf_layer);
  void integrateSliceOnGPU(const TsdfLayer& tsdf_layer,
                           const std::vector<Index3D>& block_indices,
                           float z_min, float z_max, float z_output,
                           EsdfLayer* esdf_layer);

  // Setting setters and getting getters
  float max_distance_m() const { return max_distance_m_; }
  float& max_distance_m() { return max_distance_m_; }

  float min_site_distance_vox() const { return min_site_distance_vox_; }
  float& min_site_distance_vox() { return min_site_distance_vox_; }

  float min_weight() const { return min_weight_; }
  float& min_weight() { return min_weight_; }

  float max_squared_distance_vox(float voxel_size) const {
    return max_distance_m_ * max_distance_m_ / (voxel_size * voxel_size);
  }

 protected:
  void allocateBlocksOnCPU(const std::vector<Index3D>& block_indices,
                           EsdfLayer* esdf_layer);
  void markAllSitesOnCPU(const TsdfLayer& tsdf_layer,
                         const std::vector<Index3D>& block_indices,
                         EsdfLayer* esdf_layer,
                         std::vector<Index3D>* blocks_with_sites,
                         std::vector<Index3D>* blocks_to_clear);
  void computeEsdfOnCPU(const std::vector<Index3D>& blocks_with_sites,
                        EsdfLayer* esdf_layer);

  // Internal functions for ESDF computation.
  void sweepBlockBandOnCPU(int axis_to_sweep, float voxel_size,
                           EsdfBlock* esdf_block);
  void updateLocalNeighborBandsOnCPU(const Index3D& block_index,
                                     EsdfLayer* esdf_layer,
                                     std::vector<Index3D>* updated_blocks);
  void propagateEdgesOnCPU(int axis_to_sweep, float voxel_size,
                           EsdfBlock* esdf_block);

  // CPU-based incremental.
  void clearInvalidOnCPU(const std::vector<Index3D>& blocks_with_sites,
                         EsdfLayer* esdf_layer,
                         std::vector<Index3D>* updated_blocks);
  void clearNeighborsOnCPU(const Index3D& block_index, EsdfLayer* esdf_layer,
                           Index3DSet* neighbor_clearing_set);
  void clearVoxel(EsdfVoxel* voxel, float max_squared_distance_vox);

  // Test function to just clear everything in the whole map.
  void clearAllInvalidOnCPU(EsdfLayer* esdf_layer,
                            std::vector<Index3D>* updated_blocks);

  // GPU computation functions.
  void markAllSitesOnGPU(const TsdfLayer& tsdf_layer,
                         const std::vector<Index3D>& block_indices,
                         EsdfLayer* esdf_layer,
                         std::vector<Index3D>* blocks_with_sites,
                         std::vector<Index3D>* cleared_blocks);

  // Same as the other function but basically makes the whole operation 2D.
  // Considers a min and max z in a bounding box which is compressed down into a
  // single layer.
  void markSitesInSliceOnGPU(const TsdfLayer& tsdf_layer,
                             const std::vector<Index3D>& block_indices,
                             float min_z, float max_z, float output_z,
                             EsdfLayer* esdf_layer,
                             std::vector<Index3D>* output_blocks,
                             std::vector<Index3D>* cleared_blocks);

  void clearInvalidOnGPU(const std::vector<Index3D>& blocks_to_clear,
                         EsdfLayer* esdf_layer,
                         std::vector<Index3D>* updated_blocks);

  void computeEsdfOnGPU(const std::vector<Index3D>& blocks_with_sites,
                        EsdfLayer* esdf_layer);

  // Internal helpers for GPU computation.
  void sweepBlockBandOnGPU(device_vector<EsdfBlock*>& block_pointers,
                           float max_squared_distance_vox);
  void updateLocalNeighborBandsOnGPU(
      const std::vector<Index3D>& block_indices,
      device_vector<EsdfBlock*>& block_pointers, float max_squared_distance_vox,
      EsdfLayer* esdf_layer, std::vector<Index3D>* updated_blocks,
      device_vector<EsdfBlock*>* updated_block_pointers);
  void createNeighborTable(const std::vector<Index3D>& block_indices,
                           EsdfLayer* esdf_layer,
                           std::vector<Index3D>* neighbor_indices,
                           host_vector<EsdfBlock*>* neighbor_pointers,
                           host_vector<int>* neighbor_table);
  void clearBlockNeighbors(std::vector<Index3D>& clear_list,
                           EsdfLayer* esdf_layer,
                           std::vector<Index3D>* new_clear_list);

  // Helper function to figure out which axes to sweep first and second.
  void getSweepAxes(int axis_to_sweep, int* first_axis, int* second_axis) const;

  /// Maximum distance to compute the ESDF.
  float max_distance_m_ = 10.0;
  /// Minimum distance to a site
  float min_site_distance_vox_ = 1.0;
  /// Minimum weight to consider a TSDF voxel observed.
  float min_weight_ = 1e-4;

  // State.
  cudaStream_t cuda_stream_ = nullptr;

  // Temporary storage variables so we don't have to reallocate as much.
  host_vector<bool> updated_blocks_host_;
  device_vector<bool> updated_blocks_device_;
  host_vector<bool> cleared_blocks_host_;
  device_vector<bool> cleared_blocks_device_;
  host_vector<const TsdfBlock*> tsdf_pointers_host_;
  device_vector<const TsdfBlock*> tsdf_pointers_device_;
  host_vector<EsdfBlock*> block_pointers_host_;
  device_vector<EsdfBlock*> block_pointers_device_;
  host_vector<int> neighbor_table_host_;
  device_vector<int> neighbor_table_device_;
  host_vector<EsdfBlock*> neighbor_pointers_host_;
  device_vector<EsdfBlock*> neighbor_pointers_device_;
};

}  // namespace nvblox
