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
#include "nvblox/integrators/esdf_integrator.h"

namespace nvblox {

/// @brief A class, just for testing, that performs the ESDF integration on CPU.
/// Was originally written for development, now only used for testing.
class EsdfIntegratorCPU : public EsdfIntegrator {
 public:
  EsdfIntegratorCPU() : EsdfIntegrator() {}
  virtual ~EsdfIntegratorCPU() {}

  /// Build an EsdfLayer from a TsdfLayer (incremental on CPU)
  /// @param tsdf_layer The input TsdfLayer
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the TSDF at these indices has changed).
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateBlocks(const TsdfLayer& tsdf_layer,
                       const std::vector<Index3D>& block_indices,
                       EsdfLayer* esdf_layer) override;

 protected:
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

  /// Helper function to figure out which axes to sweep first and second.
  void getSweepAxes(int axis_to_sweep, int* first_axis, int* second_axis) const;
};

}  // namespace nvblox