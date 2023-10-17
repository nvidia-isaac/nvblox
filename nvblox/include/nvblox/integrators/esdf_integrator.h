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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/log_odds.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

// Forward declaration.
struct Index3DDeviceSet;

struct OccupancySiteFunctor;
struct TsdfSiteFunctor;

/// A class performing (incremental) ESDF integration
///
/// The Euclidean Signed Distance Function (ESDF) is a distance function where
/// obstacle distances are true (in the sense that they are not distances along
/// the observation ray as they are in the TSDF). This class calculates an
/// ESDFLayer from an input TSDFLayer.
class EsdfIntegrator {
 public:
  static constexpr float kDefaultMaxEsdfDistanceM = 2.0;
  static constexpr float kDefaultMaxTsdfSiteDistanceVox = 1.0;
  static constexpr float kDefaultMinTsdfWeight = 1e-4;

  EsdfIntegrator();
  EsdfIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~EsdfIntegrator() = default;

  /// Build an EsdfLayer from a TsdfLayer
  /// @param tsdf_layer The input TsdfLayer
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateLayer(const TsdfLayer& tsdf_layer, EsdfLayer* esdf_layer);

  /// Build an EsdfLayer from a TsdfLayer and a FreespaceLayer
  /// @param tsdf_layer The input TsdfLayer
  /// @param freespace_layer The input freespace layer (esdf sites are
  /// ignored if they fall into freespace)
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateLayer(const TsdfLayer& tsdf_layer,
                      const FreespaceLayer& freespace_layer,
                      EsdfLayer* esdf_layer);

  /// Build an EsdfLayer from a OccupancyLayer
  /// @param occupancy_layer The input OccupancyLayer
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateLayer(const OccupancyLayer& occupancy_layer,
                      EsdfLayer* esdf_layer);

  /// Build an EsdfLayer from a TsdfLayer (incremental) (on GPU)
  /// @param tsdf_layer The input TsdfLayer
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the TSDF at these indices has changed).
  /// @param[out] esdf_layer The output EsdfLayer
  virtual void integrateBlocks(const TsdfLayer& tsdf_layer,
                               const std::vector<Index3D>& block_indices,
                               EsdfLayer* esdf_layer);

  /// Build an EsdfLayer from a TsdfLayer and a FreespaceLayer
  /// (incremental) (on GPU)
  /// @param tsdf_layer The input TsdfLayer
  /// @param freespace_layer The input freespace layer (esdf sites are
  /// ignored if they fall into freespace)
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the TSDF at these indices has changed).
  /// @param[out] esdf_layer The output EsdfLayer
  virtual void integrateBlocks(const TsdfLayer& tsdf_layer,
                               const FreespaceLayer& freespace_layer,
                               const std::vector<Index3D>& block_indices,
                               EsdfLayer* esdf_layer);

  /// @brief Build an EsdfLayer from an OccupancyLayer(incremental) (on GPU)
  /// @param occupancy_layer The input OccupancyLayer
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the Occupancy at these indices has changed).
  /// @param[out] esdf_layer The output EsdfLayer
  virtual void integrateBlocks(const OccupancyLayer& occupancy_layer,
                               const std::vector<Index3D>& block_indices,
                               EsdfLayer* esdf_layer);

  /// Build an EsdfLayer slice from a TsdfLayer (incremental) (on GPU)
  /// This function takes the voxels between z_min and z_max in the TsdfLayer.
  /// Any obstacle in this z range generates an obstacle in the ESDF output
  /// slice. The 2D ESDF is written to voxels with a single z index in the
  /// output (i.e. the z_output).
  /// @param tsdf_layer The input TsdfLayer
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the TSDF at these indices has changed).
  /// @param z_min The minimum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param z_max The maximum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param  z_output The height (in meters) (in the layer frame) where the
  /// ESDF slice is written to.
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateSlice(const TsdfLayer& tsdf_layer,
                      const std::vector<Index3D>& block_indices, float z_min,
                      float z_max, float z_output, EsdfLayer* esdf_layer);

  /// Build an EsdfLayer slice from a TsdfLayer and a FreespaceLayer
  /// (incremental) (on GPU)
  /// This function takes the voxels between z_min and
  /// z_max in the TsdfLayer. Any surface in this z range generates a surface
  /// for ESDF computation in 2D. The 2D ESDF if written to a voxels with a
  /// single z index in the output.
  /// @param tsdf_layer The input TsdfLayer
  /// @param freespace_layer The input freespace layer (esdf sites are
  /// ignored if they fall into freespace)
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the TSDF at these indices has changed).
  /// @param z_min The minimum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param z_max The maximum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param  z_output The height (in meters) (in the layer frame) where the
  /// ESDF slice is written to.
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateSlice(const TsdfLayer& tsdf_layer,
                      const FreespaceLayer& freespace_layer,
                      const std::vector<Index3D>& block_indices, float z_min,
                      float z_max, float z_output, EsdfLayer* esdf_layer);

  /// Build an EsdfLayer slice from a OccupancyLayer (incremental) (on GPU)
  /// This function takes the voxels between z_min and z_max in the TsdfLayer.
  /// Any obstacle in this z range generates an obstacle in the ESDF output
  /// slice. The 2D ESDF is written to voxels with a single z index in the
  /// output (i.e. the z_output).
  /// @param occupancy_layer The input OccupancyLayer
  /// @param block_indices The indices of the EsdfLayer which should be updated
  /// (usually because the Occupancy at these indices has changed).
  /// @param z_min The minimum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param z_max The maximum height (in meters) (in the layer frame) at which
  /// an obstacle is considered.
  /// @param  z_output The height (in meters) (in the layer frame) where the
  /// ESDF slice is written to.
  /// @param[out] esdf_layer The output EsdfLayer
  void integrateSlice(const OccupancyLayer& occupancy_layer,
                      const std::vector<Index3D>& block_indices, float z_min,
                      float z_max, float z_output, EsdfLayer* esdf_layer);

  /// A parameter getter
  /// The maximum distance in meters out to which to calculate the ESDF.
  /// @returns the maximum distance
  float max_esdf_distance_m() const;

  /// A parameter getter
  /// The maximum (TSDF) distance at which we call a voxel in the TsdfLatyer a
  /// "site". A site is a voxel that is on the surface for the purposes of ESDF
  /// calculation.
  /// @returns the maximum distance
  float max_site_distance_vox() const;

  /// A parameter getter
  /// The minimum (TSDF) weight at which we call a voxel in the TsdfLatyer a
  /// "site". A site is a voxel that is on the surface for the purposes of ESDF
  /// calculation.
  /// @returns the minimum weight
  float min_weight() const;

  /// A parameter getter
  /// The minimum probability (between 0.0 and 1.0) which we consider an
  /// occupancy voxel occupied.
  /// @returns the minimum probability
  float occupied_threshold() const;

  /// A parameter setter
  /// See truncation_distance_vox().
  /// @param max_esdf_distance_m The maximum distance out to which to calculate
  /// the ESDF.
  void max_esdf_distance_m(float max_esdf_distance_m);

  /// A parameter setter
  /// See max_site_distance_vox().
  /// @param max_site_distance_vox the max distance to a site in voxels.
  void max_site_distance_vox(float max_site_distance_vox);

  /// A parameter setter
  /// See min_weight().
  /// @param min_weight the minimum weight at which to consider a voxel a site.
  void min_weight(float min_weight);

  /// A parameter setter
  /// See occupied_threshold()
  /// @param occupied_threshold the minimum probability.
  void occupied_threshold(float occupied_threshold);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  /// Templated version of the public functions above, used internally.
  template <typename LayerType>
  void integrateBlocksTemplate(
      const LayerType& layer, const std::vector<Index3D>& block_indices,
      EsdfLayer* esdf_layer,
      const FreespaceLayer* freespace_layer_ptr = nullptr);

  template <typename LayerType>
  void integrateSliceTemplate(
      const LayerType& layer, const std::vector<Index3D>& block_indices,
      float z_min, float z_max, float z_output, EsdfLayer* esdf_layer,
      const FreespaceLayer* freespace_layer_ptr = nullptr);

  /// Allocate all blocks in the given block indices list.
  void allocateBlocksOnCPU(const std::vector<Index3D>& block_indices,
                           EsdfLayer* esdf_layer);

  /// Gets the site-finding functors for a specific layer type.
  OccupancySiteFunctor getSiteFunctor(const OccupancyLayer& layer);
  TsdfSiteFunctor getSiteFunctor(const TsdfLayer& layer);

  template <typename LayerType>
  void markAllSites(const LayerType& layer,
                    const std::vector<Index3D>& block_indices,
                    const FreespaceLayer* freespace_layer_ptr,
                    EsdfLayer* esdf_layer,
                    device_vector<Index3D>* blocks_with_sites,
                    device_vector<Index3D>* cleared_blocks);

  // Same as the markAllSites function above but basically makes the
  // whole operation in 2D. Considers a min and max z in a bounding box which is
  // compressed down into a single layer.
  template <typename LayerType>
  void markSitesInSlice(const LayerType& layer,
                        const std::vector<Index3D>& block_indices, float min_z,
                        float max_z, float output_z,
                        const FreespaceLayer* freespace_layer_ptr,
                        EsdfLayer* esdf_layer,
                        device_vector<Index3D>* updated_blocks,
                        device_vector<Index3D>* cleared_blocks);

  // Internal helpers for GPU computation.
  void updateNeighborBands(device_vector<Index3D>* block_indices,
                           EsdfLayer* esdf_layer,
                           float max_squared_esdf_distance_vox,
                           device_vector<Index3D>* updated_block_indices);

  void sweepBlockBand(device_vector<Index3D>* block_indices,
                      EsdfLayer* esdf_layer,
                      float max_squared_esdf_distance_vox);
  void computeEsdf(const device_vector<Index3D>& blocks_with_sites,
                   EsdfLayer* esdf_layer);
  void clearAllInvalid(const std::vector<Index3D>& blocks_to_clear,
                       EsdfLayer* esdf_layer,
                       device_vector<Index3D>* updated_blocks);

  // Helper method to de-dupe block indices.
  void sortAndTakeUniqueIndices(device_vector<Index3D>* block_indices);

  /// @brief EsdfLayer related parameter
  /// Maximum distance to compute the ESDF.
  float max_esdf_distance_m_ = kDefaultMaxEsdfDistanceM;

  /// @brief TsdfLayer related parameter
  /// Maximum (TSDF) distance at which a voxel is considered a site
  float max_tsdf_site_distance_vox_ = kDefaultMaxTsdfSiteDistanceVox;

  /// @brief TsdfLayer related parameter
  /// Minimum weight to consider a TSDF voxel observed.
  float tsdf_min_weight_ = kDefaultMinTsdfWeight;

  /// @brief OccupancyLayer related parameter
  /// The log odds value greater than which we consider a voxel occupied
  float occupied_threshold_log_odds_ = logOddsFromProbability(0.5f);

  // State.
  std::shared_ptr<CudaStream> cuda_stream_;

  // Temporary storage variables so we don't have to reallocate as much.
  device_vector<Index3D> block_indices_device_;
  host_vector<Index3D> block_indices_host_;
  device_vector<Index3D> updated_indices_device_;
  host_vector<Index3D> updated_indices_host_;
  device_vector<Index3D> to_clear_indices_device_;
  host_vector<Index3D> to_clear_indices_host_;
  device_vector<Index3D> temp_indices_device_;
  host_vector<Index3D> temp_indices_host_;
  device_vector<Index3D> cleared_block_indices_device_;

  unified_ptr<int> updated_counter_device_;
  unified_ptr<int> updated_counter_host_;
  unified_ptr<int> cleared_counter_device_;
  unified_ptr<int> cleared_counter_host_;
};

}  // namespace nvblox
