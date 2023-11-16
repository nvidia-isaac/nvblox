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

#include "nvblox/integrators/internal/projective_integrator.h"
#include "nvblox/integrators/weighting_function.h"

#include "nvblox/core/log_odds.h"

namespace nvblox {

struct UpdateOccupancyVoxelFunctor;

/// A class performing occupancy intregration
///
/// Integrates depth images and lidar scans into occupancy layers. The
/// "projective" describes one type of integration. Namely that voxels in view
/// are projected into the depth image (the alternative being casting rays out
/// from the camera).
class ProjectiveOccupancyIntegrator
    : public ProjectiveIntegrator<OccupancyVoxel> {
 public:
  static constexpr float kDefaultFreeRegionOccupancyProbability = 0.3;
  static constexpr float kDefaultOccupiedRegionOccupancyProbability = 0.7;
  static constexpr float kDefaultUnobservedRegionOccupancyProbability = 0.5;
  static constexpr float kDefaultOccupiedRegionHalfWidthM = 0.1;

  ProjectiveOccupancyIntegrator();
  ProjectiveOccupancyIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~ProjectiveOccupancyIntegrator();

  /// Integrates a depth image in to the passed occupancy layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param layer A pointer to the layer into which this observation will
  /// be intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain
  /// the 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera, OccupancyLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a depth image in to the passed occupancy layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param lidar A the LiDAR model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain
  /// the 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Lidar& lidar, OccupancyLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the free region
  /// observed on the sensor.
  /// @returns the free region occupancy probability
  float free_region_occupancy_probability() const;

  /// A parameter setter
  /// See free_region_occupancy_probability().
  /// @param value the free region occupancy probability.
  void free_region_occupancy_probability(float value);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the occupied region
  /// observed on the sensor.
  /// @returns the occupied occupancy probability
  float occupied_region_occupancy_probability() const;

  /// A parameter setter
  /// See occupied_region_occupancy_probability().
  /// @param value the occupied occupancy probability
  void occupied_region_occupancy_probability(float value);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the unobserved region
  /// @returns the unobserved occupancy probability
  float unobserved_region_occupancy_probability() const;

  /// A parameter setter
  /// See unobserved_region_occupancy_probability().
  /// @param value the unobserved occupancy probability
  void unobserved_region_occupancy_probability(float value);

  /// A parameter getter
  /// Half the width of the region which is consided as occupied i.e. where
  /// occupied_region_log_odds is applied to all voxels on update. The region is
  /// centered at the measured surface depth on the integrated frame.
  /// @returns the occupied region half width in meters
  float occupied_region_half_width_m() const;

  /// A parameter setter
  /// See occupied_region_half_width_m().
  /// @param occupied_region_half_width_m the occupied region half width in
  /// meters
  void occupied_region_half_width_m(float occupied_region_half_width_m);

  /// For voxels with a radius, allocate memory and give a small weight and
  /// truncation distance, effectively making these voxels free-space. Does not
  /// affect voxels which are already observed.
  /// @param center The center of the sphere affected.
  /// @param radius The radius of the sphere affected.
  /// @param layer A pointed to the layer which will be affected by the update.
  /// @param updated_blocks_ptr Optional pointer to a list of blocks affected by
  /// the update.
  void markUnobservedFreeInsideRadius(
      const Vector3f& center, float radius, OccupancyLayer* layer,
      std::vector<Index3D>* updated_blocks_ptr = nullptr);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  void setFunctorParameters(const float block_size);
  std::string getIntegratorName() const override;

  // Sensor model parameters
  float free_region_log_odds_ =
      logOddsFromProbability(kDefaultFreeRegionOccupancyProbability);
  float occupied_region_log_odds_ =
      logOddsFromProbability(kDefaultOccupiedRegionOccupancyProbability);
  float unobserved_region_log_odds_ =
      logOddsFromProbability(kDefaultUnobservedRegionOccupancyProbability);
  float occupied_region_half_width_m_ = kDefaultOccupiedRegionHalfWidthM;

  // Functor which defines the voxel update operation.
  unified_ptr<UpdateOccupancyVoxelFunctor> update_functor_host_ptr_;

  // Cuda stream
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
