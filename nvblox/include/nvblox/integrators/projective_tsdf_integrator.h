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

#include "nvblox/core/parameter_tree.h"
#include "nvblox/integrators/internal/projective_integrator.h"
#include "nvblox/integrators/weighting_function.h"

namespace nvblox {

struct UpdateTsdfVoxelFunctor;

/// A class performing TSDF intregration
///
/// Integrates depth images and lidar scans into TSDF layers. The "projective"
/// describes one type of integration. Namely that voxels in view are projected
/// into the depth image (the alternative being casting rays out from the
/// camera).
class ProjectiveTsdfIntegrator : public ProjectiveIntegrator<TsdfVoxel> {
 public:
  ProjectiveTsdfIntegrator();
  ProjectiveTsdfIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~ProjectiveTsdfIntegrator();

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera, TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param lidar A the LiDAR model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Lidar& lidar, TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// A parameter getter
  /// The maximum weight that voxels can have. The integrator clips the
  /// voxel weight to this value after integration. Note that currently each
  /// intragration to a voxel increases the weight by 1.0 (if not clipped).
  /// @returns the maximum weight
  float max_weight() const;

  /// A parameter setter
  /// See max_weight().
  /// @param max_weight the maximum of a voxel.
  void max_weight(float max_weight);

  /// A parameter getter
  /// The type of weighting function used to fuse observations
  /// @returns The weighting function type used.
  WeightingFunctionType weighting_function_type() const;

  /// A parameter setter
  /// The type of weighting function used to fuse observations
  /// See weighting_function_type().
  /// @param weighting_function_type The type of weighting function to be used
  void weighting_function_type(WeightingFunctionType weighting_function_type);

  /// A parameter getter.
  /// The distance given to unobserved voxels when
  /// markUnobservedFreeInsideRadius() is called. Note that a negative value
  /// means that the truncation distance will be used.
  /// @return The assigned distance
  float marked_unobserved_voxels_distance_m() const;

  /// A parameter setter
  /// See marked_unobserved_voxels_distance_m()
  /// @param marked_unobserved_voxels_distance_m The assigned distance
  void marked_unobserved_voxels_distance_m(
      float marked_unobserved_voxels_distance_m);

  /// A parameter getter
  // The weight given to unobserved voxels when markUnobservedFreeInsideRadius()
  // is called.
  /// @returns The assigned weight
  float marked_unobserved_voxels_weight() const;

  /// A parameter setter
  /// See marked_unobserved_voxels_weight()
  /// @param marked_unobserved_voxels_weight The assigned weight
  void marked_unobserved_voxels_weight(float marked_unobserved_voxels_weight);

  /// For voxels with a radius, allocate memory and give a small weight and
  /// truncation distance, effectively making these voxels free-space. Does not
  /// affect voxels which are already observed.
  /// @param center The center of the sphere affected.
  /// @param radius The radius of the sphere affected.
  /// @param layer A pointed to the layer which will be affected by the update.
  /// @param updated_blocks_ptr Optional pointer to a list of blocks affected by
  /// the update.
  void markUnobservedFreeInsideRadius(
      const Vector3f& center, float radius, TsdfLayer* layer,
      std::vector<Index3D>* updated_blocks_ptr = nullptr);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  std::string getIntegratorName() const override;

  // Internally used to move the VoxelUpdateFunctor to the device
  unified_ptr<UpdateTsdfVoxelFunctor> getTsdfUpdateFunctorOnDevice(
      float voxel_size);

  // Functor which defines the voxel update operation.
  unified_ptr<UpdateTsdfVoxelFunctor> update_functor_host_ptr_;

  // The maximum weight that a TsdfVoxel can accumulate.
  float max_weight_ = kDefaultMaxWeight;

  // The type of the weighting function to be applied to observations
  WeightingFunctionType weighting_function_type_ =
      kDefaultWeightingFunctionType;

  // The distance given to unobserved voxels when
  // markUnobservedFreeInsideRadius() is called. Note that a negative value
  // means that the truncation distance will be used.
  float marked_unobserved_voxels_distance_m_ = -1.0;
  // The weight given to unobserved voxels when markUnobservedFreeInsideRadius()
  // is called.
  float marked_unobserved_voxels_weight_ = 0.1f;
};

}  // namespace nvblox
