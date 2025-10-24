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
#include "nvblox/core/parameter_tree.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/projective_integrator_params.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"
namespace nvblox {

/// A pure-virtual base-class for the projective occupancy and tsdf integrators.
///
/// Integrators deriving from this base class insert (integrate) image and lidar
/// data into a voxel grid. The "projective" in ProjectiveIntegratorBase refers
/// to the fact that the mode of operation is that voxels in the view frustrum
/// are projected into the image in order to associate each voxel with a data
/// pixel.
template <typename VoxelType>
class ProjectiveIntegrator {
 public:
  ProjectiveIntegrator();
  ProjectiveIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~ProjectiveIntegrator() = default;

  /// Update a generic layer using depth image
  template <typename UpdateFunctor, typename SensorType>
  void integrateFrame(const MaskedDepthImageConstView& depth_frame,
                      const Transform& T_L_C, const SensorType& sensor,
                      UpdateFunctor* op, VoxelBlockLayer<VoxelType>* layer,
                      std::vector<Index3D>* updated_blocks);

  /// A parameter getter
  /// The truncation distance parameter associated with this integrator. The
  /// truncation distance also determines the max. distance behind a surface up
  /// until which blocks are integrated.
  /// @returns the truncation distance in voxels
  float truncation_distance_vox() const;

  /// Gets the metric truncation distance which is calculated from truncation
  /// distance in voxels and the input voxel size.
  /// @param voxel_size The voxel size of the layer you want the truncation
  /// distance for.
  /// @return The truncation distance
  float get_truncation_distance_m(float voxel_size) const;

  /// A parameter getter
  /// The maximum distance at which voxels are updated. Voxels beyond this
  /// distance from the camera are not affected by integration.
  /// @returns the maximum integration distance
  float max_integration_distance_m() const;

  /// A parameter setter
  /// See truncation_distance_vox().
  /// @param truncation_distance_vox the truncation distance in voxels.
  void truncation_distance_vox(float truncation_distance_vox);

  /// A parameter setter
  /// See max_integration_distance_m().
  /// @param max_integration_distance_m the maximum integration distance in
  /// meters.
  void max_integration_distance_m(float max_integration_distance_m);

  /// Returns the object used to calculate the blocks in camera views.
  const ViewCalculator& view_calculator() const;
  /// Returns the object used to calculate the blocks in camera views.
  ViewCalculator& view_calculator();

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  /// For voxels with a radius, allocate memory and give a small weight and
  /// truncation distance, effectively making these voxels free-space. Does not
  /// affect voxels which are already observed.
  /// @param center The center of the sphere affected.
  /// @param radius The radius of the sphere affected.
  /// @param layer A pointed to the layer which will be affected by the update.
  /// @param updated_blocks Optional pointer to a list of blocks affected by the
  /// update.
  void markUnobservedFreeInsideRadiusTemplate(
      const Vector3f& center, float radius, VoxelBlockLayer<VoxelType>* layer,
      std::vector<Index3D>* updated_blocks = nullptr);

  // Called from the integrateFrame() interfaces.
  // Captures common behaviour between sensors.
  template <typename SensorType, typename UpdateFunctor>
  void integrateFrameTemplate(const MaskedDepthImageConstView& depth_frame,
                              const MaskedColorImageConstView& color_frame,
                              const Transform& T_L_C, const SensorType& sensor,
                              UpdateFunctor* op,
                              VoxelBlockLayer<VoxelType>* layer,
                              std::vector<Index3D>* updated_blocks = nullptr);

  // Methods below are specialized for different combinations of depth sensor +
  // appearance image type.

  // Depth integration
  template <typename UpdateFunctor, typename SensorType>
  void integrateBlocks(const MaskedDepthImageConstView& depth_frame,
                       const MaskedColorImageConstView&, /*unused*/
                       const Transform& T_C_L, const SensorType& sensor,
                       UpdateFunctor* op,
                       VoxelBlockLayer<VoxelType>* layer_ptr);

  // Depth camera + feature image
  template <typename UpdateFunctor, typename SensorType>
  void integrateBlocks(const MaskedDepthImageConstView& depth_frame,
                       const MaskedFeatureImageConstView& feature_frame,
                       const Transform& T_C_L, const SensorType& sensor,
                       UpdateFunctor* op,
                       VoxelBlockLayer<VoxelType>* layer_ptr);
  // Get the child integrator name
  virtual std::string getIntegratorName() const = 0;
  bool integrator_name_initialized_ = false;
  std::string integrator_name_;

  // Params
  // NOTE(alexmillane): See the getters above for a description.
  float truncation_distance_vox_ =
      kProjectiveIntegratorTruncationDistanceVoxParamDesc.default_value;
  float max_integration_distance_m_ =
      kProjectiveIntegratorMaxIntegrationDistanceMParamDesc.default_value;

  // Frustum calculation.
  mutable ViewCalculator view_calculator_;

  // Blocks to integrate on the current call (indices and pointers)
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<VoxelBlock<VoxelType>*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<VoxelBlock<VoxelType>*> block_ptrs_host_;

  // CUDA stream to process integration on
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/projective_integrator_impl.h"
