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

#include <optional>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/integrators/internal/projective_integrator.h"
#include "nvblox/integrators/projective_integrator_params.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/internal/image_cache.h"

namespace nvblox {

template <typename VoxelType>
struct UpdateAppearanceVoxelFunctor;
/// A class performing intregration of appearance images (e.g. RGB or general
/// features)
///
/// The "projective" describes one type of integration. Namely that voxels in
/// view are projected into the depth image (the alternative being casting rays
/// out from the camera).
template <class _LayerType>
class ProjectiveAppearanceIntegrator
    : public ProjectiveIntegrator<typename _LayerType::VoxelType> {
 public:
  using MaskedImageType =
      MaskedImageView<const typename _LayerType::VoxelType::ArrayType>;
  using LayerType = _LayerType;
  using VoxelType = typename _LayerType::VoxelType;
  using BlockType = typename _LayerType::BlockType;
  using ImageElementType = typename MaskedImageType::ElementType;

  ProjectiveAppearanceIntegrator();
  ProjectiveAppearanceIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~ProjectiveAppearanceIntegrator();

  /// Integrates an appearance image into the passed layer.
  ///
  /// When depth_image is provided, it is used directly for occlusion testing
  /// (e.g., from an RGBD camera with aligned depth). Otherwise, a synthetic
  /// depth image is generated via sphere tracing.
  ///
  /// @param image An appearance image (color or features)
  /// @param depth_image Optional aligned depth image for occlusion testing.
  /// When absent, a synthetic depth image is generated via sphere tracing.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera The camera (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the appearance layer is
  /// associated. Integration is only performed on the voxels corresponding to
  /// the truncation band of this layer.
  /// @param layer A pointer to the layer into which this image will be
  /// integrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const MaskedImageType& image,
                      std::optional<MaskedDepthImageConstView> depth_image,
                      const Transform& T_L_C, const Camera& camera,
                      const TsdfLayer& tsdf_layer, LayerType* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Convenience overload without a depth image (uses sphere tracing).
  void integrateFrame(const MaskedImageType& image, const Transform& T_L_C,
                      const Camera& camera, const TsdfLayer& tsdf_layer,
                      LayerType* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Returns the sphere tracer used for appearance integration.
  /// In order to perform integration from an appearance image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  const SphereTracer& sphere_tracer() const { return sphere_tracer_; }
  /// Returns the object used to calculate the blocks in camera views.
  /// In order to perform integration from an appearance image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  SphereTracer& sphere_tracer() { return sphere_tracer_; }

  /// A parameter getter
  /// We find a surface on which to integrate appearance by sphere tracing from
  /// the camera. This is a relatively expensive operation. This parameter
  /// controls how many rays are traced for an image. For example, for a 100px
  /// by 100px image with a subsampling factor of 4, 25x25 rays are traced.
  /// @returns the ray subsampling factor
  int sphere_tracing_ray_subsampling_factor() const;

  /// A parameter setter
  /// See sphere_tracing_ray_subsampling_factor()
  /// @param sphere_tracing_ray_subsampling_factor the ray subsampling factor
  void sphere_tracing_ray_subsampling_factor(
      int sphere_tracing_ray_subsampling_factor);

  /// A parameter getter
  /// The maximum weight that voxels can have. The integrator clips the
  /// voxel weight to this value after integration. Note that currently each
  /// integration to a voxel increases the weight by 1.0 (if not clipped).
  /// @returns the maximum weight
  float max_weight() const;

  /// Gets the metric truncation distance which is calculated from truncation
  /// distance in voxels and the input voxel size.
  /// @param voxel_size The voxel size of the layer you want the truncation
  /// distance for.
  /// @return The truncation distance
  float get_truncation_distance_m(float voxel_size) const;

  /// A parameter setter
  /// See max_weight().
  /// @param max_weight the maximum weight of a voxel.
  void max_weight(float max_weight);

  /// A parameter getter
  /// Get the weight for the exponential filter used when fusing appearance
  /// vectors.
  /// @return The measurement weight
  float measurement_weight() const;

  /// A parameter setter
  /// See measurement_weight()
  /// @param measurement_weight The measurement weight
  void measurement_weight(float measurement_weight);

  /// A parameter getter
  /// The type of weighting function used to fuse observations
  /// @returns The weighting function type used.
  WeightingFunctionType weighting_function_type() const;

  /// A parameter setter
  /// The type of weighting function used to fuse observations
  /// See weighting_function_type().
  /// @param weighting_function_type The type of weighting function to be used
  void weighting_function_type(WeightingFunctionType weighting_function_type);

  /// Returns the object used to calculate the blocks in camera views.
  const ViewCalculator& view_calculator() const;
  /// Returns the object used to calculate the blocks in camera views.
  ViewCalculator& view_calculator();

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  std::string getIntegratorName() const override;

  // Takes a list of block indices and returns a subset containing the block
  // indices containing at least on voxel inside the truncation band of the
  // passed TSDF layer.
  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  unified_ptr<UpdateAppearanceVoxelFunctor<VoxelType>>
  getAppearanceUpdateFunctorOnDevice(float voxel_size);

  // Functor which defines the voxel update operation.
  unified_ptr<UpdateAppearanceVoxelFunctor<VoxelType>> update_functor_host_ptr_;

  // Params
  int sphere_tracing_ray_subsampling_factor_ = 4;
  float max_weight_ = kProjectiveIntegratorMaxWeightParamDesc.default_value;
  WeightingFunctionType weighting_function_type_ =
      kProjectiveIntegratorWeightingModeParamDesc.default_value;
  float measurement_weight_ =
      kProjectiveAppearanceIntegratorMeasurementWeightParamDesc.default_value;

  // Frustum calculation.
  mutable ViewCalculator view_calculator_;

  // Object to do ray tracing to generate occlusions
  SphereTracer sphere_tracer_;

  // DepthImages to render synthetic images for occlusions
  ImageCache<DepthImage> synthetic_depth_images_;

  // Buffers for getting blocks in truncation band
  device_vector<const TsdfBlock*> truncation_band_block_ptrs_device_;
  host_vector<const TsdfBlock*> truncation_band_block_ptrs_host_;
  device_vector<bool> block_in_truncation_band_device_;
  host_vector<bool> block_in_truncation_band_host_;
};

///  Integrator for color images
using ProjectiveColorIntegrator = ProjectiveAppearanceIntegrator<ColorLayer>;

///  Integrator for general-purpose feature vectors
using ProjectiveFeatureIntegrator =
    ProjectiveAppearanceIntegrator<FeatureLayer>;

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/projective_appearance_integrator_impl.h"
