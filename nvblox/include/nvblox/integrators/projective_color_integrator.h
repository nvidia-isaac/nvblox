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

#include "nvblox/integrators/view_calculator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/rays/sphere_tracer.h"

namespace nvblox {

/// A class performing color intregration
///
/// Integrates color images into color layers. The "projective" describes
/// one type of integration. Namely that voxels in view are projected into the
/// depth image (the alternative being casting rays out from the camera).
class ProjectiveColorIntegrator {
 public:
  ProjectiveColorIntegrator();
  virtual ~ProjectiveColorIntegrator();

  /// Integrates a color image into the passed color layer.
  /// @param color_frame A color image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the color layer associated.
  /// Color integration is only performed on the voxels corresponding to the
  /// truncation band of this layer.
  /// @param color_layer A pointer to the layer into which this image will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const ColorImage& color_frame, const Transform& T_L_C,
                      const Camera& camera, const TsdfLayer& tsdf_layer,
                      ColorLayer* color_layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Returns the sphere tracer used for color integration.
  /// In order to perform color integration from an rgb image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  const SphereTracer& sphere_tracer() const { return sphere_tracer_; }
  /// Returns the object used to calculate the blocks in camera views.
  /// In order to perform color integration from an rgb image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  SphereTracer& sphere_tracer() { return sphere_tracer_; }

  /// A parameter getter
  /// We find a surface on which to integrate color by sphere tracing from the
  /// camera. This is a relatively expensive operation. This parameter controls
  /// how many rays are traced for an image. For example, for a 100px by 100px
  /// image with a subsampling factor of 4, 25x25 rays are traced.
  /// @returns the ray subsampling factor
  int sphere_tracing_ray_subsampling_factor() const;

  /// A parameter setter
  /// See sphere_tracing_ray_subsampling_factor()
  /// @param sphere_tracing_ray_subsampling_factor the ray subsampling factor
  void sphere_tracing_ray_subsampling_factor(
      int sphere_tracing_ray_subsampling_factor);

  /// A parameter getter
  /// The truncation distance parameter associated with this integrator. The
  /// truncation distance also determines the max. distance behind a surface up
  /// until which blocks are integrated.
  /// @returns the truncation distance in voxels
  float truncation_distance_vox() const;

  /// A parameter getter
  /// The maximum weight that voxels can have. The integrator clips the
  /// voxel weight to this value after integration. Note that currently each
  /// integration to a voxel increases the weight by 1.0 (if not clipped).
  /// @returns the maximum weight
  float max_weight() const;

  /// A parameter getter
  /// The maximum distance at which voxels are updated. Voxels beyond this
  /// distance from the camera are not affected by integration.
  /// @returns the maximum intragration distance
  float max_integration_distance_m() const;

  /// A parameter setter
  /// See truncation_distance_vox().
  /// @param truncation_distance_vox the truncation distance in voxels.
  void truncation_distance_vox(float truncation_distance_vox);

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

  /// A parameter setter
  /// See max_integration_distance_m().
  /// @param max_integration_distance_m the maximum intragration distance in
  /// meters.
  void max_integration_distance_m(float max_integration_distance_m);

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

 protected:
  // Given a set of blocks in view (block_indices) perform color updates on all
  // voxels within these blocks on the GPU.
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const ColorImage& color_frame,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    ColorLayer* layer);

  // Takes a list of block indices and returns a subset containing the block
  // indices containing at least on voxel inside the truncation band of the
  // passed TSDF layer.
  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  // Params
  int sphere_tracing_ray_subsampling_factor_ = 4;
  float truncation_distance_vox_ = 4.0f;
  float max_weight_ = 100.0f;
  float max_integration_distance_m_ = 7.0f;
  WeightingFunctionType weighting_function_type_ =
      kDefaultWeightingFunctionType;

  // Frustum calculation.
  mutable ViewCalculator view_calculator_;

  // Object to do ray tracing to generate occlusions
  SphereTracer sphere_tracer_;

  // DepthImage to render synthetic images for occlusions
  DepthImage synthetic_depth_image_;

  // Blocks to integrate on the current call and their indices
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<ColorBlock*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<ColorBlock*> block_ptrs_host_;

  // Buffers for getting blocks in truncation band
  device_vector<const TsdfBlock*> truncation_band_block_ptrs_device_;
  host_vector<const TsdfBlock*> truncation_band_block_ptrs_host_;
  device_vector<bool> block_in_truncation_band_device_;
  host_vector<bool> block_in_truncation_band_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox
