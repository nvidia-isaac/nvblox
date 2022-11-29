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

#include <vector>

#include "nvblox/core/camera.h"
#include "nvblox/core/image.h"
#include "nvblox/core/types.h"
#include "nvblox/integrators/view_calculator.h"

namespace nvblox {

/// A pure-virtual base-class for several integrators.
///
/// Integrators deriving from this base class insert (integrate) image data into
/// a voxel grid. The "projective" in ProjectiveIntegratorBase refers to the
/// fact that the mode of operation is that voxels in the view frustrum are
/// projected into the image in order to associate each voxel with a data pixel.
class ProjectiveIntegratorBase {
 public:
  ProjectiveIntegratorBase(){};
  virtual ~ProjectiveIntegratorBase() {}

  /// A parameter getter
  /// The truncation distance parameter associated with this integrator. Beyond
  /// the truncation distance voxels are assigned the maximum distance (the
  /// truncation distance).
  /// @returns the truncation distance in voxels
  float truncation_distance_vox() const;

  /// A parameter getter
  /// The maximum wieght that voxels can have. The integrator clips the
  /// voxel weight to this value after integration. Note that currently each
  /// intragration to a voxel increases the weight by 1.0 (if not clipped).
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

  /// A parameter setter
  /// See max_weight().
  /// @param max_weight the maximum of a voxel.
  void max_weight(float max_weight);

  /// A parameter setter
  /// See max_integration_distance_m().
  /// @param max_integration_distance_m the maximum intragration distance in
  /// meters.
  void max_integration_distance_m(float max_integration_distance_m);

  /// Blocks until GPU operations are complete
  /// Ensure outstanding operations are finished (relevant for integrators
  /// launching asynchronous work)
  virtual void finish() const = 0;

  /// Returns the object used to calculate the blocks in camera views.
  const ViewCalculator& view_calculator() const;
  /// Returns the object used to calculate the blocks in camera views.
  ViewCalculator& view_calculator();

 protected:
  // Truncation distance in meters for this block size.
  float truncation_distance_m(float block_size) const;

  // Params
  float truncation_distance_vox_ = 4.0f;
  float max_weight_ = 100.0f;
  float max_integration_distance_m_ = 10.0f;

  // Frustum calculation.
  mutable ViewCalculator view_calculator_;
};

}  // namespace nvblox
