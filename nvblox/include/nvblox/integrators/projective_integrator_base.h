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
#include "nvblox/integrators/frustum.h"

namespace nvblox {

class ProjectiveIntegratorBase {
 public:
  ProjectiveIntegratorBase(){};
  virtual ~ProjectiveIntegratorBase() {}

  // Getters
  float truncation_distance_vox() const;
  float max_weight() const;
  float max_integration_distance_m() const;

  // Setters
  void truncation_distance_vox(float truncation_distance_vox);
  void max_weight(float max_weight);
  void max_integration_distance_m(float max_integration_distance_m);

  // Scale dependent getter
  float truncation_distance_m(float block_size) const;

  // Gets blocks in view (called by sub-classes)
  // Using the image:
  std::vector<Index3D> getBlocksInView(const DepthImage& depth_frame,
                                       const Transform& T_L_C,
                                       const Camera& camera,
                                       float block_size) const;
  // Does not use the image, just uses the AABB.
  std::vector<Index3D> getBlocksInView(const Transform& T_L_C,
                                       const Camera& camera,
                                       float block_size) const;
  // Using the image and raycasting:
  std::vector<Index3D> getBlocksInViewUsingRaycasting(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float block_size) const;

  // Ensure outstanding operations are finished (relevant for integrators
  // launching asynchronous work)
  virtual void finish() const = 0;

  const FrustumCalculator& frustum_calculator() const {
    return frustum_calculator_;
  }
  FrustumCalculator& frustum_calculator() { return frustum_calculator_; }

 protected:
  // Params
  float truncation_distance_vox_ = 4.0f;
  float max_weight_ = 100.0f;
  float max_integration_distance_m_ = 10.0f;

  // Frustum calculation.
  mutable FrustumCalculator frustum_calculator_;
};

}  // namespace nvblox
