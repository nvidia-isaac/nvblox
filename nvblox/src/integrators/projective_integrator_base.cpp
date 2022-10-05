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
#include "nvblox/integrators/projective_integrator_base.h"

#include "nvblox/core/bounding_boxes.h"

namespace nvblox {

float ProjectiveIntegratorBase::truncation_distance_vox() const {
  return truncation_distance_vox_;
}

float ProjectiveIntegratorBase::max_weight() const { return max_weight_; }

float ProjectiveIntegratorBase::max_integration_distance_m() const {
  return max_integration_distance_m_;
}

void ProjectiveIntegratorBase::truncation_distance_vox(
    float truncation_distance_vox) {
  CHECK_GT(truncation_distance_vox, 0.0f);
  truncation_distance_vox_ = truncation_distance_vox;
}

void ProjectiveIntegratorBase::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

void ProjectiveIntegratorBase::max_integration_distance_m(
    float max_integration_distance_m) {
  CHECK_GT(max_integration_distance_m, 0.0f);
  max_integration_distance_m_ = max_integration_distance_m;
}

float ProjectiveIntegratorBase::truncation_distance_m(float block_size) const {
  return truncation_distance_vox_ * block_size /
         VoxelBlock<bool>::kVoxelsPerSide;
}

const ViewCalculator& ProjectiveIntegratorBase::view_calculator() const {
  return view_calculator_;
}
/// Returns the object used to calculate the blocks in camera views.
ViewCalculator& ProjectiveIntegratorBase::view_calculator() {
  return view_calculator_;
}

}  // namespace nvblox
