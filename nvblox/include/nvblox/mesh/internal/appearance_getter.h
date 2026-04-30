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

#include <cassert>

#include <cuda_runtime.h>

#include "nvblox/core/color.h"
#include "nvblox/core/feature_array.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

/// Trait to extract appearance values from voxel types.
/// Specialize for each supported AppearanceVoxelType.
template <typename AppearanceVoxelType>
struct AppearanceGetter {
  __host__ __device__ static typename AppearanceVoxelType::ArrayType
  getAppearance(const AppearanceVoxelType& voxel) {
    // AppearanceGetter not implemented for this voxel type
    assert(false);
  }

  __host__ __device__ static typename AppearanceVoxelType::ArrayType
  getDefaultAppearance() {
    // AppearanceGetter not implemented for this voxel type
    assert(false);
  }
};

template <>
struct AppearanceGetter<ColorVoxel> {
  __host__ __device__ static Color getAppearance(const ColorVoxel& voxel) {
    return voxel.color;
  }

  __host__ __device__ static Color getDefaultAppearance() {
    // The color that the mesh takes if no coloring is available.
    return Color::Gray();
  }
};

template <>
struct AppearanceGetter<FeatureVoxel> {
  __host__ __device__ static FeatureArray getAppearance(
      const FeatureVoxel& voxel) {
    return voxel.feature;
  }

  __host__ __device__ static FeatureArray getDefaultAppearance() {
    // The feature that the mesh takes if no features are available.
    return FeatureArray();
  }
};

}  // namespace nvblox
