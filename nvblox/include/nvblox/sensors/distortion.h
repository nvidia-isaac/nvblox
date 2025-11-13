/*
Copyright 2025 NVIDIA CORPORATION

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

#include <cuda_runtime.h>
#include "nvblox/core/types.h"

namespace nvblox {

// Parameters for radial distortion:
// p_distorted = p * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 *
// r^4 + k6 * r^6)
struct RadialDistortionParams {
  float k1 = 0.F;  // First numerator coefficient
  float k2 = 0.F;  // Second numerator coefficient
  float k3 = 0.F;  // Third numerator coefficient
  float k4 = 0.F;  // First denominator coefficient
  float k5 = 0.F;  // Second denominator coefficient
  float k6 = 0.F;  // Third denominator coefficient

  __host__ __device__ inline bool operator==(
      const RadialDistortionParams& other) const {
    return std::abs(k1 - other.k1) <= 1e-6 && std::abs(k2 - other.k2) <= 1e-6 &&
           std::abs(k3 - other.k3) <= 1e-6 && std::abs(k4 - other.k4) <= 1e-6 &&
           std::abs(k5 - other.k5) <= 1e-6 && std::abs(k6 - other.k6) <= 1e-6;
  }
};

// Parameters for tangential distortion:
// p_distorted_x = 2*p1*x*y + p2*(r^2 + 2*x^2)
// p_distorted_y = 2*p2*x*y + p1*(r^2 + 2*y^2)
struct TangentialDistortionParams {
  float p1 = 0.F;  // First tangential distortion coefficient
  float p2 = 0.F;  // Second tangential distortion coefficient

  __host__ __device__ inline bool operator==(
      const TangentialDistortionParams& other) const {
    return std::abs(p1 - other.p1) <= 1e-6 && std::abs(p2 - other.p2) <= 1e-6;
  }
};

// Distortion parameters for the Brown / Conrady model that combines radial and
// tangential distortion. See https://en.wikipedia.org/wiki/Distortion_(optics)
struct RadialTangentialDistortionParams {
  RadialDistortionParams radial;
  TangentialDistortionParams tangential;

  __host__ __device__ inline bool operator==(
      const RadialTangentialDistortionParams& other) const {
    return radial == other.radial && tangential == other.tangential;
  }
};

/// Apply distortion
/// @param u_norm Undistorted normalized coordinates
/// @param distortion_params Distortion parameters
/// @return Distorted normalized coordinates
__host__ __device__ inline Vector2f applyDistortion(
    const Vector2f& u_norm,
    const RadialTangentialDistortionParams& distortion_params);

/// Remove distortion
/// Note that this function uses an iterative method that can be slow.
/// @param u_dist Distorted normalized coordinates
/// @param distortion_params Distortion parameters
/// @return Undistorted normalized coordinates
__host__ __device__ inline Vector2f removeDistortion(
    const Vector2f& u_dist,
    const RadialTangentialDistortionParams& distortion_params);

}  // namespace nvblox

#include "nvblox/sensors/internal/impl/distortion_impl.h"
