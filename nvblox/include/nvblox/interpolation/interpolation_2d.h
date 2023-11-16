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

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {
namespace interpolation {

namespace checkers {

/// A checker that always returns that a pixel is valid (the default below)
template <typename ElementType>
struct PixelAlwaysValid;

/// A checker that returns true if a float pixel is greater than 0.0f.
struct FloatPixelGreaterThanZero;

/// A checker that returns true if the alpha channel of a color pixel is greater
/// than 0.
struct ColorPixelAlphaGreaterThanZero;

}  // namespace checkers

template <typename ElementType>
struct Interpolation2DNeighbours {
  ElementType p00;
  ElementType p01;
  ElementType p10;
  ElementType p11;
  Index2D u_low_side_px;
};

// CPU interfaces
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ inline bool interpolate2DClosest(const Image<ElementType>& frame,
                                          const Vector2f& u_px,
                                          ElementType* value_interpolated_ptr);
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ inline bool interpolate2DLinear(const Image<ElementType>& frame,
                                         const Vector2f& u_px,
                                         ElementType* value_interpolated_ptr);
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ bool interpolate2D(
    const Image<ElementType>& frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr,
    const InterpolationType type = InterpolationType::kLinear);

// CPU/GPU interfaces
// NOTE(alexmillane): These function do the interpolation. They're called
// indirectly on the CPU through the interfaces above, and called directly on
// the GPU.
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ __device__ inline bool interpolate2DClosest(
    const ElementType* frame, const Vector2f& u_px, const int rows,
    const int cols, ElementType* value_interpolated_ptr,
    Index2D* u_px_closest_ptr = nullptr);

template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ __device__ inline bool interpolate2DLinear(
    const ElementType* frame, const Vector2f& u_px, const int rows,
    const int cols, ElementType* value_interpolated_ptr,
    Interpolation2DNeighbours<ElementType>* neighbours_ptr = nullptr);

// LiDAR GPU interpolation
__device__ inline bool interpolateLidarImage(
    const Lidar& lidar, const Vector3f& p_voxel_center_C, const float* image,
    const Vector2f& u_px, const int rows, const int cols,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    float* image_value);

}  // namespace interpolation
}  // namespace nvblox

#include "nvblox/interpolation/internal/impl/interpolation_2d_impl.h"
