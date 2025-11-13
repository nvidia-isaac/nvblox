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

namespace nvblox {
namespace interpolation {

namespace checkers {

/// A checker that always returns that a pixel is valid (the default below)
template <typename ElementType>
struct PixelAlwaysValid;

/// A checker that returns true if a float pixel is finite (not NaN/inf) and
/// positive. This is useful for depth values which should be valid finite
/// numbers > 0.
struct PixelIsValidDepth;

}  // namespace checkers

/// Information about the interpolation neighborhood
template <typename ElementType>
struct Interpolation2DNeighbours {
  /// Top-left corner
  ElementType p00;

  /// Bottom-left corner
  ElementType p01;

  /// Top-right corner
  ElementType p10;

  /// Bottom-right corner
  ElementType p11;

  /// Image coordinate of the top-left corner
  Index2D u_low_side_px;
};

/// Interpolate values from an image
///
/// @param frame Input image.
/// @param u_px Location of pixel to interpolate.
/// @param value_interpolated_ptr Resulting interpolated value.
/// @param type Type of interpolation to perform.
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ __device__ bool interpolate2D(
    const ImageView<const ElementType>& frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr, const InterpolationType type);

/// Nearest neighbor
///
/// @param frame Input image.
/// @param u_px Location of pixel to interpolate.
/// @param value_interpolated_ptr Resulting interpolated value.
/// @param u_px_closest_ptr Optional index of the nearest neighbor
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ __device__ inline bool interpolate2DClosest(
    const ImageView<const ElementType> frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr, Index2D* u_px_closest_ptr = nullptr);

/// Bilinear interpolation
///
/// @param frame Input image.
/// @param u_px Location of pixel to interpolate.
/// @param value_interpolated_ptr Resulting interpolated value.
/// @param neighbours_ptr Optional neighborhood information.
template <typename ElementType, typename PixelValidityChecker =
                                    checkers::PixelAlwaysValid<ElementType>>
__host__ __device__ inline bool interpolate2DLinear(
    const ImageView<const ElementType> frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr,
    Interpolation2DNeighbours<ElementType>* neighbours_ptr = nullptr);

}  // namespace interpolation
}  // namespace nvblox

#include "nvblox/interpolation/internal/impl/interpolation_2d_impl.h"
