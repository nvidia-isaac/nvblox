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

#include <Eigen/Core>
#include <cmath>

namespace nvblox {
namespace interpolation {
namespace internal {

// Interpolate a float given a square neighborhood. This is the base function
// called by other overloads.
//
// @param xy_px  Point to interpolate
// @param f00  Top-left pixel
// @param f01  Bottom-left pixel
// @param f10  Top-right pixel
// @param f11  Bottom-right pixel
// @param interpolated Resulting interpolated pixel.
template <typename FloatType>
__host__ __device__ inline void interpolatePixels(
    const Vector2f& xy_px, const FloatType& f00, const FloatType& f01,
    const FloatType& f10, const FloatType& f11, FloatType* interpolated) {
  static_assert(isFloatType<FloatType>(),
                "Only floating point types supported");
  // Formula is obtained by starting from "Interpolation of a grid on with 1
  // pixel spacing" and simplifying. See
  // https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square.
  // Note that the order of operations seems sensitive, esepcially when using
  // half-precision.
  const FloatType x = xy_px.x();
  const FloatType y = xy_px.y();
  const FloatType dx = f10 - f00;
  *interpolated = f00 + x * dx + y * (f01 - f00) + x * y * (f11 - f01 - dx);
}

/// Overload for interpolating color arrays
__host__ __device__ inline void interpolatePixels(
    const Vector2f& xy_px, const Color& c00, const Color& c01, const Color& c10,
    const Color& c11, Color* interpolated) {
  for (size_t i = 0; i < c00.size(); ++i) {
    float interpolated_float;
    interpolatePixels<float>(xy_px, c00[i], c01[i], c10[i], c11[i],
                             &interpolated_float);
    (*interpolated)[i] = static_cast<uint8_t>(std::round(interpolated_float));
  }
}

/// Overload for interpolating float arrays
template <typename FloatType, size_t NumElements>
__host__ __device__ inline void interpolatePixels(
    const Vector2f& xy_px, const Array<FloatType, NumElements>& c00,
    const Array<FloatType, NumElements>& c01,
    const Array<FloatType, NumElements>& c10,
    const Array<FloatType, NumElements>& c11,
    Array<FloatType, NumElements>* interpolated) {
  for (size_t i = 0; i < NumElements; ++i) {
    interpolatePixels(xy_px, c00[i], c01[i], c10[i], c11[i],
                      interpolated->data() + i);
  }
}

// Return false if any of the pixels fail the validity check.
template <typename PixelValidityChecker, typename ElementType>
__device__ inline bool neighboursValid(const ElementType& p00,
                                       const ElementType& p01,
                                       const ElementType& p10,
                                       const ElementType& p11) {
  return PixelValidityChecker::check(p00) && PixelValidityChecker::check(p01) &&
         PixelValidityChecker::check(p10) && PixelValidityChecker::check(p11);
}

}  //  namespace internal

namespace checkers {

template <typename ElementType>
struct PixelAlwaysValid {
  __host__ __device__ constexpr static inline bool check(const ElementType&) {
    return true;
  }
};

/// Checker that validates pixel is finite (not NaN/inf) and positive.
struct PixelIsValidDepth {
  __host__ __device__ static inline bool check(const float& pixel_value) {
    constexpr float kEps = 1e-6;
    return std::isfinite(pixel_value) && pixel_value > kEps;
  }
};

}  // namespace checkers

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2D(const ImageView<const ElementType>& frame,
                   const Vector2f& u_px, ElementType* value_interpolated_ptr,
                   const InterpolationType type) {
  if (type == InterpolationType::kNearestNeighbor) {
    return interpolate2DClosest<ElementType, PixelValidityChecker>(
        frame, u_px, value_interpolated_ptr);
  }
  if (type == InterpolationType::kLinear) {
    return interpolate2DLinear<ElementType, PixelValidityChecker>(
        frame, u_px, value_interpolated_ptr);
  } else {
    NVBLOX_ABORT("Requested interpolation method is not implemented.");
  }
  return 0.0;
}

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DClosest(const ImageView<const ElementType> frame,
                          const Vector2f& u_px,
                          ElementType* value_interpolated_ptr,
                          Index2D* u_px_closest_ptr) {
  // Closest pixel
  const Index2D u_M = u_px.array().floor().cast<int>();
  // Check bounds:
  if (u_M.x() < 0 || u_M.y() < 0 || u_M.x() >= frame.cols() ||
      u_M.y() >= frame.rows()) {
    return false;
  }
  // "Interpolate"
  const ElementType pixel_value = frame(u_M.y(), u_M.x());
  // Check result for validity
  if (!PixelValidityChecker::check(pixel_value)) {
    return false;
  }

  *value_interpolated_ptr = pixel_value;

  if (u_px_closest_ptr) {
    *u_px_closest_ptr = u_M;
  }
  return true;
}

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DLinear(
    const ImageView<const ElementType> frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr,
    Interpolation2DNeighbours<ElementType>* neighbours_ptr) {
  const Vector2f u_center_referenced_px = u_px - Vector2f(0.5, 0.5);
  // Get the pixel index of the pixel on the low side (which is
  // also the image plane location of the pixel center).
  const Index2D u_low_side_px =
      Index2D(static_cast<int>(floorf(u_center_referenced_px.x())),
              static_cast<int>(floorf(u_center_referenced_px.y())));
  // If we're gonna access out of bounds, fail.
  if ((u_low_side_px.array() < 0).any() ||
      ((u_low_side_px.x() + 1) > (frame.cols() - 1)) ||
      ((u_low_side_px.y() + 1) > (frame.rows() - 1))) {
    return false;
  }

  // Access the image (in global GPU memory)
  const ElementType& p00 = frame(u_low_side_px.y(), u_low_side_px.x());
  const ElementType& p01 = frame(u_low_side_px.y() + 1, u_low_side_px.x());
  const ElementType& p10 = frame(u_low_side_px.y(), u_low_side_px.x() + 1);
  const ElementType& p11 = frame(u_low_side_px.y() + 1, u_low_side_px.x() + 1);

  // Validate the pixels
  if (!internal::neighboursValid<PixelValidityChecker>(p00, p01, p10, p11)) {
    return false;
  }

  // Offset of the requested point to the low side center.
  const Eigen::Vector2f u_offset =
      (u_center_referenced_px - u_low_side_px.cast<float>());

  // Do the interpolation.
  internal::interpolatePixels(u_offset, p00, p01, p10, p11,
                              value_interpolated_ptr);

  // Optionally populate the neighborhood info.
  if (neighbours_ptr != nullptr) {
    neighbours_ptr->p00 = p00;
    neighbours_ptr->p01 = p01;
    neighbours_ptr->p10 = p10;
    neighbours_ptr->p11 = p11;
    neighbours_ptr->u_low_side_px = u_low_side_px;
  }

  return true;
}

}  // namespace interpolation
}  // namespace nvblox
