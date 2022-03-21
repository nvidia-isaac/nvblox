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

namespace nvblox {
namespace interpolation {
namespace internal {

__host__ __device__ inline float interpolatePixels(const Vector2f& xy_px,
                                                   const float& f00,
                                                   const float& f01,
                                                   const float& f10,
                                                   const float& f11) {
  // Interpolation of a grid on with 1 pixel spacing.
  // https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square
  const Eigen::Matrix2f value_matrix =
      (Eigen::Matrix2f() << f00, f01, f10, f11).finished();
  const Eigen::Vector2f x_vec(1.0f - xy_px.x(), xy_px.x());
  const Eigen::Vector2f y_vec(1.0f - xy_px.y(), xy_px.y());
  return x_vec.transpose() * value_matrix * y_vec;
}

__host__ __device__ inline Color interpolatePixels(const Vector2f& xy_px,
                                                   const Color& c00,
                                                   const Color& c01,
                                                   const Color& c10,
                                                   const Color& c11) {
  // We define interpolating two colors as independantly interpolating their
  // channels.
  return Color(
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.r, c01.r, c10.r, c11.r))),
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.g, c01.g, c10.g, c11.g))),
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.b, c01.b, c10.b, c11.b))));
}

}  //  namespace internal

template <typename ElementType>
bool interpolate2D(const Image<ElementType>& frame, const Vector2f& u_px,
                   ElementType* value_interpolated_ptr,
                   const InterpolationType type) {
  if (type == InterpolationType::kNearestNeighbor) {
    return interpolate2DClosest(frame, u_px, value_interpolated_ptr);
  }
  if (type == InterpolationType::kLinear) {
    return interpolate2DLinear(frame, u_px, value_interpolated_ptr);
  } else {
    CHECK(false) << "Requested interpolation method is not implemented.";
  }
  return 0.0;
}

template <typename ElementType>
bool interpolate2DClosest(const Image<ElementType>& frame, const Vector2f& u_px,
                          ElementType* value_interpolated_ptr) {
  return interpolate2DClosest(frame.dataConstPtr(), u_px, frame.rows(),
                              frame.cols(), value_interpolated_ptr);
}

template <typename ElementType>
bool interpolate2DClosest(const ElementType* frame, const Vector2f& u_px,
                          const int rows, const int cols,
                          ElementType* value_interpolated_ptr) {
  // Closest pixel
  const Index2D u_M_rounded = u_px.array().round().cast<int>();
  // Check bounds:
  if (u_M_rounded.x() < 0 || u_M_rounded.y() < 0 || u_M_rounded.x() >= cols ||
      u_M_rounded.y() >= rows) {
    return false;
  }
  // "Interpolate"
  *value_interpolated_ptr =
      image::access(u_M_rounded.y(), u_M_rounded.x(), cols, frame);
  return true;
}

template <typename ElementType>
bool interpolate2DLinear(const Image<ElementType>& frame, const Vector2f& u_px,
                         ElementType* value_interpolated_ptr) {
  return interpolate2DLinear(frame.dataConstPtr(), u_px, frame.rows(),
                             frame.cols(), value_interpolated_ptr);
}

template <typename ElementType>
bool interpolate2DLinear(const ElementType* frame, const Vector2f& u_px,
                         const int rows, const int cols,
                         ElementType* value_interpolated_ptr) {
  // Subtraction of Vector2f(0.5, 0.5) takes our coordinates from
  // corner-referenced to center-referenced.
  const Vector2f u_center_referenced_px = u_px - Vector2f(0.5, 0.5);
  // Get the pixel index of the pixel on the low side (which is also the image
  // plane location of the pixel center).
  const Index2D u_low_side_px = u_center_referenced_px.cast<int>();
  // If we're gonna access out of bounds, fail.
  if ((u_low_side_px.array() < 0).any() ||
      ((u_low_side_px.x() + 1) > (cols - 1)) ||
      ((u_low_side_px.y() + 1) > (rows - 1))) {
    return false;
  }
  // Offset of the requested point to the low side center.
  const Eigen::Vector2f u_offset =
      (u_center_referenced_px - u_low_side_px.cast<float>());
  // clang-format off
  *value_interpolated_ptr = internal::interpolatePixels(
      u_offset,
      image::access(u_low_side_px.y(), u_low_side_px.x(), cols, frame),
      image::access(u_low_side_px.y() + 1, u_low_side_px.x(), cols, frame),
      image::access(u_low_side_px.y(), u_low_side_px.x() + 1, cols, frame),
      image::access(u_low_side_px.y() + 1, u_low_side_px.x() + 1, cols, frame));
  // clang-format on
  return true;
}

}  // namespace interpolation
}  // namespace nvblox
