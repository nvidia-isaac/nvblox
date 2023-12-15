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
  // We define interpolating two colors as independently interpolating their
  // channels.
  return Color(
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.r, c01.r, c10.r, c11.r))),
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.g, c01.g, c10.g, c11.g))),
      static_cast<uint8_t>(
          std::round(interpolatePixels(xy_px, c00.b, c01.b, c10.b, c11.b))));
}

template <typename ElementType, typename PixelValidityChecker>
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
  __host__ __device__ static inline bool check(const ElementType&) {
    return true;
  }
};

struct FloatPixelGreaterThanZero {
  __host__ __device__ static inline bool check(const float& pixel_value) {
    constexpr float kEps = 1e-6;
    return pixel_value > kEps;
  }
};

struct ColorPixelAlphaGreaterThanZero {
  __host__ __device__ static inline bool check(const Color& pixel_value) {
    return pixel_value.a > 0;
  }
};

}  // namespace checkers

// CPU interfaces

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2D(const Image<ElementType>& frame, const Vector2f& u_px,
                   ElementType* value_interpolated_ptr,
                   const InterpolationType type) {
  if (type == InterpolationType::kNearestNeighbor) {
    return interpolate2DClosest<ElementType, PixelValidityChecker>(
        frame, u_px, value_interpolated_ptr);
  }
  if (type == InterpolationType::kLinear) {
    return interpolate2DLinear<ElementType, PixelValidityChecker>(
        frame, u_px, value_interpolated_ptr);
  } else {
    CHECK(false) << "Requested interpolation method is not implemented.";
  }
  return 0.0;
}

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DClosest(const Image<ElementType>& frame, const Vector2f& u_px,
                          ElementType* value_interpolated_ptr) {
  return interpolate2DClosest<ElementType, PixelValidityChecker>(
      frame.dataConstPtr(), u_px, frame.rows(), frame.cols(),
      value_interpolated_ptr);
}

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DLinear(const Image<ElementType>& frame, const Vector2f& u_px,
                         ElementType* value_interpolated_ptr) {
  return interpolate2DLinear<ElementType, PixelValidityChecker>(
      frame.dataConstPtr(), u_px, frame.rows(), frame.cols(),
      value_interpolated_ptr);
}

// CPU/GPU interfaces
// NOTE(alexmillane): These function do the interpolation. They're called
// indirectly on the CPU through the interfaces above, and called directly on
// the GPU.
template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DClosest(const ElementType* frame, const Vector2f& u_px,
                          const int rows, const int cols,
                          ElementType* value_interpolated_ptr,
                          Index2D* u_px_closest_ptr) {
  // Closest pixel
  const Index2D u_M_rounded = u_px.array().round().cast<int>();
  // Check bounds:
  if (u_M_rounded.x() < 0 || u_M_rounded.y() < 0 || u_M_rounded.x() >= cols ||
      u_M_rounded.y() >= rows) {
    return false;
  }
  // "Interpolate"
  const ElementType pixel_value =
      image::access(u_M_rounded.y(), u_M_rounded.x(), cols, frame);
  // Check result for validity
  if (!PixelValidityChecker::check(pixel_value)) {
    return false;
  }
  *value_interpolated_ptr = pixel_value;
  if (u_px_closest_ptr) {
    *u_px_closest_ptr = u_M_rounded;
  }
  return true;
}

template <typename ElementType, typename PixelValidityChecker>
bool interpolate2DLinear(
    const ElementType* frame, const Vector2f& u_px, const int rows,
    const int cols, ElementType* value_interpolated_ptr,
    Interpolation2DNeighbours<ElementType>* neighbours_ptr) {
  // Subtraction of Vector2f(0.5, 0.5) takes our coordinates from
  // corner-referenced to center-referenced.
  const Vector2f u_center_referenced_px = u_px - Vector2f(0.5, 0.5);
  // Get the pixel index of the pixel on the low side (which is also the image
  // plane location of the pixel center).
  const Index2D u_low_side_px =
      Index2D(static_cast<int>(floorf(u_center_referenced_px.x())),
              static_cast<int>(floorf(u_center_referenced_px.y())));
  // If we're gonna access out of bounds, fail.
  if ((u_low_side_px.array() < 0).any() ||
      ((u_low_side_px.x() + 1) > (cols - 1)) ||
      ((u_low_side_px.y() + 1) > (rows - 1))) {
    return false;
  }
  // Access the image (in global GPU memory)
  // clang-format off
  const ElementType& p00 = image::access(u_low_side_px.y(), u_low_side_px.x(), cols, frame);
  const ElementType& p01 = image::access(u_low_side_px.y() + 1, u_low_side_px.x(), cols, frame);
  const ElementType& p10 = image::access(u_low_side_px.y(), u_low_side_px.x() + 1, cols, frame);
  const ElementType& p11 = image::access(u_low_side_px.y() + 1, u_low_side_px.x() + 1, cols, frame);
  // clang-format on

  // Do 2D interpolation if all neighbours valid
  if (!internal::neighboursValid<ElementType, PixelValidityChecker>(p00, p01,
                                                                    p10, p11)) {
    return false;
  }

  // Offset of the requested point to the low side center.
  const Eigen::Vector2f u_offset =
      (u_center_referenced_px - u_low_side_px.cast<float>());
  *value_interpolated_ptr =
      internal::interpolatePixels(u_offset, p00, p01, p10, p11);
  if (neighbours_ptr) {
    neighbours_ptr->p00 = p00;
    neighbours_ptr->p01 = p01;
    neighbours_ptr->p10 = p10;
    neighbours_ptr->p11 = p11;
    neighbours_ptr->u_low_side_px = u_low_side_px;
  }
  return true;
}

__device__ bool interpolateLidarImage(
    const Lidar& lidar, const Vector3f& p_voxel_center_C, const float* image,
    const Vector2f& u_px, const int rows, const int cols,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    float* image_value) {
  // Try linear interpolation first
  interpolation::Interpolation2DNeighbours<float> neighbours;
  bool linear_interpolation_success = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, rows, cols, image_value, &neighbours);

  // Additional check
  // Check that we're not interpolating over a discontinuity
  // NOTE(alexmillane): This prevents smearing are object edges.
  if (linear_interpolation_success) {
    const float d00 = fabsf(neighbours.p00 - *image_value);
    const float d01 = fabsf(neighbours.p01 - *image_value);
    const float d10 = fabsf(neighbours.p10 - *image_value);
    const float d11 = fabsf(neighbours.p11 - *image_value);
    float maximum_depth_difference_to_neighbours =
        fmax(fmax(d00, d01), fmax(d10, d11));
    if (maximum_depth_difference_to_neighbours >
        linear_interpolation_max_allowable_difference_m) {
      linear_interpolation_success = false;
    }
  }

  // If linear didn't work - try nearest neighbour interpolation
  if (!linear_interpolation_success) {
    Index2D u_neighbour_px;
    if (!interpolation::interpolate2DClosest<
            float, interpolation::checkers::FloatPixelGreaterThanZero>(
            image, u_px, rows, cols, image_value, &u_neighbour_px)) {
      // If we can't successfully do closest, fail to intgrate this voxel.
      return false;
    }
    // Additional check
    // Check that this voxel is close to the ray passing through the pixel.
    // Note(alexmillane): This is to prevent large numbers of voxels
    // being integrated by a single pixel at long ranges.
    const Vector3f closest_ray = lidar.vectorFromPixelIndices(u_neighbour_px);
    const float off_ray_squared_distance =
        (p_voxel_center_C - p_voxel_center_C.dot(closest_ray) * closest_ray)
            .squaredNorm();
    if (off_ray_squared_distance >
        nearest_interpolation_max_allowable_squared_dist_to_ray_m) {
      return false;
    }
  }

  // TODO(alexmillane): We should add clearing rays, even in the case both
  // interpolations fail.

  return true;
}

}  // namespace interpolation
}  // namespace nvblox
