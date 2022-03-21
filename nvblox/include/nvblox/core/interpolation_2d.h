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

#include "nvblox/core/image.h"
#include "nvblox/core/types.h"

namespace nvblox {
namespace interpolation {

// CPU interfaces
template <typename ElementType>
__host__ inline bool interpolate2DClosest(const Image<ElementType>& frame,
                                          const Vector2f& u_px,
                                          ElementType* value_interpolated_ptr);
template <typename ElementType>
__host__ inline bool interpolate2DLinear(const Image<ElementType>& frame,
                                         const Vector2f& u_px,
                                         ElementType* value_interpolated_ptr);
template <typename ElementType>
__host__ bool interpolate2D(
    const Image<ElementType>& frame, const Vector2f& u_px,
    ElementType* value_interpolated_ptr,
    const InterpolationType type = InterpolationType::kLinear);

// GPU interfaces
template <typename ElementType>
__host__ __device__ inline bool interpolate2DClosest(
    const ElementType* frame, const Vector2f& u_px, const int rows,
    const int cols, ElementType* value_interpolated_ptr);
template <typename ElementType>
__host__ __device__ inline bool interpolate2DLinear(
    const ElementType* frame, const Vector2f& u_px, const int rows,
    const int cols, ElementType* value_interpolated_ptr);

}  // namespace interpolation
}  // namespace nvblox

#include "nvblox/core/impl/interpolation_2d_impl.h"
