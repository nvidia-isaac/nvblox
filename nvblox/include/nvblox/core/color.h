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

#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

namespace nvblox {

/// Color, stored as 8-bit RGBA, with helper functions for commonly-used colors.
struct Color {
  __host__ __device__ Color() : r(0), g(0), b(0), a(0) {}
  __host__ __device__ Color(uint8_t _r, uint8_t _g, uint8_t _b)
      : r(_r), g(_g), b(_b), a(255) {}
  __host__ __device__ Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a)
      : r(_r), g(_g), b(_b), a(_a) {}

  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;

  /// Check if colors are exactly identical. Also checks alpha.
  __host__ __device__ bool operator==(const Color& other) const {
    return (r == other.r) && (g == other.g) && (b == other.b) && (a == other.a);
  }

  /// Static functions for working with colors
  __host__ __device__ static Color blendTwoColors(const Color& first_color,
                                                  float first_weight,
                                                  const Color& second_color,
                                                  float second_weight);

  // Now a bunch of static colors to use! :)
  __host__ __device__ static const Color White() {
    return Color(255, 255, 255);
  }
  __host__ __device__ static const Color Black() { return Color(0, 0, 0); }
  __host__ __device__ static const Color Gray() { return Color(127, 127, 127); }
  __host__ __device__ static const Color Red() { return Color(255, 0, 0); }
  __host__ __device__ static const Color Green() { return Color(0, 255, 0); }
  __host__ __device__ static const Color Blue() { return Color(0, 0, 255); }
  __host__ __device__ static const Color Yellow() { return Color(255, 255, 0); }
  __host__ __device__ static const Color Orange() { return Color(255, 127, 0); }
  __host__ __device__ static const Color Purple() { return Color(127, 0, 255); }
  __host__ __device__ static const Color Teal() { return Color(0, 255, 255); }
  __host__ __device__ static const Color Pink() { return Color(255, 0, 127); }
};

}  // namespace nvblox
