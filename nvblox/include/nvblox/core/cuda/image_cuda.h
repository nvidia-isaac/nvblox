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

#include <utility>

#include "nvblox/core/color.h"

namespace nvblox {
namespace cuda {

template <typename T>
void copy(const int rows, const int cols, const T* from, T* to);

template <typename T>
void toGPU(const T* data, const int rows, const int cols);

float max(const int rows, const int cols, const float* image);
float min(const int rows, const int cols, const float* image);
std::pair<float, float> minmax(const int rows, const int cols,
                               const float* image);

void elementWiseMinInPlace(const int rows, const int cols, const float constant,
                           float* image);
void elementWiseMaxInPlace(const int rows, const int cols, const float constant,
                           float* image);

void diff(const int rows, const int cols, const Color* image_1, const Color* image_2,
          Color* diff_image_ptr);
void diff(const int rows, const int cols, const float* image_1, const float* image_2,
          float* diff_image_ptr);

}  // namespace cuda
}  // namespace nvblox

#include "nvblox/core/cuda/impl/image_cuda_impl.cuh"
