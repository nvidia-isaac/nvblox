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

#include <gflags/gflags.h>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/sensors/image.h"

DECLARE_bool(nvblox_test_file_output);

namespace nvblox {
namespace test_utils {

float randomFloatInRange(float f_min, float f_max);

float randomIntInRange(int i_min, int i_max);

float randomSign();

Index3D getRandomIndex3dInRange(const int min, const int max);

Vector3f getRandomVector3fInRange(const float min, const float max);

Vector3f getRandomVector3fInRange(const Vector3f& min, const Vector3f& max);

Vector3f getRandomUnitVector3f();

Color randomColor();

enum class MaskImageType : int64_t {
  kFromDisk = 0,
  kEverythingZero = 1,    // all pixels 255
  kEverythingFilled = 2,  // all pixels 255
  kGrid = 3,              // grid pattern
  kTwoSquares = 4         // Two filled squares
};
void createMaskImage(MonoImage* mask, MaskImageType type);
}  // namespace test_utils
}  // namespace nvblox
