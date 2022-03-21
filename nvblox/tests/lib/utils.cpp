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
#include "nvblox/tests/utils.h"

#include <cstdlib>

#include "nvblox/core/accessors.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
namespace test_utils {

float randomFloatInRange(float f_min, float f_max) {
  float f = static_cast<float>(std::rand()) / RAND_MAX;
  return f_min + f * (f_max - f_min);
}

float randomIntInRange(int i_min, int i_max) {
  return static_cast<int>(randomFloatInRange(i_min, i_max + 1));
}

float randomSign() {
  constexpr int kHalfMaxRandMax = RAND_MAX / 2;
  return (std::rand() < kHalfMaxRandMax) ? -1.0f : 1.0f;
}

Index3D getRandomIndex3dInRange(const int min, const int max) {
  return Index3D(test_utils::randomIntInRange(min, max),
                 test_utils::randomIntInRange(min, max),
                 test_utils::randomIntInRange(min, max));
}

Vector3f getRandomVector3fInRange(const float min, const float max) {
  return Vector3f(test_utils::randomFloatInRange(min, max),
                  test_utils::randomFloatInRange(min, max),
                  test_utils::randomFloatInRange(min, max));
}

Vector3f getRandomVector3fInRange(const Vector3f& min, const Vector3f& max) {
  return Vector3f(test_utils::randomFloatInRange(min.x(), max.x()),
                  test_utils::randomFloatInRange(min.y(), max.y()),
                  test_utils::randomFloatInRange(min.z(), max.z()));
}

Vector3f getRandomUnitVector3f() {
  return getRandomVector3fInRange(-1.0, 1.0).normalized();
}

Color randomColor() {
  return Color(static_cast<uint8_t>(randomIntInRange(0, 255)),
               static_cast<uint8_t>(randomIntInRange(0, 255)),
               static_cast<uint8_t>(randomIntInRange(0, 255)));
}

}  // namespace test_utils
}  // namespace nvblox
