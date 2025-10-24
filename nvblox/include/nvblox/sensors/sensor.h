/*
Copyright 2025 NVIDIA CORPORATION

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
#include "nvblox/core/types.h"

namespace nvblox {

/// Enum to identify different sensor modalities
enum class SensorModality { kCamera, kLidar };

// Base class for sensor. Since we generally rely on templated functions instead
// of polymorphism, nothing is defined in the base class. However it is useful
// in cases where different sensor types need to coexist.
class SensorBase {
 public:
  virtual __host__ __device__ ~SensorBase() = default;
  static constexpr float kDefaultMinProjectionDepth = 1E-6;
};

// Trait for checking if a class defines functions required for a sensor
// interface.
// * Note that we can not rely on polymorphism due to restrictions in CUDA.
// * For simplicity, we do not check for return types.
template <typename T>
struct is_sensor_interface {
 private:
  // Overload that checks for required interface.
  // Need int argument to prioritize this overload over the variadic.
  template <typename U>
  static auto test(int)
      -> decltype(U().project(Vector3f(), std::declval<Vector2f*>(), float()),
                  U().getDepth(Vector3f()), U().sensor_modality(),
                  U().unprojectFromImagePlaneCoordinates(Vector2f(), float()),
                  U().unprojectFromPixelIndices(Index2D(), float()),
                  U().vectorFromImagePlaneCoordinates(Vector2f()),
                  U().vectorFromPixelIndices(Index2D()), U().width(),
                  U().height(), U().getViewAABB(Transform(), float(), float()),
                  U().interpolateDepthImage(DepthImageConstView(), Vector2f(),
                                            Vector3f(), float(),
                                            std::declval<float*>(),
                                            std::declval<Index2D*>()),
                  std::true_type{});

  // Variadic overload that returns false_type if interface is not satisfied.
  template <typename>
  static auto test(...) -> std::false_type;

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

}  // namespace nvblox
