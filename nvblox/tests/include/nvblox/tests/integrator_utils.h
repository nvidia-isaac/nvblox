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
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace test_utils {

struct Plane {
  Plane(const Vector3f& _p, const Vector3f& _n) : p(_p), n(_n){};

  static Plane RandomAtPoint(const Vector3f& p) {
    return Plane(p, Vector3f::Random().normalized());
  }

  const Vector3f p;
  const Vector3f n;
};

DepthImage matrixToDepthImage(const Eigen::MatrixXf& mat);

Eigen::MatrixX3f backProjectToPlaneVectorized(
    const Eigen::MatrixX2f& uv_coordinates, const Plane& plane,
    const Camera& camera);

DepthImage getDepthImage(const Plane& plane, const Camera& camera);

Eigen::MatrixX2f getRandomPixelLocations(const int num_samples,
                                         const Camera& camera);

primitives::Scene getSphereInBox();

}  // namespace test_utils
}  // namespace nvblox
