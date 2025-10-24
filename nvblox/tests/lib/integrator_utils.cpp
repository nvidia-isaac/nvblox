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
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {
namespace test_utils {

DepthImage matrixToDepthImage(const Eigen::MatrixXf& mat) {
  DepthImage depth_frame(mat.rows(), mat.cols(), MemoryType::kUnified);
  for (int col_idx = 0; col_idx < mat.cols(); col_idx++) {
    for (int row_idx = 0; row_idx < mat.rows(); row_idx++) {
      depth_frame(row_idx, col_idx) = mat(row_idx, col_idx);
    }
  }
  return depth_frame;
}

primitives::Scene getSphereInBox() {
  // Scene is bounded to -5, -5, 0 to 5, 5, 5.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  // Create a scene with a ground plane and a sphere.
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
  return scene;
}

}  // namespace test_utils
}  // namespace nvblox
