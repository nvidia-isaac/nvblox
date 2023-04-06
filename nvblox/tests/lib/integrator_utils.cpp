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

Eigen::MatrixX3f backProjectToPlaneVectorized(
    const Eigen::MatrixX2f& uv_coordinates, const Plane& plane,
    const Camera& camera) {
  CHECK((uv_coordinates.col(0).array() >= 0.0f).all() &&
        (uv_coordinates.col(0).array() < camera.width()).all());
  CHECK((uv_coordinates.col(1).array() >= 0.0f).all() &&
        (uv_coordinates.col(1).array() < camera.height()).all());
  // Plane-ray intersection
  Eigen::ArrayX3f rays_matrix(uv_coordinates.rows(), 3);
  rays_matrix.col(0) =
      (uv_coordinates.col(0).array() - camera.cu()) / camera.fu();
  rays_matrix.col(1) =
      (uv_coordinates.col(1).array() - camera.cv()) / camera.fv();
  rays_matrix.col(2) = 1.0f;
  const Eigen::ArrayXf t_matrix =
      plane.p.dot(plane.n) *
      (rays_matrix.col(0) * plane.n.x() + rays_matrix.col(1) * plane.n.y() +
       rays_matrix.col(2) * plane.n.z())
          .inverse();
  // Each pixel's 3D point
  return rays_matrix.colwise() * t_matrix;
}

DepthImage getDepthImage(const Plane& plane, const Camera& camera) {
  CHECK(plane.p.z() > 0.0f);
  // Enumerate all pixel locations.
  Eigen::MatrixX2f uv_coordinates(camera.height() * camera.width(), 2);
  int linear_idx = 0;
  for (int u = 0; u < camera.width(); u++) {
    for (int v = 0; v < camera.height(); v++) {
      uv_coordinates(linear_idx, 0) = u;
      uv_coordinates(linear_idx, 1) = v;
      ++linear_idx;
    }
  }
  // Back project and get depth frame
  const Eigen::MatrixX3f points_C =
      backProjectToPlaneVectorized(uv_coordinates, plane, camera);
  Eigen::MatrixXf depths = (points_C.col(2).array());
  depths.resize(camera.height(), camera.width());
  return matrixToDepthImage(depths);
}

Eigen::MatrixX2f getRandomPixelLocations(const int num_samples,
                                         const Camera& camera) {
  // Note: Eigen's Random() generates numbers between -1.0 and 1.0 -> hence the
  // abs().
  Eigen::MatrixX2f uv_coordinates =
      Eigen::MatrixX2f::Random(num_samples, 2).array().abs();
  constexpr int border_px = 20;
  uv_coordinates.col(0) =
      (uv_coordinates.col(0) *
       static_cast<float>(camera.width() - 1 - 2 * border_px))
          .array() +
      border_px;
  uv_coordinates.col(1) =
      (uv_coordinates.col(1) *
       static_cast<float>(camera.height() - 1 - 2 * border_px))
          .array() +
      border_px;
  return uv_coordinates;
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
