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

#include <algorithm>
#include <random>

#include "nvblox/core/types.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace test_utils {

constexpr float kInvalidDepth = -1.F;
DepthImage matrixToDepthImage(const Eigen::MatrixXf& mat);

primitives::Scene getSphereInBox();

DepthImage getPlaneDepthImage(const primitives::Plane& plane,
                              const float invalid_depth = -1.F);

template <typename SensorType>
Eigen::MatrixX3f backProjectVectorized(const Eigen::MatrixX2f& uv_coordinates,
                                       const DepthImage& depth_frame,
                                       const SensorType& sensor) {
  CHECK((uv_coordinates.col(0).array() >= 0.0f).all() &&
        (uv_coordinates.col(0).array() < sensor.width()).all());
  CHECK((uv_coordinates.col(1).array() >= 0.0f).all() &&
        (uv_coordinates.col(1).array() < sensor.height()).all());
  // Plane-ray intersection
  Eigen::ArrayX3f p3ds_matrix(uv_coordinates.rows(), 3);

  for (int row = 0; row < uv_coordinates.rows(); ++row) {
    Vector2f uv = uv_coordinates.row(row);
    p3ds_matrix.row(row) = sensor.unprojectFromImagePlaneCoordinates(
        uv, depth_frame(uv.y(), uv.x()));
  }

  return p3ds_matrix;
}

template <typename SensorType>
Eigen::MatrixX2f getRandomPixelLocationsWhereDepthIsValid(
    const int num_samples, const SensorType& sensor,
    const DepthImage& depth_frame) {
  const int border_px = sensor.width() / 10;

  // Collect all x-y coordinates where depth is valid
  std::vector<Vector2f> valid_pixels;
  for (int y = border_px; y < depth_frame.rows() - border_px; ++y) {
    for (int x = border_px; x < depth_frame.cols() - border_px; ++x) {
      Vector2f uv(x, y);

      if (depth_frame(uv.y(), uv.x()) != kInvalidDepth) {
        valid_pixels.push_back(uv);
      }
    }
  }

  // Shuffle them and collect
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(valid_pixels.begin(), valid_pixels.end(),
               std::default_random_engine(seed));
  Eigen::MatrixX2f uv_coordinates = Eigen::MatrixX2f(num_samples, 2).array();
  for (int row = 0; row < std::min<int>(valid_pixels.size(), num_samples);
       ++row) {
    uv_coordinates.row(row) = valid_pixels[row];
  }

  return uv_coordinates;
}

template <typename SensorType>
DepthImage getPlaneDepthImage(const primitives::Plane& plane,
                              const SensorType& sensor,
                              const Transform T_L_C = Transform::Identity()) {
  // Get a depth map of our view of the plane.
  primitives::Scene scene;
  scene.addPrimitive(std::make_unique<primitives::Plane>(plane));
  DepthImage depth_frame(sensor.height(), sensor.width(), MemoryType::kHost);

  constexpr float kMaxDist = 20.F;

  scene.generateDepthImageFromScene(sensor, T_L_C, kMaxDist, &depth_frame,
                                    kInvalidDepth);

  return depth_frame;
}

}  // namespace test_utils
}  // namespace nvblox
