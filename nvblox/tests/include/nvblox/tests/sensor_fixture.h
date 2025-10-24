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

#include <gtest/gtest.h>
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/type_indexed_store.h"
#include "nvblox/tests/custom_camera_sensor.h"

namespace nvblox {
namespace test_utils {

// Fixture for storing sensors used in unit tests. Apart from the built-in
// Camera and Lidar, we also store a CustomSensorType to also test an externally
// declared sensor type. This type is identical to camera.
template <typename SensorType>
class SensorFixture : public ::testing::Test {
 protected:
  SensorFixture() {
    sensor_store_.set<Camera>(getDefaultCamera());
    sensor_store_.set(getDefaultLidar());
    sensor_store_.set<CustomCameraSensor>(
        test_utils::CustomCameraSensor(getDefaultCamera()));
  }
  SensorFixture(const Camera& camera) {
    sensor_store_.set<Camera>(camera);
    sensor_store_.set(getDefaultLidar());
    // CustomCameraSensor is constructible from Camera, so construct one from
    // 'camera' and store it
    sensor_store_.set<CustomCameraSensor>(CustomCameraSensor(camera));
  }

  SensorFixture(const Lidar& lidar) {
    sensor_store_.set(lidar);
    sensor_store_.set<Camera>(getDefaultCamera());
    sensor_store_.set<CustomCameraSensor>(getDefaultCamera());
  }
  SensorFixture(const Lidar& lidar, const Camera& camera) {
    sensor_store_.set(lidar);
    sensor_store_.set(camera);
  }

  Camera getDefaultCamera() {
    return Camera(fu_, fv_, cu_, cv_, width_, height_, k1_, k2_, k3_, p1_, p2_);
  }

  Lidar getDefaultLidar() {
    return Lidar(kNumAzimuthDivisions, kNumElevationDivisions, kMinValidRangeM,
                 kMaxValidRangeM, kVerticalFovRad);
  }

  // Camera parameters.
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  constexpr static float k1_ = 0.1f;
  constexpr static float k2_ = 0.1f;
  constexpr static float k3_ = 0.01f;
  constexpr static float p1_ = 0.01f;
  constexpr static float p2_ = 0.02f;

  // Lidar parameters
  constexpr static int kNumAzimuthDivisions = 500;
  constexpr static int kNumElevationDivisions = 500;
  constexpr static float kMinValidRangeM = 1E-2;
  constexpr static float kMaxValidRangeM = 50.0F;
  constexpr static float kVerticalFovRad = 90.F * M_PI / 180.F;

  // Get the sensor depending on SensorType
  SensorType sensor() { return sensor_store_.get<SensorType>(); }

 private:
  TypeIndexedStore sensor_store_;
};

// Since the local coordinate frames of the sensors differ, we provide per-type
// functions for transforming from sensor to layer.
template <typename SensorType>
Transform getSensorTLC();

/// @return
template <>
Transform getSensorTLC<Camera>() {
  Eigen::Matrix3f R_L_C;

  // Column vectors of R_L_C are the camera axes seen in the layer coordinate
  // frame.
  R_L_C.col(0) << 0, -1, 0;  // Cam X maps to layer -Y
  R_L_C.col(1) << 0, 0, -1;  // Cam Y maps to layer -Z
  R_L_C.col(2) << 1, 0, 0;   // Cam Z maps to layer X

  Transform T_L_C = Transform::Identity();
  T_L_C.linear() = R_L_C;
  return T_L_C;
}

template <>
Transform getSensorTLC<Lidar>() {
  return Transform::Identity();
}

}  // namespace test_utils
}  // namespace nvblox
