/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include "nvblox_torch/py_sensor.h"

namespace pynvblox {

c10::intrusive_ptr<PySensor> PySensor::fromCamera(double fu, double fv,
                                                  double cu, double cv,
                                                  int64_t width,
                                                  int64_t height) {
  auto sensor = c10::make_intrusive<PySensor>();

  nvblox::Camera camera(fu, fv, cu, cv, width, height);
  sensor->sensor_store_.set(std::move(camera));
  sensor->modality_ = nvblox::SensorModality::kCamera;
  sensor->width_ = width;
  sensor->height_ = height;

  return sensor;
}

c10::intrusive_ptr<PySensor> PySensor::fromCameraDistorted(
    double fu, double fv, double cu, double cv, int64_t width, int64_t height,
    double k1, double k2, double k3, double k4, double k5, double k6, double p1,
    double p2) {
  auto sensor = c10::make_intrusive<PySensor>();

  nvblox::RadialTangentialDistortionParams distortion_params;
  distortion_params.radial.k1 = k1;
  distortion_params.radial.k2 = k2;
  distortion_params.radial.k3 = k3;
  distortion_params.radial.k4 = k4;
  distortion_params.radial.k5 = k5;
  distortion_params.radial.k6 = k6;
  distortion_params.tangential.p1 = p1;
  distortion_params.tangential.p2 = p2;

  nvblox::Camera camera(fu, fv, cu, cv, width, height, distortion_params);
  sensor->sensor_store_.set(std::move(camera));
  sensor->modality_ = nvblox::SensorModality::kCamera;
  sensor->width_ = width;
  sensor->height_ = height;

  return sensor;
}

c10::intrusive_ptr<PySensor> PySensor::fromLidar(
    int64_t num_azimuth_divisions, int64_t num_elevation_divisions,
    double vertical_fov_rad, double min_valid_range_m) {
  auto sensor = c10::make_intrusive<PySensor>();

  nvblox::Lidar lidar(num_azimuth_divisions, num_elevation_divisions,
                      vertical_fov_rad, min_valid_range_m);
  sensor->sensor_store_.set(std::move(lidar));
  sensor->modality_ = nvblox::SensorModality::kLidar;
  sensor->width_ = num_azimuth_divisions;
  sensor->height_ = num_elevation_divisions;

  return sensor;
}

c10::intrusive_ptr<PySensor> PySensor::fromLidarAsymmetric(
    int64_t num_azimuth_divisions, int64_t num_elevation_divisions,
    double min_angle_below_zero_elevation_rad,
    double max_angle_above_zero_elevation_rad, double min_valid_range_m) {
  auto sensor = c10::make_intrusive<PySensor>();

  nvblox::Lidar lidar(num_azimuth_divisions, num_elevation_divisions,
                      min_valid_range_m, min_angle_below_zero_elevation_rad,
                      max_angle_above_zero_elevation_rad);
  sensor->sensor_store_.set(std::move(lidar));
  sensor->modality_ = nvblox::SensorModality::kLidar;
  sensor->width_ = num_azimuth_divisions;
  sensor->height_ = num_elevation_divisions;

  return sensor;
}

std::string PySensor::getSensorModality() const {
  switch (modality_) {
    case nvblox::SensorModality::kCamera:
      return "camera";
    case nvblox::SensorModality::kLidar:
      return "lidar";
    default:
      return "unknown";
  }
}

int64_t PySensor::width() const { return width_; }

int64_t PySensor::height() const { return height_; }

}  // namespace pynvblox
