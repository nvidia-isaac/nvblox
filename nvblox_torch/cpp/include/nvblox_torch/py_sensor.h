/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/lidar.h>
#include <nvblox/sensors/type_indexed_store.h>

namespace pynvblox {

/// Python wrapper for nvblox sensors using TypeIndexedStore
/// This provides a single unified interface for all sensor types
struct PySensor : torch::CustomClassHolder {
  PySensor() = default;

  /// Create sensor from Camera (pinhole, no distortion)
  static c10::intrusive_ptr<PySensor> fromCamera(double fu, double fv,
                                                 double cu, double cv,
                                                 int64_t width, int64_t height);

  /// Create sensor from Camera with distortion
  static c10::intrusive_ptr<PySensor> fromCameraDistorted(
      double fu, double fv, double cu, double cv, int64_t width, int64_t height,
      double k1, double k2, double k3, double k4, double k5, double k6,
      double p1, double p2);

  /// Create sensor from Lidar
  static c10::intrusive_ptr<PySensor> fromLidar(int64_t num_azimuth_divisions,
                                                int64_t num_elevation_divisions,
                                                double vertical_fov_rad,
                                                double min_valid_range_m);

  /// Create sensor from Lidar with an asymmetric elevation envelope, for beams
  /// not distributed evenly around zero (see the second nvblox::Lidar ctor).
  static c10::intrusive_ptr<PySensor> fromLidarAsymmetric(
      int64_t num_azimuth_divisions, int64_t num_elevation_divisions,
      double min_angle_below_zero_elevation_rad,
      double max_angle_above_zero_elevation_rad, double min_valid_range_m);

  /// Get the sensor modality
  std::string getSensorModality() const;

  /// Get width/height (works for both Camera and Lidar)
  int64_t width() const;
  int64_t height() const;

  /// Get the type-indexed store
  const nvblox::TypeIndexedStore& getSensorStore() const {
    return sensor_store_;
  }

  /// Check if the sensor is of a specific type
  template <typename SensorType>
  bool isSensorType() const {
    // Note that the sensor_store can technically hold multiple sensors of
    // different types, but in this class we only store a single one.
    return sensor_store_.hasType<SensorType>();
  }

  /// Get the stored nvblox sensor. This function is templated to comply with
  /// the mapper interface that is generic for all sensor types. The template
  /// param should be the same as the constructed sensor type.
  template <typename SensorType>
  const SensorType& getNvbloxSensor() const {
    return sensor_store_.get<SensorType>();
  }

 private:
  nvblox::TypeIndexedStore sensor_store_;
  nvblox::SensorModality modality_;
  int64_t width_ = 0;
  int64_t height_ = 0;
};

}  // namespace pynvblox
