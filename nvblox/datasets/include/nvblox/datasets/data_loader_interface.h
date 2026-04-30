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

#include "nvblox/core/time.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"
#include "nvblox/sensors/sensor.h"

namespace nvblox {
namespace datasets {

enum class DataLoadResult { kSuccess, kBadFrame, kNoMoreData };

/// Templated base interface for all dataset loaders
/// @tparam SensorType The sensor model type (Camera or Lidar)
/// @tparam SensorDataType The data type of the sensor (DepthImage or
/// Pointcloud)
template <typename SensorType, typename SensorDataType>
class DataLoaderInterface {
 public:
  DataLoaderInterface() = default;
  virtual ~DataLoaderInterface() = default;

  /// Indicates if the data loader was successfully set up.
  /// @return True if the DataLoader was successfully set up.
  bool setup_success() const { return setup_success_; }

  /// Indicates whether this data loader provides color data
  /// @return True if color data is available
  virtual bool provides_color() const = 0;

  /// Indicates whether this data loader provides timestamps
  /// @return True if timestamp data is available
  virtual bool provides_frame_timestamps() const = 0;

  /// Indicates whether this data loader provides lidar scan data
  /// @return True if lidar scan data is available
  virtual bool provides_lidar_scan_data() const = 0;

  /// Interface for a function that loads the next frames in a dataset.
  ///@param[out] sensor_data_ptr The loaded sensor data (DepthImage or
  /// Pointcloud).
  ///@param[out] T_L_S_ptr Transform from sensor to the Layer frame.
  ///@param[out] sensor_ptr The intrinsic sensor model.
  ///@param[out] color_frame_ptr Optional color frame (if provides_color() is
  /// true).
  ///@param[out] T_L_C_ptr Optional color sensor pose (if provides_color() is
  /// true).
  ///@param[out] color_sensor_ptr Optional color camera (if provides_color() is
  /// true).
  ///@param[out] frame_timestamp_ms_ptr Optional timestamp of the frame in
  /// milliseconds (if provides_frame_timestamps() is true).
  ///@param[out] T_L_S_scanEnd_ptr Optional lidar pose at scan end (if
  /// provides_lidar_scan_data() is true).
  ///@param[out] scan_duration_ms_ptr Optional lidar scan duration (if
  /// provides_lidar_scan_data() is true).
  ///@return Whether loading succeeded.
  virtual DataLoadResult loadNext(
      SensorDataType* sensor_data_ptr,            // NOLINT
      Transform* T_L_S_ptr,                       // NOLINT
      SensorType* sensor_ptr,                     // NOLINT
      ColorImage* color_frame_ptr = nullptr,      // NOLINT
      Transform* T_L_C_ptr = nullptr,             // NOLINT
      Camera* color_sensor_ptr = nullptr,         // NOLINT
      Time* frame_timestamp_ms_ptr = nullptr,     // NOLINT
      Transform* T_L_S_scanEnd_ptr = nullptr,     // NOLINT
      Time* scan_duration_ms_ptr = nullptr) = 0;  // NOLINT

 protected:
  // Indicates if the dataset loader was constructed in a state that was good to
  // go. Initializes to true, so child class constructors indicate failure by
  // setting it to false;
  bool setup_success_ = true;
};

/// Interface for RGBD (camera-based) dataset loaders
class RgbdDataLoaderInterface : public DataLoaderInterface<Camera, DepthImage> {
 public:
  RgbdDataLoaderInterface() = default;
  virtual ~RgbdDataLoaderInterface() = default;

  /// RGBD loaders provide color data
  bool provides_color() const override { return true; }

  /// RGBD loaders do not provide lidar scan data
  bool provides_lidar_scan_data() const override { return false; }
};

/// Interface for lidar dataset loaders that provide raw pointclouds
class LidarDataLoaderInterface : public DataLoaderInterface<Lidar, Pointcloud> {
 public:
  LidarDataLoaderInterface() = default;
  virtual ~LidarDataLoaderInterface() = default;

  /// Lidar loaders do not provide color data
  bool provides_color() const override { return false; }
};

}  // namespace datasets
}  // namespace nvblox
