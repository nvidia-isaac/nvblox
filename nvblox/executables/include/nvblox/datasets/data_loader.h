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

#include "nvblox/core/camera.h"
#include "nvblox/core/image.h"
#include "nvblox/core/lidar.h"
#include "nvblox/core/oslidar.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/image_loader.h"

namespace nvblox {
namespace datasets {

enum class DataLoadResult { kSuccess, kBadFrame, kNoMoreData };
enum class SensorType { RGBD, LIDAR, OSLIDAR };

class RgbdDataLoaderInterface {
 public:
  RgbdDataLoaderInterface(
      std::unique_ptr<ImageLoader<DepthImage>>&& depth_image_loader,
      std::unique_ptr<ImageLoader<ColorImage>>&& color_image_loader,
      SensorType sensor_type = SensorType::RGBD)
      : depth_image_loader_(std::move(depth_image_loader)),
        color_image_loader_(std::move(color_image_loader)),
        sensor_type_(sensor_type) {}

  virtual ~RgbdDataLoaderInterface() = default;

  /// Interface for a function that loads the next frames in a dataset
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  ///@param[out] camera_ptr The intrinsic camera model.
  ///@param[out] color_frame_ptr Optional, load color frame.
  ///@return Whether loading succeeded.
  virtual DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                                  Transform* T_L_C_ptr,         // NOLINT
                                  Camera* camera_ptr,           // NOLINT
                                  ColorImage* color_frame_ptr = nullptr) = 0;

  /// Interface for a function that loads the next frames in a dataset
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  ///@param[out] camera_ptr The intrinsic camera model.
  ///@param[out] lidar_ptr The intrinsic Ouster lidar model.
  ///@param[out] height_frame_ptr The loaded z frame.
  ///@param[out] color_frame_ptr Optional, load color frame.
  ///@return Whether loading succeeded.
  virtual DataLoadResult loadNext(
      DepthImage* depth_frame_ptr,             // NOLINT
      Transform* T_L_C_ptr,                    // NOLINT
      Camera* camera_ptr,                      // NOLINT
      OSLidar* lidar_ptr,                      // NOLINT
      DepthImage* height_frame_ptr = nullptr,  // NOLINT
      ColorImage* color_frame_ptr = nullptr) = 0;

  SensorType getSensorType() const { return sensor_type_; }

 protected:
  // Objects which do (multithreaded) image loading.
  std::unique_ptr<ImageLoader<DepthImage>> depth_image_loader_;
  std::unique_ptr<ImageLoader<ColorImage>> color_image_loader_;
  // sensor type
  SensorType sensor_type_;
};

}  // namespace datasets
}  // namespace nvblox