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

#include <memory>
#include <string>

#include "nvblox/core/types.h"
#include "nvblox/datasets/data_loader.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/executables/fuser_lidar.h"

namespace nvblox {
namespace datasets {
namespace fusionportable {

// TODO(jjiao): the default settings of preprocess height_image
constexpr float kDefaultUintDepthScaleFactor = 1.0f / 1000.0f;
constexpr float kDefaultUintDepthScaleOffset = 10.0f;

// Build a FuserLidar for the FusionPortable dataset
std::unique_ptr<FuserLidar> createFuser(const std::string base_path,
                                        const int seq_id);

///@brief A class for loading FusionPortable data
class DataLoader : public RgbdDataLoaderInterface {
 public:
  DataLoader(const std::string& base_path, const int seq_id,
             bool multithreaded = true);

  // NOTE(jjiao): need to define the virutal function (not used) here
  DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                          Transform* T_L_C_ptr,         // NOLINT
                          Camera* camera_ptr,           // NOLINT
                          ColorImage* color_frame_ptr = nullptr) override;

  /// Interface for a function that loads the next frames in a dataset
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  ///@param[out] camera_ptr The intrinsic camera model.
  ///@param[out] lidar_ptr The intrinsic oslidar model.
  ///@param[out] height_frame_ptr The loaded z frame.
  ///@param[out] color_frame_ptr Optional, load color frame.
  ///@return Whether loading succeeded.
  DataLoadResult loadNext(DepthImage* depth_frame_ptr,             // NOLINT
                          Transform* T_L_C_ptr,                    // NOLINT
                          Camera* camera_ptr,                      // NOLINT
                          OSLidar* lidar_ptr,                      // NOLINT
                          DepthImage* height_frame_ptr = nullptr,  // NOLINT
                          ColorImage* color_frame_ptr = nullptr) override;

 protected:
  std::unique_ptr<ImageLoader<DepthImage>> height_image_loader_;

  const std::string base_path_;
  const int seq_id_;

  // The next frame to be loaded
  int frame_number_ = 0;
};

namespace internal {

bool parsePoseFromFile(const std::string& filename, Transform* transform);
bool parseCameraFromFile(const std::string& filename,
                         Eigen::Matrix3f* intrinsics);
bool parseLidarFromFile(const std::string& filename,
                        Eigen::Matrix<double, 4, 1>* intrinsics);
std::string getPathForCameraIntrinsics(const std::string& base_path);
std::string getPathForLidarIntrinsics(const std::string& base_path);
std::string getPathForFramePose(const std::string& base_path, const int seq_id,
                                const int frame_id);

std::string getPathForDepthImage(const std::string& base_path, const int seq_id,
                                 const int frame_id);
std::string getPathForHeightImage(const std::string& base_path,
                                  const int seq_id, const int frame_id);
std::string getPathForColorImage(const std::string& base_path, const int seq_id,
                                 const int frame_id);

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
    const std::string& base_path, const int seq_id,
    const bool multithreaded = true);
std::unique_ptr<ImageLoader<DepthImage>> createHeightImageLoader(
    const std::string& base_path, const int seq_id,
    const bool multithreaded = true);
std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
    const std::string& base_path, const int seq_id,
    const bool multithreaded = true);

}  // namespace internal
}  // namespace fusionportable
}  // namespace datasets
}  // namespace nvblox
