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

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/datasets/data_loader_interface.h"
#include "nvblox/datasets/image_loader.h"

// Forward declaration for simple JSON parser
class SimpleJson;

namespace nvblox {
namespace datasets {
namespace cusfm_data {

struct KeyframeMetadata {
  uint64_t timestamp_microseconds = 0;
  std::string color_image_path = "";
  std::string depth_image_path = "";
  uint32_t camera_params_id = 0;
  Transform camera_to_world;

  bool fromJson(const SimpleJson& json);
};

/// @brief A class for loading mapping data comprising of color images,
/// depth images and a metadata json file that specifies the image paths
/// under color_image_dir and depth_image_dir, camera poses for each image,
/// and projection matrix for each camera sensor. The image paths under
/// color_image_dir and depth_image_dir must match.
class DataLoader : public RgbdDataLoaderInterface {
 public:
  /// Constructors not intended to be called directly, use factory
  /// DataLoader::create();
  DataLoader(const std::string& color_image_dir,
             const std::string& depth_image_dir,
             const std::vector<KeyframeMetadata>& keyframe_metadatas,
             const std::unordered_map<uint32_t, Camera>& cameras,
             bool multithreaded = true, bool fit_to_z_plane = false,
             const std::string& output_dir = "");
  virtual ~DataLoader() = default;

  /// Builds a DatasetLoader
  /// @param color_image_dir Path to the color image folder.
  /// @param depth_image_dir Path to the depth image folder.
  /// @param keyframe_metadata_file Path to the frames_meta file.
  /// @param multithreaded Whether or not to multi-thread image loading
  /// @param fit_to_z_plane Whether to fit all poses to z=0 plane
  /// @param output_dir Output directory for saving transform JSON
  /// @return std::unique_ptr<DataLoader> The dataset loader. May be nullptr if
  /// construction fails.
  static std::unique_ptr<DataLoader> create(
      const std::string& color_image_dir, const std::string& depth_image_dir,
      const std::string& keyframe_metadata_file, bool multithreaded = true,
      bool fit_to_z_plane = false, const std::string& output_dir = "");

  /// CUSFM datasets do not provide frame timestamps
  bool provides_frame_timestamps() const override { return false; }

  /// Interface for a function that loads the next frames in a dataset
  /// @param[out] depth_frame_ptr The loaded depth frame.
  /// @param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  /// @param[out] camera_ptr The intrinsic camera model.
  /// @param[out] color_frame_ptr Optional, load color frame.
  /// @return Whether loading succeeded.
  DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                          Transform* T_L_C_ptr,         // NOLINT
                          Camera* camera_ptr,           // NOLINT
                          ColorImage* color_frame_ptr = nullptr);

  /// Interface for a function that loads the next frames in a dataset.
  /// This is the version of the function for different depth and color cameras.
  /// @param[out] depth_frame_ptr The loaded depth frame.
  /// @param[out] T_L_D_ptr Transform from depth camera to the Layer frame.
  /// @param[out] depth_camera_ptr The intrinsic depth camera model.
  /// @param[out] color_frame_ptr The loaded color frame.
  /// @param[out] T_L_C_ptr Transform from color camera to the Layer frame.
  /// @param[out] color_camera_ptr The intrinsic color camera model.
  /// @param[out] unused Needed to match data loader interface (pass nullptr).
  /// @param[out] unused Needed to match data loader interface (pass nullptr).
  /// @param[out] unused Needed to match data loader interface (pass nullptr).
  /// @return Whether loading succeeded.
  DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                          Transform* T_L_D_ptr,         // NOLINT
                          Camera* depth_camera_ptr,     // NOLINT
                          ColorImage* color_frame_ptr,  // NOLINT
                          Transform* T_L_C_ptr,         // NOLINT
                          Camera* color_camera_ptr,     // NOLINT
                          Time*,                        // NOLINT
                          Transform*,                   // NOLINT
                          Time*) override;              // NOLINT

 protected:
  // Mapping data directory
  const std::string base_path_;

  // Keyframes metadatas
  const std::vector<KeyframeMetadata> keyframe_metadatas_;

  // Camera intrinsics
  const std::unordered_map<uint32_t, Camera> cameras_;

  // The next frame to be loaded
  int32_t frame_number_ = 0;

  std::unique_ptr<ImageLoader<DepthImage>> depth_image_loader_;
  std::unique_ptr<ImageLoader<ColorImage>> color_image_loader_;

 private:
  // Z-plane alignment functionality
  bool fit_to_z_plane_;
  std::string output_dir_;
  Transform T_world_to_z0_plane_;
  bool has_z_plane_transform_;

  /// Compute transform to align all poses to z=0 plane
  void computeZPlaneTransform();

  /// Save the z-plane transform to JSON file
  bool saveTransformToJson() const;
};

}  // namespace cusfm_data
}  // namespace datasets
}  // namespace nvblox
