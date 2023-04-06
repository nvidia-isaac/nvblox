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

#include <fstream>
#include <memory>
#include <string>

#include "nvblox/core/types.h"
#include "nvblox/datasets/data_loader.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/executables/fuser.h"

namespace nvblox {
namespace datasets {
namespace redwood {

// Build a Fuser for the Redwood dataset
std::unique_ptr<Fuser> createFuser(const std::string base_path);

///@brief A class for loading Redwood data
class DataLoader : public RgbdDataLoaderInterface {
 public:
  /// Constructors not intended to be called directly, use factor
  /// DataLoader::create();
  DataLoader(const std::string& base_path, bool multithreaded = true);
  virtual ~DataLoader() = default;

  /// Builds a DatasetLoader
  ///@param base_path Path to the replica dataset sequence base folder.
  ///@param multithreaded Whether or not to multi-thread image loading
  ///@return std::unique_ptr<DataLoader> The dataset loader. May be nullptr if
  /// construction fails.
  static std::unique_ptr<DataLoader> create(const std::string& base_path,
                                            bool multithreaded = true);

  /// Interface for a function that loads the next frames in a dataset
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  ///@param[out] camera_ptr The intrinsic camera model.
  ///@param[out] color_frame_ptr Optional, load color frame.
  ///@return Whether loading succeeded.
  DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                          Transform* T_L_C_ptr,         // NOLINT
                          Camera* camera_ptr,           // NOLINT
                          ColorImage* color_frame_ptr = nullptr) override;

 protected:
  const std::string base_path_;
  const std::string dataset_name_;

  // The pose file.
  // Note(alexmillane): Note that all the poses are in a single file so we keep
  // the file open here.
  std::ifstream trajectory_file_;

  // The camera intrinsics for this dataset are just described on the  website,
  // not by a file, so we hardcode them into this class.
  Camera camera_;

  // The next frame to be loaded
  int frame_number_ = 0;
};

}  // namespace redwood
}  // namespace datasets
}  // namespace nvblox
