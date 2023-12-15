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
#include "nvblox/datasets/replica.h"

#include "nvblox/utils/logging.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include "nvblox/utils/timing.h"

namespace nvblox {
namespace datasets {
namespace replica {
namespace internal {

template <typename Derived>
void readMatrixFromLine(const std::string& line,
                        Eigen::DenseBase<Derived>* mat_ptr) {
  std::istringstream iss(line);
  for (int row = 0; row < mat_ptr->rows(); row++) {
    for (int col = 0; col < mat_ptr->cols(); col++) {
      float item = 0.0;
      iss >> item;
      (*mat_ptr)(row, col) = item;
    }
  }
}

bool parseCameraFromFile(const std::string& filename, Camera* camera_ptr,
                         float* scale_ptr) {
  CHECK_NOTNULL(camera_ptr);
  CHECK_NOTNULL(scale_ptr);

  // Parameter strings to search for, and how they map into the parameter file
  static const std::unordered_map<std::string, int> parameter_string_to_index =
      {{"w", 0},  {"h", 1},  {"fx", 2},   {"fy", 3},
       {"cx", 4}, {"cy", 5}, {"scale", 6}};

  // Camera parameters stored as a vector of floats.
  std::array<float, 7> camera_params_vec = {-1.0, -1.0, -1.0, -1.0,
                                            -1.0, -1.0, -1.0};

  // For each line
  std::ifstream fin(filename);
  if (!fin.good()) {
    LOG(WARNING) << "No camera file at: " << filename;
    return false;
  }
  std::string line;
  while (getline(fin, line)) {
    // Search line for each parameter string
    for (auto& string_index_pair : parameter_string_to_index) {
      const size_t pos = line.find(string_index_pair.first);
      if (pos != std::string::npos) {
        // Parameter string found
        std::istringstream iss(line);
        std::string item;
        // Find float in the line.
        while (iss >> item) {
          float param_float;
          if (std::istringstream(item) >> param_float) {
            camera_params_vec[string_index_pair.second] = param_float;
          }
        }
        break;
      }
    }
  }
  // Check that we found all parameters
  for (const float param : camera_params_vec) {
    if (param <= 0.0f) {
      LOG(WARNING) << "Failed to find all camera parameters.";
      return false;
    }
  }
  // Output
  *camera_ptr = Camera(camera_params_vec[parameter_string_to_index.at("fx")],
                       camera_params_vec[parameter_string_to_index.at("fy")],
                       camera_params_vec[parameter_string_to_index.at("cx")],
                       camera_params_vec[parameter_string_to_index.at("cy")],
                       camera_params_vec[parameter_string_to_index.at("w")],
                       camera_params_vec[parameter_string_to_index.at("h")]);
  *scale_ptr = camera_params_vec[parameter_string_to_index.at("scale")];
  LOG(INFO) << "Loaded camera\n" << *camera_ptr;
  return true;
}

std::string getPathForCameraIntrinsics(const std::string& base_path) {
  return base_path + "/../cam_params.json";
}

std::string getPathForTrajectory(const std::string& base_path) {
  return base_path + "/traj.txt";
}

std::string getPathForDepthImage(const std::string& base_path,
                                 const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/results/depth" << std::setfill('0') << std::setw(6)
     << frame_id << ".png";
  return ss.str();
}

std::string getPathForColorImage(const std::string& base_path,
                                 const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/results/frame" << std::setfill('0') << std::setw(6)
     << frame_id << ".jpg";
  return ss.str();
}

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
    const std::string& base_path, const float depth_scaling_factor,
    const bool multithreaded) {
  return createImageLoader<DepthImage>(
      std::bind(getPathForDepthImage, base_path, std::placeholders::_1),
      multithreaded, depth_scaling_factor);
}

std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
    const std::string& base_path, const bool multithreaded) {
  return createImageLoader<ColorImage>(
      std::bind(getPathForColorImage, base_path, std::placeholders::_1),
      multithreaded);
}

}  // namespace internal

std::unique_ptr<Fuser> createFuser(const std::string base_path) {
  auto data_loader = DataLoader::create(base_path);
  if (!data_loader) {
    return std::unique_ptr<Fuser>();
  }
  return std::make_unique<Fuser>(std::move(data_loader));
}

DataLoader::DataLoader(const std::string& base_path, bool multithreaded)
    : RgbdDataLoaderInterface(
          replica::internal::createDepthImageLoader(
              base_path, datasets::kDefaultUintDepthScaleFactor, multithreaded),
          replica::internal::createColorImageLoader(base_path, multithreaded)),
      base_path_(base_path),
      trajectory_file_(
          std::ifstream(replica::internal::getPathForTrajectory(base_path))) {
  // We load the scale from camera file and reset the depth image loader to
  // include it.
  if (!std::filesystem::exists(base_path)) {
    LOG(WARNING) << "Tried to create a dataloader with a non-existant path.";
    setup_success_ = false;
    return;
  }
  float inv_depth_image_scaling_factor;
  if (replica::internal::parseCameraFromFile(
          replica::internal::getPathForCameraIntrinsics(base_path_), &camera_,
          &inv_depth_image_scaling_factor)) {
    depth_image_loader_ = replica::internal::createDepthImageLoader(
        base_path, 1.0f / inv_depth_image_scaling_factor, multithreaded);
    setup_success_ = true;
  } else {
    LOG(ERROR)
        << "Dataloader failed to initialize. Couldn't load camera parameters.";
    setup_success_ = false;
  }
}

std::unique_ptr<DataLoader> DataLoader::create(const std::string& base_path,
                                               bool multithreaded) {
  // Construct a dataset loader but only return it if everything worked.
  auto dataset_loader = std::make_unique<DataLoader>(base_path, multithreaded);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

/// Interface for a function that loads the next frames in a dataset
///@param[out] depth_frame_ptr The loaded depth frame.
///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
///@param[out] camera_ptr The intrinsic camera model.
///@param[out] color_frame_ptr Optional, load color frame.
///@return Whether loading succeeded.
DataLoadResult DataLoader::loadNext(DepthImage* depth_frame_ptr,
                                    Transform* T_L_C_ptr, Camera* camera_ptr,
                                    ColorImage* color_frame_ptr) {
  CHECK(setup_success_);
  CHECK_NOTNULL(depth_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(camera_ptr);
  // CHECK_NOTNULL(color_frame_ptr); // Can be null

  // Because we might fail along the way, increment the frame number before we
  // start.
  ++frame_number_;

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    LOG(INFO) << "Couldn't find depth image";
    return DataLoadResult::kNoMoreData;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      LOG(INFO) << "Couldn't find color image";
      return DataLoadResult::kNoMoreData;
    }
    timer_file_color.Stop();
  }

  float scale;

  // Get the camera for this frame.
  timing::Timer timer_file_intrinsics("file_loading/camera");
  if (camera_cached_) {
    *camera_ptr = camera_;
  } else {
    if (!replica::internal::parseCameraFromFile(
            replica::internal::getPathForCameraIntrinsics(base_path_), &camera_,
            &scale)) {
      LOG(INFO) << "Couldn't find camera params file";
      return DataLoadResult::kNoMoreData;
    }
    camera_cached_ = true;
    *camera_ptr = camera_;
  }
  timer_file_intrinsics.Stop();

  // Get the next pose
  timing::Timer timer_file_pose("file_loading/pose");
  CHECK(trajectory_file_.is_open());
  std::string line;
  if (std::getline(trajectory_file_, line)) {
    Eigen::Matrix4f T_odom_cam;
    replica::internal::readMatrixFromLine(line, &T_odom_cam);
    *T_L_C_ptr = Eigen::Isometry3f(T_odom_cam);
  } else {
    LOG(INFO) << "Couldn't find pose";
    return DataLoadResult::kNoMoreData;
  }

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C_ptr->matrix().allFinite() ||
      std::abs(T_L_C_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad CSV data.";
    return DataLoadResult::kBadFrame;  // Bad data, but keep going.
  }

  timer_file_pose.Stop();
  return DataLoadResult::kSuccess;
}

}  // namespace replica
}  // namespace datasets
}  // namespace nvblox
