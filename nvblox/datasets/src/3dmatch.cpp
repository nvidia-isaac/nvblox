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
#include "nvblox/datasets/3dmatch.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

#include "nvblox/utils/timing.h"

namespace nvblox {
namespace datasets {
namespace threedmatch {
namespace internal {

bool parsePoseFromFile(const std::string& filename, Transform* transform) {
  CHECK_NOTNULL(transform);
  constexpr int kDimension = 4;

  std::ifstream fin(filename);
  if (fin.is_open()) {
    for (int row = 0; row < kDimension; row++)
      for (int col = 0; col < kDimension; col++) {
        float item = 0.0;
        fin >> item;
        (*transform)(row, col) = item;
      }
    fin.close();
    return true;
  }
  return false;
}

bool parseCameraFromFile(const std::string& filename,
                         Eigen::Matrix3f* intrinsics,
                         RadialTangentialDistortionParams* distortion_params) {
  CHECK_NOTNULL(intrinsics);
  constexpr int kDimension = 3;

  std::ifstream fin(filename);
  if (fin.is_open()) {
    for (int row = 0; row < kDimension; row++)
      for (int col = 0; col < kDimension; col++) {
        float item = 0.0;
        fin >> item;
        (*intrinsics)(row, col) = item;
      }
    if (!fin.eof() && distortion_params != nullptr) {
      fin >> distortion_params->radial.k1;
      fin >> distortion_params->radial.k2;
      fin >> distortion_params->radial.k3;
      fin >> distortion_params->radial.k4;
      fin >> distortion_params->radial.k5;
      fin >> distortion_params->radial.k6;
      fin >> distortion_params->tangential.p1;
      fin >> distortion_params->tangential.p2;
    }
    fin.close();

    DLOG(INFO) << "Read camera params:\n" << *intrinsics << std::endl;
    if (distortion_params != nullptr) {
      DLOG(INFO) << "Read distortion params:\n"
                 << distortion_params->radial.k1 << " "
                 << distortion_params->radial.k2 << " "
                 << distortion_params->radial.k3 << " "
                 << distortion_params->radial.k4 << " "
                 << distortion_params->radial.k5 << " "
                 << distortion_params->radial.k6 << " "
                 << distortion_params->tangential.p1 << " "
                 << distortion_params->tangential.p2 << std::endl;
    }
    return true;
  }
  return false;
}

std::string getPathForCameraIntrinsics(const std::string& base_path) {
  return base_path + "/camera-intrinsics.txt";
}

std::string getPathForFramePose(const std::string& base_path, const int seq_id,
                                const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".pose.txt";

  return ss.str();
}

std::string getPathForDepthImage(const std::string& base_path, const int seq_id,
                                 const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".depth.png";

  return ss.str();
}

std::string getPathForColorImage(const std::string& base_path, const int seq_id,
                                 const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".color.png";

  return ss.str();
}

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
    const std::string& base_path, const int seq_id, const bool multithreaded) {
  return createImageLoader<DepthImage>(
      std::bind(getPathForDepthImage, base_path, seq_id, std::placeholders::_1),
      multithreaded);
}

std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
    const std::string& base_path, const int seq_id, const bool multithreaded) {
  return createImageLoader<ColorImage>(
      std::bind(getPathForColorImage, base_path, seq_id, std::placeholders::_1),
      multithreaded);
}

}  // namespace internal

std::unique_ptr<DataLoader> DataLoader::create(const std::string& base_path,
                                               const int seq_id,
                                               bool multithreaded) {
  // Construct a dataset loader but only return it if everything worked.
  auto dataset_loader =
      std::make_unique<DataLoader>(base_path, seq_id, multithreaded);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

DataLoader::DataLoader(const std::string& base_path, const int seq_id,
                       bool multithreaded)
    : RgbdDataLoaderInterface(),
      base_path_(base_path),
      seq_id_(seq_id),
      depth_image_loader_(threedmatch::internal::createDepthImageLoader(
          base_path, seq_id, multithreaded)),
      color_image_loader_(threedmatch::internal::createColorImageLoader(
          base_path, seq_id, multithreaded)) {
  // If the base path doesn't exist return fail
  if (!std::filesystem::exists(base_path)) {
    LOG(WARNING) << "Tried to create a dataloader with a non-existant path.";
    setup_success_ = false;
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
  const int frame_number = frame_number_;
  ++frame_number_;

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  // DepthImage depth_frame;
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    return DataLoadResult::kNoMoreData;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    // ColorImage color_frame;
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      return DataLoadResult::kNoMoreData;
    }
    timer_file_color.Stop();
  }

  // Get the camera for this frame.
  timing::Timer timer_file_camera("file_loading/camera");
  Eigen::Matrix3f camera_intrinsics;
  RadialTangentialDistortionParams distortion_params;
  if (!threedmatch::internal::parseCameraFromFile(
          threedmatch::internal::getPathForCameraIntrinsics(base_path_),
          &camera_intrinsics, &distortion_params)) {
    return DataLoadResult::kNoMoreData;
  }
  // Create a camera object.
  const int image_width = depth_frame_ptr->cols();
  const int image_height = depth_frame_ptr->rows();
  *camera_ptr = Camera::fromIntrinsicsMatrix(camera_intrinsics, image_width,
                                             image_height);
  timer_file_camera.Stop();

  // Get the transform.
  timing::Timer timer_file_pose("file_loading/pose");
  Transform T_O_C;
  if (!threedmatch::internal::parsePoseFromFile(
          threedmatch::internal::getPathForFramePose(base_path_, seq_id_,
                                                     frame_number),
          &T_O_C)) {
    return DataLoadResult::kNoMoreData;
  }

  // Rotate the world frame since Y is up in the normal 3D match datasets.
  Eigen::Quaternionf q_L_O =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 1, 0), Vector3f(0, 0, 1));
  *T_L_C_ptr = q_L_O * T_O_C;

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C_ptr->matrix().allFinite() || !camera_intrinsics.allFinite() ||
      std::abs(T_L_C_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad CSV data.";
    return DataLoadResult::kBadFrame;  // Bad data, but keep going.
  }

  timer_file_pose.Stop();

  return DataLoadResult::kSuccess;
}

DataLoadResult DataLoader::loadNext(
    DepthImage* depth_frame_ptr, Transform* T_L_D_ptr, Camera* depth_camera_ptr,
    ColorImage* color_frame_ptr, Transform* T_L_C_ptr, Camera* color_camera_ptr,
    Time*, Transform*, Time*) {
  // NOTE: The other pointers are checked non-null below
  CHECK_NOTNULL(color_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(color_camera_ptr);

  // For the replica dataset the depth and color cameras are the same, so just
  // copying over.
  auto result =
      loadNext(depth_frame_ptr, T_L_D_ptr, depth_camera_ptr, color_frame_ptr);
  *T_L_C_ptr = *T_L_D_ptr;
  *color_camera_ptr = *depth_camera_ptr;
  return result;
}

}  // namespace threedmatch
}  // namespace datasets
}  // namespace nvblox
