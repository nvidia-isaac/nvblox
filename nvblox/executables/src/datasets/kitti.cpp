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
#include "nvblox/datasets/kitti.h"

#include <glog/logging.h>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

#include "nvblox/utils/timing.h"

namespace nvblox {
namespace datasets {
namespace kitti {
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

bool parseCameraFromFile(const std::string& filename, Matrix3x4f* P_rect,
                         Matrix3f* R_rect, int* height, int* width) {
  CHECK_NOTNULL(P_rect);
  CHECK_NOTNULL(R_rect);
  CHECK_NOTNULL(height);
  CHECK_NOTNULL(width);

  std::ifstream fin(filename);
  if (fin.is_open()) {
    fin >> *height >> *width;

    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 4; col++) {
        float item = 0.0;
        fin >> item;
        (*P_rect)(row, col) = item;
      }
    }

    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        float item = 0.0;
        fin >> item;
        (*R_rect)(row, col) = item;
      }
    }

    fin.close();
    return true;
  }
  return false;
}  // namespace internal

bool parseLidarFromFile(const std::string& filename,
                        Eigen::Matrix<double, 8, 1>* intrinsics) {
  CHECK_NOTNULL(intrinsics);
  const int kDimension = intrinsics->rows();
  std::ifstream fin(filename);
  if (fin.is_open()) {
    for (int col = 0; col < kDimension; col++) {
      float item = 0.0;
      fin >> item;
      (*intrinsics)(col) = item;
    }
    fin.close();
    return true;
  }
  return false;
}  // namespace internal

// *********************
// *********************
std::string getPathForCameraIntrinsics(const std::string& base_path) {
  return base_path + "/camera-intrinsics.txt";
}

std::string getPathForLidarIntrinsics(const std::string& base_path,
                                      const int seq_id, const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".lidar-intrinsics.txt";
  return ss.str();
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

std::string getPathForHeightImage(const std::string& base_path,
                                  const int seq_id, const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".height.png";
  return ss.str();
}

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
    const std::string& base_path, const int seq_id, const bool multithreaded) {
  return createImageLoader<DepthImage>(
      std::bind(getPathForDepthImage, base_path, seq_id, std::placeholders::_1),
      multithreaded, kDefaultUintDepthScaleFactor, 0.0f);
}

std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
    const std::string& base_path, const int seq_id, const bool multithreaded) {
  return createImageLoader<ColorImage>(
      std::bind(getPathForColorImage, base_path, seq_id, std::placeholders::_1),
      multithreaded);
}

std::unique_ptr<ImageLoader<DepthImage>> createHeightImageLoader(
    const std::string& base_path, const int seq_id, const bool multithreaded) {
  return createImageLoader<DepthImage>(
      std::bind(getPathForHeightImage, base_path, seq_id,
                std::placeholders::_1),
      multithreaded, kDefaultUintDepthScaleFactor,
      kDefaultUintDepthScaleOffset);
}

}  // namespace internal

std::unique_ptr<FuserLidar> createFuser(const std::string base_path,
                                        const int seq_id) {
  bool multithreaded = false;
  // Object to load FusionPortable data
  auto data_loader =
      std::make_unique<DataLoader>(base_path, seq_id, multithreaded);
  // FuserLidar
  return std::make_unique<FuserLidar>(std::move(data_loader));
}

DataLoader::DataLoader(const std::string& base_path, const int seq_id,
                       bool multithreaded)
    : RgbdDataLoaderInterface(kitti::internal::createDepthImageLoader(
                                  base_path, seq_id, multithreaded),
                              kitti::internal::createColorImageLoader(
                                  base_path, seq_id, multithreaded)),
      height_image_loader_(std::move(kitti::internal::createHeightImageLoader(
          base_path, seq_id, multithreaded))),
      base_path_(base_path),
      seq_id_(seq_id) {
  //
}

/// Interface for a function that loads the next frames in a dataset
///@param[out] depth_frame_ptr The loaded depth frame.
///@param[out] T_L_C_ptr Transform from the Base camera to the Layer frame.
///@param[out] camera_ptr The intrinsic camera model.
///@param[out] height_frame_ptr The loaded z frame.
///@param[out] color_frame_ptr Optional, load color frame.
///@return Whether loading succeeded.
DataLoadResult DataLoader::loadNext(DepthImage* depth_frame_ptr,
                                    Transform* T_L_C_ptr,
                                    CameraPinhole* camera_ptr,
                                    OSLidar* lidar_ptr,
                                    DepthImage* height_frame_ptr,
                                    ColorImage* color_frame_ptr) {
  CHECK_NOTNULL(depth_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(camera_ptr);
  CHECK_NOTNULL(lidar_ptr);
  CHECK_NOTNULL(height_frame_ptr);
  // CHECK_NOTNULL(color_frame_ptr);  // can be null

  // Because we might fail along the way, increment the frame number before we
  // start.
  const int frame_number = frame_number_;
  ++frame_number_;

  // *********************************************
  // *********************************************
  // *********************************************
  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    return DataLoadResult::kNoMoreData;
  }
  timer_file_depth.Stop();

  // Load the image into a Height Frame.
  CHECK(height_image_loader_);
  timing::Timer timer_file_coord("file_loading/height_image");
  if (!height_image_loader_->getNextImage(height_frame_ptr)) {
    return DataLoadResult::kNoMoreData;
  }
  timer_file_coord.Stop();

  // NOTE(gogojjh): Load lidar intrinsics:
  //  num_azimuth_divisions
  //  num_elevation_divisions
  //  horizontal_fov_rad
  //  vertical_fov_rad
  //  start_azimuth_angle_rad
  //  end_azimuth_angle_rad
  //  start_elevation_angle_rad
  //  end_elevation_angle_rad
  timing::Timer timer_file_camera("file_loading/lidar");
  Eigen::Matrix<double, 8, 1> lidar_intrinsics;
  if (!kitti::internal::parseLidarFromFile(
          kitti::internal::getPathForLidarIntrinsics(base_path_, seq_id_,
                                                     frame_number),
          &lidar_intrinsics)) {
    return DataLoadResult::kNoMoreData;
  }
  *lidar_ptr =
      OSLidar(lidar_intrinsics(0), lidar_intrinsics(1), lidar_intrinsics(2),
              lidar_intrinsics(3), lidar_intrinsics(4), lidar_intrinsics(5),
              lidar_intrinsics(6), lidar_intrinsics(7));
  CHECK(depth_frame_ptr->rows() == lidar_ptr->num_elevation_divisions());
  CHECK(depth_frame_ptr->cols() == lidar_ptr->num_azimuth_divisions());
  CHECK(height_frame_ptr->rows() == lidar_ptr->num_elevation_divisions());
  CHECK(height_frame_ptr->cols() == lidar_ptr->num_azimuth_divisions());

  // *********************************************
  // *********************************************
  // *********************************************
  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      return DataLoadResult::kNoMoreData;
    }
    timer_file_color.Stop();
  }

  // Get the camera for this frame.
  if (color_frame_ptr) {
    timing::Timer timer_file_camera("file_loading/camera");
    Matrix3f R_rect;
    Matrix3x4f P_rect;
    int image_width;
    int image_height;
    if (!kitti::internal::parseCameraFromFile(
            kitti::internal::getPathForCameraIntrinsics(base_path_), &P_rect,
            &R_rect, &image_height, &image_width)) {
      return DataLoadResult::kNoMoreData;
    }
    // std::cout << image_height << " " << image_width << std::endl
    //           << P_rect << std::endl
    //           << R_rect << std::endl;

    // Create a camera object.
    *camera_ptr = CameraPinhole::fromIntrinsicsMatrix(
        P_rect, R_rect, image_width, image_height);
    timer_file_camera.Stop();

    if (!P_rect.allFinite()) {
      LOG(WARNING) << "[P_rect] Bad CSV data.";
      return DataLoadResult::kBadFrame;  // Bad data, but keep going.
    }
    if (!R_rect.allFinite()) {
      LOG(WARNING) << "[R_rect] Bad CSV data.";
      return DataLoadResult::kBadFrame;  // Bad data, but keep going.
    }
  }

  // *********************************************
  // *********************************************
  // *********************************************
  // Get the transform.
  timing::Timer timer_file_pose("file_loading/pose");
  Transform T_O_C;
  if (!kitti::internal::parsePoseFromFile(
          kitti::internal::getPathForFramePose(base_path_, seq_id_,
                                               frame_number),
          &T_O_C)) {
    return DataLoadResult::kNoMoreData;
  }
  *T_L_C_ptr = T_O_C;

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

// NOTE(gogojjh): need to define the virutal function (not used) here
/// Interface for a function that loads the next frames in a dataset
///@param[out] depth_frame_ptr The loaded depth frame.
///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
///@param[out] camera_ptr The intrinsic camera model.
///@param[out] color_frame_ptr Optional, load color frame.
///@return Whether loading succeeded.
DataLoadResult DataLoader::loadNext(DepthImage* depth_frame_ptr,
                                    Transform* T_L_C_ptr, Camera* camera_ptr,
                                    ColorImage* color_frame_ptr) {
  return DataLoadResult::kNoMoreData;
}

}  // namespace kitti
}  // namespace datasets
}  // namespace nvblox
