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
#include "nvblox/datasets/parse_3dmatch.h"

#include <glog/logging.h>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

namespace nvblox {
namespace datasets {
namespace threedmatch {

// Parses a 4x4 matrix from a text file in 3D Match format: space-separated.
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

// Parse 3x3 camera intrinsics matrix from 3D Match format: space-separated.
bool parseCameraFromFile(const std::string& filename,
                         Eigen::Matrix3f* intrinsics) {
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
    fin.close();

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

std::unique_ptr<nvblox::datasets::ImageLoader<DepthImage>>
createDepthImageLoader(const std::string& base_path, const int seq_id,
                       MemoryType memory_type) {
  return std::make_unique<nvblox::datasets::ImageLoader<DepthImage>>(
      std::bind(getPathForDepthImage, base_path, seq_id, std::placeholders::_1),
      memory_type);
}

std::unique_ptr<nvblox::datasets::ImageLoader<ColorImage>>
createColorImageLoader(const std::string& base_path, const int seq_id,
                       MemoryType memory_type) {
  return std::make_unique<nvblox::datasets::ImageLoader<ColorImage>>(
      std::bind(getPathForColorImage, base_path, seq_id, std::placeholders::_1),
      memory_type);
}

std::unique_ptr<nvblox::datasets::ImageLoader<DepthImage>>
createMultithreadedDepthImageLoader(const std::string& base_path,
                                    const int seq_id, const int num_threads,
                                    MemoryType memory_type) {
  return std::make_unique<
      nvblox::datasets::MultiThreadedImageLoader<DepthImage>>(
      std::bind(getPathForDepthImage, base_path, seq_id, std::placeholders::_1),
      num_threads, memory_type);
}

std::unique_ptr<nvblox::datasets::ImageLoader<ColorImage>>
createMultithreadedColorImageLoader(const std::string& base_path,
                                    const int seq_id, const int num_threads,
                                    MemoryType memory_type) {
  return std::make_unique<
      nvblox::datasets::MultiThreadedImageLoader<ColorImage>>(
      std::bind(getPathForColorImage, base_path, seq_id, std::placeholders::_1),
      num_threads, memory_type);
}

}  // namespace threedmatch
}  // namespace datasets
}  // namespace nvblox
