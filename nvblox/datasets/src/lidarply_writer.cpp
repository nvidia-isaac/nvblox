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
#include "nvblox/datasets/lidarply_writer.h"

#include <filesystem>
#include <fstream>

#include <nvblox/datasets/lidarply_loader.h>
#include <nvblox/io/pointcloud_io.h>

namespace nvblox {

// Hardcoded sequence ID
constexpr int kSeqId = 1;

LidarPlyWriter::LidarPlyWriter(const std::string& base_path)
    : base_path_(base_path) {
  if (!std::filesystem::exists(base_path_)) {
    LOG(ERROR) << "Lidar PLY writer: Directory does not exist: " << base_path_;
    setup_success_ = false;
    return;
  }
  // Create sequence directory if it doesn't exist
  std::stringstream ss;
  ss << base_path_ << "/seq-" << std::setfill('0') << std::setw(2) << kSeqId
     << "/";
  std::filesystem::create_directories(ss.str());
}

bool LidarPlyWriter::writeNext(const Pointcloud& pointcloud,
                               const Transform& pose, const Lidar& lidar,
                               const CudaStream& cuda_stream,
                               Time timestamp_ms) {
  if (!setup_success_) {
    LOG(ERROR) << "Lidar PLY writer: Not setup successfully";
    return false;
  }

  // Generate filenames
  const std::string ply_filename =
      datasets::lidarply::internal::getPathForPointcloud(base_path_, kSeqId,
                                                         frame_count_);
  const std::string pose_filename =
      datasets::lidarply::internal::getPathForFramePose(base_path_, kSeqId,
                                                        frame_count_);
  const std::string timestamp_filename =
      datasets::lidarply::internal::getPathToFrameTimestampFile(
          base_path_, kSeqId, frame_count_);
  frame_count_++;

  // Save pointcloud to PLY file
  if (!io::outputPointcloudToPly(pointcloud, ply_filename, cuda_stream)) {
    LOG(ERROR) << "Failed to save pointcloud to: " << ply_filename;
    return false;
  }

  // Save pose to file
  if (!writePoseToFile(pose_filename, pose)) {
    LOG(ERROR) << "Failed to save pose to: " << pose_filename;
    return false;
  }

  // Save timestamp to file
  if (!writeTimestampToFile(timestamp_filename, timestamp_ms)) {
    LOG(ERROR) << "Failed to save timestamp to: " << timestamp_filename;
    return false;
  }

  // Save lidar intrinsics to file if not already saved
  if (!intrinsics_saved_) {
    const std::string intrinsics_filename =
        datasets::lidarply::internal::getPathForLidarIntrinsics(base_path_);

    if (!writeLidarIntrinsicsToFile(intrinsics_filename, lidar)) {
      LOG(ERROR) << "Failed to save lidar intrinsics to: "
                 << intrinsics_filename;
      return false;
    }

    intrinsics_saved_ = true;
    LOG(INFO) << "Saved lidar intrinsics to: " << intrinsics_filename;
  }

  return true;
}

bool LidarPlyWriter::writePoseToFile(const std::string& filename,
                                     const Transform& transform) {
  std::ofstream pose_file(filename);
  if (!pose_file.is_open()) {
    LOG(ERROR) << "Failed to open pose file for writing: " << filename;
    return false;
  }

  const auto& matrix = transform.matrix();
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      pose_file << matrix(row, col);
      if (col < 3) {
        pose_file << " ";
      }
    }
    pose_file << "\n";
  }

  pose_file.close();
  return true;
}

bool LidarPlyWriter::writeLidarIntrinsicsToFile(const std::string& filename,
                                                const Lidar& lidar) {
  std::ofstream intrinsics_file(filename);
  if (!intrinsics_file.is_open()) {
    LOG(ERROR) << "Failed to open lidar intrinsics file for writing: "
               << filename;
    return false;
  }

  intrinsics_file << "# Lidar intrinsics\n";
  intrinsics_file << "# num_azimuth_divisions num_elevation_divisions\n";
  intrinsics_file << lidar.num_azimuth_divisions() << " "
                  << lidar.num_elevation_divisions() << "\n";
  intrinsics_file << "# min_valid_range_m\n";
  intrinsics_file << lidar.min_valid_range_m() << "\n";
  intrinsics_file << "# min_angle_below_zero_elevation_rad "
                     "max_angle_above_zero_elevation_rad\n";
  intrinsics_file << lidar.min_angle_below_zero_elevation_rad() << " "
                  << lidar.max_angle_above_zero_elevation_rad() << "\n";

  intrinsics_file.close();
  return true;
}

bool LidarPlyWriter::writeTimestampToFile(const std::string& filename,
                                          Time timestamp_ms) {
  std::ofstream timestamp_file(filename);
  if (!timestamp_file.is_open()) {
    LOG(ERROR) << "Failed to open timestamp file for writing: " << filename;
    return false;
  }

  // Write the timestamp in milliseconds
  timestamp_file << timestamp_ms << "ms\n";

  timestamp_file.close();
  return true;
}

}  // namespace nvblox
