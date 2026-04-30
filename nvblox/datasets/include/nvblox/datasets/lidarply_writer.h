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

#include <memory>
#include <string>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {

/// @brief Helper class for writing lidar PLY pointclouds and poses to disk
/// in a PLY format.
class LidarPlyWriter {
 public:
  /// @param base_path Base path to the dataset folder containing the PLY files
  LidarPlyWriter(const std::string& base_path);

  /// @brief Write a pointcloud and its associated pose
  /// @param pointcloud The nvblox pointcloud to save
  /// @param pose The transform associated with this pointcloud
  /// @param lidar The lidar sensor object
  /// @param cuda_stream CUDA stream for asynchronous operations
  /// @param timestamp_ms Timestamp in milliseconds (Unix epoch)
  /// @return true if successful, false otherwise
  bool writeNext(const Pointcloud& pointcloud, const Transform& pose,
                 const Lidar& lidar, const CudaStream& cuda_stream,
                 Time timestamp_ms);

 private:
  /// Write a 4x4 transformation matrix to a file in 3dmatch/lidarply format
  /// @param filename Path to the output file
  /// @param transform The transformation matrix to write
  /// @return true if successful, false otherwise
  bool writePoseToFile(const std::string& filename, const Transform& transform);

  /// Write lidar intrinsics to a file in lidarply format
  /// @param filename Path to the output file
  /// @param lidar The lidar sensor object containing intrinsics
  /// @return true if successful, false otherwise
  bool writeLidarIntrinsicsToFile(const std::string& filename,
                                  const Lidar& lidar);

  /// Write timestamp to a file
  /// @param filename Path to the output file
  /// @param timestamp_ms Timestamp in milliseconds (Unix epoch)
  /// @return true if successful, false otherwise
  bool writeTimestampToFile(const std::string& filename, Time timestamp_ms);

  std::string base_path_;
  int frame_count_ = 0;
  bool intrinsics_saved_ = false;
  bool setup_success_ = true;
};

}  // namespace nvblox
