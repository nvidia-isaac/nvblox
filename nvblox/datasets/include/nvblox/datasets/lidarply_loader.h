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
#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/data_loader_interface.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {
namespace datasets {
namespace lidarply {

///@brief A class for loading lidar PLY pointcloud data
class DataLoader : public LidarDataLoaderInterface {
 public:
  /// Constructor
  ///@param base_path Base path to the dataset folder containing the PLY files
  ///@param seq_id Sequence index of the dataset to be loaded
  DataLoader(const std::string& base_path, const int seq_id);
  virtual ~DataLoader() = default;

  /// Builds a DataLoader
  ///@param base_path Path to the dataset folder
  ///@param seq_id Sequence index of the dataset to be loaded
  ///@return std::unique_ptr<DataLoader> The dataset loader. May be nullptr if
  /// construction fails.
  static std::unique_ptr<DataLoader> create(const std::string& base_path,
                                            const int seq_id);

  /// Lidarply datasets provide frame timestamps
  bool provides_frame_timestamps() const override { return true; }

  bool provides_lidar_scan_data() const override { return true; }

  /// Interface for a function that loads the next frames in a dataset.
  /// Loads pointcloud with per-point timestamps. Color parameters are ignored.
  ///@param[out] pointcloud_ptr The loaded pointcloud with per-point timestamps.
  ///@param[out] T_L_S_scanStart_ptr Transform from lidar sensor to the
  /// Layer frame at
  /// scan start.
  ///@param[out] lidar_ptr The intrinsic lidar model.
  ///@param[out] unused Needed to match data loader interface (pass nullptr).
  ///@param[out] unused Needed to match data loader interface (pass nullptr).
  ///@param[out] unused Needed to match data loader interface (pass nullptr).
  ///@param[out] frame_timestamp_ms_ptr Frame timestamp in milliseconds.
  ///@param[out] T_L_S_scanEnd_ptr Transform from lidar sensor to the
  /// Layer frame at scan end.
  ///@param[out] scan_duration_ms_ptr Lidar scan duration in milliseconds.
  ///@return Whether loading succeeded.
  DataLoadResult loadNext(Pointcloud* pointcloud_ptr,            // NOLINT
                          Transform* T_L_S_scanStart_ptr,        // NOLINT
                          Lidar* lidar_ptr,                      // NOLINT
                          ColorImage*,                           // NOLINT
                          Transform*,                            // NOLINT
                          Camera*,                               // NOLINT
                          Time* frame_timestamp_ms_ptr,          // NOLINT
                          Transform* T_L_S_scanEnd_ptr,          // NOLINT
                          Time* scan_duration_ms_ptr) override;  // NOLINT

 protected:
  /// Helper function to load pose and timestamp for a given frame number
  /// @param frame_num The frame number to load
  /// @param pose_ptr Output pointer for the transform
  /// @param time_ptr Output pointer for the timestamp
  /// @return True if loading succeeded, false otherwise
  bool loadPoseAndTimestamp(int frame_num, Transform* pose_ptr, Time* time_ptr);

  const std::string base_path_;
  const int seq_id_;

  // The next frame to be loaded
  int frame_number_ = 0;

  // Cached lidar intrinsics
  Lidar lidar_;
  bool lidar_loaded_ = false;

  // CUDA stream
  CudaStreamOwning cuda_stream_;

  // Temporary storage for pointcloud
  Pointcloud pointcloud_{MemoryType::kDevice};

  // Indicates if the dataset loader was constructed successfully
  bool setup_success_ = true;
};

namespace internal {

/// Parse lidar intrinsics from a text file
bool parseLidarFromFile(const std::string& filename, Lidar* lidar);

/// Get the path to the lidar intrinsics file
std::string getPathForLidarIntrinsics(const std::string& base_path);

/// Parse a 4x4 transformation matrix from a text file (calls 3dmatch version)
inline bool parsePoseFromFile(const std::string& filename,
                              Transform* transform) {
  return threedmatch::internal::parsePoseFromFile(filename, transform);
}

/// Get the path to a frame's pose file (calls 3dmatch version)
inline std::string getPathForFramePose(const std::string& base_path,
                                       const int seq_id, const int frame_id) {
  return threedmatch::internal::getPathForFramePose(base_path, seq_id,
                                                    frame_id);
}

/// Get the path to a frame's pointcloud PLY file
std::string getPathForPointcloud(const std::string& base_path, const int seq_id,
                                 const int frame_id);

/// Get the path to a frame's timestamp file
std::string getPathToFrameTimestampFile(const std::string& base_path,
                                        const int seq_id, const int frame_id);

/// Parse timestamp from a text file
bool parseTimestampFromFile(const std::string& filename, Time* timestamp_ms);

}  // namespace internal
}  // namespace lidarply
}  // namespace datasets
}  // namespace nvblox
