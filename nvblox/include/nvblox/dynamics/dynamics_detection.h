/*
Copyright 2023 NVIDIA CORPORATION

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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {

/// A class for detecting dynamic objects.
///
/// It takes a depth frame and compares it to the freespace layer.
/// If any surface seen on the depth image falls into freespace it is assumed to
/// be dynamic.
class DynamicsDetection {
 public:
  DynamicsDetection();
  DynamicsDetection(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~DynamicsDetection() = default;

  /// @brief Detect dynamics on the depth frames by comparing it to the tsdf
  /// @param depth_frame_C The depth frame.
  /// @param tsdf_layer_L  The tsdf layer.
  /// @param camera The camera that belongs to the depth frame.
  /// @param T_L_C  The transform from camera to tsdf layer.
  void computeDynamics(const DepthImage& depth_frame_C,
                       const FreespaceLayer& tsdf_layer_L, const Camera& camera,
                       const Transform& T_L_C);

  /// @brief Gets the 3D points detected as dynamics on the last depth frame.
  /// @return The dynamic points as a 3*X matrix on host.
  Eigen::Matrix3Xf getDynamicPointsHost();

  /// @brief Gets the 3D points detected as dynamics on the last depth frame.
  /// @return The dynamic points as a pointcloud object on device.
  const Pointcloud& getDynamicPointcloudDevice();

  /// @brief Gets the dynamic mask of the last depth frame (dynamics labelled as
  /// 1, static pixels as 0).
  /// @return The dynamic mask image.
  const MonoImage& getDynamicMaskImage() const;

  /// @brief Gets the dynamic overlay image of the last depth frame (dynamics
  /// marked in red color).
  /// @return The dynamic overlay image.
  const ColorImage& getDynamicOverlayImage() const;

 private:
  // Helper functions
  void prepareOutputs(const DepthImage& input_frame);

  // Image buffers
  ColorImage dynamics_overlay_{MemoryType::kDevice};
  MonoImage dynamics_mask_{MemoryType::kDevice};

  // Dynamic point counters and buffers
  unified_ptr<int> dynamic_points_counter_host_;
  unified_ptr<int> dynamic_points_counter_device_;
  host_vector<Vector3f> dynamic_points_host_;
  device_vector<Vector3f> dynamic_points_device_;
  Pointcloud dynamic_pointcloud_device_;

  // CUDA stream to process dynamics detection on
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
