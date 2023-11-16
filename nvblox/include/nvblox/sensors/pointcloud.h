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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

constexpr MemoryType kDefaultPointcloudMemoryType = MemoryType::kDevice;

/// Pointcloud that lives in either device, host or unified memory.
/// We represent a pointcloud as a vector of 3D vectors.
class Pointcloud {
 public:
  /// Construct an empty pointcloud
  Pointcloud(int num_points,
             MemoryType memory_type = kDefaultPointcloudMemoryType);
  Pointcloud(MemoryType memory_type = kDefaultPointcloudMemoryType);

  /// Move operations
  Pointcloud(Pointcloud&& other) = default;
  Pointcloud& operator=(Pointcloud&& other) = default;
  Pointcloud(const Pointcloud& other) = delete;
  void copyFrom(const Pointcloud& other);
  void copyFromAsync(const Pointcloud& other, const CudaStream cuda_stream);
  void copyFrom(const std::vector<Vector3f>& points);
  void copyFromAsync(const std::vector<Vector3f>& points, const CudaStream cuda_stream);
  void copyFrom(const unified_vector<Vector3f>& points);
  void copyFromAsync(const unified_vector<Vector3f>& points, const CudaStream cuda_stream);

  /// Deep copy constructor (second can be used to transition memory type)
  /// Pointcloud(const Pointcloud& other);
  Pointcloud(const Pointcloud& other, MemoryType memory_type);
  Pointcloud& operator=(const Pointcloud& other);

  /// Expand memory available
  void resizeAsync(int num_points, const CudaStream cuda_stream) {
    points_.resizeAsync(num_points, cuda_stream);
  }
  void resize(int num_points) {
    points_.resizeAsync(num_points, CudaStreamOwning());
  }

  /// Attributes
  inline int num_points() const { return points_.size(); }
  inline int size() const { return points_.size(); }
  inline MemoryType memory_type() const { return points_.memory_type(); }
  inline bool empty() const { return points_.empty(); }

  /// Access
  /// NOTE(alexmillane): The guard-rails are off here. If you declare a kDevice
  /// Image and try to access its data, you will get undefined behaviour. If you
  /// access out of bounds, you're gonna have a bad time.
  inline const Vector3f& operator()(const int idx) const {
    return points_[idx];
  }
  inline Vector3f& operator()(const int idx) { return points_[idx]; }
  const unified_vector<Vector3f>& points() const { return points_; }
  unified_vector<Vector3f>& points() { return points_; }

  /// Add points
  void push_back(Vector3f&& point) { points_.push_back(point); }

  /// Raw pointer access
  inline Vector3f* dataPtr() { return points_.data(); }
  inline const Vector3f* dataConstPtr() const { return points_.data(); }

  /// Set the image to 0.
  void setZero() { points_.setZero(); }

 protected:
  unified_vector<Vector3f> points_;
};

/// Transforms the points in a pointcloud into another frame
/// @param T_out_in Transform that takes a point in frame "in" to frame "out".
/// @param pointcloud_in Pointcloud in frame "in".
/// @param[out] pointcloud_out Pointer to pointcloud in frame "out".
void transformPointcloudOnGPU(const Transform& T_out_in,        // NOLINT
                              const Pointcloud& pointcloud_in,  // NOLINT
                              Pointcloud* pointcloud_out_ptr);

/// Transforms the points in a pointcloud into another frame
/// See transformPointcloudOnGPU(). Same function just on a stream.
void transformPointcloudOnGPU(const Transform& T_out_in,        // NOLINT
                              const Pointcloud& pointcloud_in,  // NOLINT
                              Pointcloud* pointcloud_out_ptr,   // NOLINT
                              CudaStream* cuda_stream_ptr);

}  // namespace nvblox
