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

#include <limits>
#include <memory>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {

/// A class which takes care of back projecting images on the GPU.
class DepthImageBackProjector {
 public:
  DepthImageBackProjector();
  DepthImageBackProjector(std::shared_ptr<CudaStream> cuda_stream);
  ~DepthImageBackProjector() = default;

  /// Back projects a depth image to a pointcloud in the camera frame.
  ///@param image DepthImage to be back projected
  ///@param camera Pinhole camera intrinsics model
  ///@param pointcloud_C Pointer to the output pointcloud. Must be in either
  /// device or unified memory.
  ///@param max_back_projection_distance_m the maximum depth that is allowed for
  /// back projection. Pixel with bigger depth are not included in the outptut
  /// pointcloud.
  void backProjectOnGPU(const DepthImage& image, const Camera& camera,
                        Pointcloud* pointcloud_C_ptr,
                        const float max_back_projection_distance_m =
                            std::numeric_limits<float>::max());

  /// Takes a collection of points, and returns the center of the voxels that
  /// contain the points. Note that this function deletes duplicates, that is
  /// that the output pointcloud may have less points than the input pointcloud.
  ///@param pointcloud_L A collection of points in the Layer frame.
  ///@param voxel_size The side length of the voxels in the layer.
  ///@param voxel_center_pointcloud_L Voxel centers stored as a pointcloud.
  void pointcloudToVoxelCentersOnGPU(const Pointcloud& pointcloud_L,
                                     float voxel_size,
                                     Pointcloud* voxel_center_pointcloud_L);

 private:
  unified_ptr<int> pointcloud_size_device_;
  unified_ptr<int> pointcloud_size_host_;

  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox