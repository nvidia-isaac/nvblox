/*
Copyright 2024 NVIDIA CORPORATION

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

#include <curand_kernel.h>
#include <vector>

#include "nvblox/core/indexing.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter_params.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/pointcloud.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

class RansacPlaneFitter {
 public:
  RansacPlaneFitter() = delete;
  RansacPlaneFitter(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~RansacPlaneFitter() = default;

  /// @brief Fits a plane to a given point cloud.
  /// @param point_cloud The input vector of 3D points to which the plane is to
  /// be fitted.
  /// @return The best-fitting plane if found, otherwise std::nullopt.
  std::optional<Plane> fit(const Pointcloud& point_cloud);

  /// A parameter setter
  /// @param ransac_distance_threshold the maximum distance a point
  /// can be from a plane to be considered an inlier.
  void ransac_distance_threshold_m(float distance_threshold);

  /// A parameter getter
  /// @returns distance_threshold the maximum distance a point
  /// can be from a plane to be considered an inlier.
  float ransac_distance_threshold_m() const;

  /// A parameter setter
  /// @param num_ransac_iterations to run the RANSAC algorithm.
  void num_ransac_iterations(int num_ransac_iterations);

  /// A parameter getter
  /// @returns the number of iterations to run the RANSAC algorithm.
  int num_ransac_iterations() const;

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  // RANSAC configurations.
  // The maximum distance a point can be from a plane to be considered an
  // inlier.
  float ransac_distance_threshold_m_{
      kRansacDistanceThresholdMDesc.default_value};
  // The number of iterations to run the RANSAC algorithm.
  int num_ransac_iterations_{kNumRansacIterationsDesc.default_value};

  // Computed planes over all iterations on device.
  device_vector<Plane> planes_device_;
  // Computed planes over all iterations on host.
  host_vector<Plane> planes_host_;

  // Computed cost over all iterations on device.
  device_vector<float> costs_device_;
  // Computed cost over all iterations on host.
  host_vector<float> costs_host_;
  // Initialized cost on host.
  std::vector<float> initial_costs_host_;

  // The CUDA stream on which processing occurs.
  std::shared_ptr<CudaStream> cuda_stream_;
  // Holding random states to be used during Kernel execution.
  device_vector<curandState> random_states_device_;
};

}  // namespace nvblox
