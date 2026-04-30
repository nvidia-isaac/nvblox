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

#include <random>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/ground_plane_estimator_params.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter.h"
#include "nvblox/experimental/ground_plane/tsdf_zero_crossings_extractor.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {

class GroundPlaneEstimator {
 public:
  /// Initializes the TSDF zero-crossings extractor and RANSAC plane fitter
  /// using the provided CUDA stream and configuration. Sets the range for
  /// filtering ground candidate points based on their z height.
  /// @param cuda_stream Shared pointer to a CUDA stream for GPU operations.
  GroundPlaneEstimator(std::shared_ptr<CudaStream> cuda_stream);

  /// Computes the zero-crossings in the TSDF layer, filters them based on the
  /// specified z-height range, and attempts to fit a plane using the RANSAC
  /// method.
  /// If unsuccesfull, resets internal members and returns std::nullopt.
  /// @param tsdf_layer The TSDF layer used for zero-crossings extraction.
  /// @return std::optional<Plane> The estimated ground plane if successful,
  /// std::nullopt otherwise.
  std::optional<Plane> computeGroundPlane(const TsdfLayer& tsdf_layer);

  /// @brief Gets the TSDF zero-crossing points.
  std::optional<std::vector<Vector3f>> tsdf_zero_crossings() const;
  /// @brief Gets the filtered (based on min/max z height)
  /// TSDF zero-crossing points.
  std::optional<std::vector<Vector3f>> tsdf_zero_crossings_ground_candidates()
      const;
  /// @brief Gets the estimated ground plane.
  std::optional<Plane> ground_plane() const;

  /// @brief Gets the minumum z height at which tsdf zero crossings are
  /// considered ground points.
  float ground_points_candidates_min_z_m() const;
  /// @brief Gets the maximum z height up to which tsdf zero crossings are
  /// considered ground points.
  float ground_points_candidates_max_z_m() const;

  /// @brief Sets the minumum z height at which tsdf zero crossings are
  /// considered ground points.
  void ground_points_candidates_min_z_m(float ground_points_candidates_min_z_m);
  /// @brief Sets the maximum z height up to which tsdf zero crossings are
  /// considered ground points.
  void ground_points_candidates_max_z_m(float ground_points_candidates_max_z_m);

  /// Access to the ransac_plane_fitter
  RansacPlaneFitter& ransac_plane_fitter() { return ransac_plane_fitter_; };

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  /// @brief Resets the internal optional members by setting them to
  /// `std::nullopt`.
  void resetInternal();

  /// @brief Extractor for TSDF zero-crossings
  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor_;
  /// @brief RANSAC-based plane fitter used to estimate the ground plane.
  RansacPlaneFitter ransac_plane_fitter_;
  /// @brief Shared pointer to a CUDA stream used for GPU operations.
  std::shared_ptr<CudaStream> cuda_stream_;

  /// @brief Holds the found tsdf zero crossings if succesfull.
  std::optional<std::vector<Vector3f>> tsdf_zero_crossings_;
  /// @brief Represents the filtered TSDF zero-crossing candidates
  /// that are within the specified z-height range, if they exist.
  std::optional<std::vector<Vector3f>> tsdf_zero_crossings_ground_candidates_;
  /// @brief Pointcloud object of the filtered TSDF zero-crossing
  /// candidates that are within the specified z-height range, if they exist.
  Pointcloud tsdf_zero_crossings_ground_candidates_point_cloud_;

  /// @brief The most recently estimated ground plane, if one exists.
  std::optional<Plane> ground_plane_;

  /// @brief Minimum z-height for filtering ground candidate points.
  float ground_points_candidates_min_z_m_{
      kGroundPointsCandidatesMinZMDesc.default_value};
  /// @brief Maximum z-heigt for filtering ground candidate points.
  float ground_points_candidates_max_z_m_{
      kGroundPointsCandidatesMaxZMDesc.default_value};
};

// Filters a point cloud to include only points within the specified Z range,
// optionally skipping invalid points (contining nan or inf).
std::vector<Vector3f> getPointsWithinMinMaxZCPU(
    const std::vector<Vector3f>& point_cloud, float min_z, float max_z,
    bool skip_invalid = true);

}  // namespace nvblox
