/*
Copyright 2022-2023 NVIDIA CORPORATION

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

namespace nvblox {

constexpr float kWeightEpsilon = 1e-2;

WeightingFunction::WeightingFunction(WeightingFunctionType type)
    : type_(type) {}

const WeightingFunctionType& WeightingFunction::type() const { return type_; }

void WeightingFunction::type(const WeightingFunctionType type) { type_ = type; }

float WeightingFunction::operator()(float measured_depth, float voxel_depth,
                                    float truncation_distance) const {
  switch (type_) {
    case WeightingFunctionType::kConstantWeight:
      return constant_weight;
    case WeightingFunctionType::kConstantDropoffWeight:
      return constant_weight *
             computeDropoff(measured_depth, voxel_depth, truncation_distance);
    case WeightingFunctionType::kInverseSquareWeight:
      return computeInverseSquare(measured_depth, voxel_depth,
                                  truncation_distance);
    case WeightingFunctionType::kInverseSquareDropoffWeight:
      return computeInverseSquare(measured_depth, voxel_depth,
                                  truncation_distance) *
             computeDropoff(measured_depth, voxel_depth, truncation_distance);
    case WeightingFunctionType::kInverseSquareTsdfDistancePenalty:
      return computeInverseSquare(measured_depth, voxel_depth,
                                  truncation_distance) *
             computeTsdfDistancePenalty(measured_depth, voxel_depth,
                                        truncation_distance);
    default:
      LOG(FATAL) << "Requested weighting function type not implemented";
      return 0.0;
  };
}

// Computes the dropoff from 1 to 0.
float WeightingFunction::computeDropoff(float measured_depth, float voxel_depth,
                                        float truncation_distance) const {
  if (truncation_distance <= kWeightEpsilon) {
    return 0.0f;
  }
  // Behind surface - we're gonna drop the weight
  if (voxel_depth > measured_depth) {
    // Distance behind surface
    const float voxel_distance_to_surface = voxel_depth - measured_depth;
    // After the truncation band -> 0 weight
    if (voxel_distance_to_surface > truncation_distance) {
      return 0.0f;
    }
    // Distance behind surface as a fraction of the truncation band.
    // This fraction is returned as the weight.
    const float scaled_distance =
        (truncation_distance - voxel_distance_to_surface) / truncation_distance;
    return scaled_distance;
  }
  return 1.0f;
}

/// Returns 1/(z^2) where z is the voxel distance from the camera.
float WeightingFunction::computeInverseSquare(float measured_depth,
                                              float voxel_depth,
                                              float truncation_distance) const {
  // NOTE(alexmillane): This is only a function of the voxel distance from
  // camera. Ensure we don't get a divide by zero. Close to the cam ->
  // weight 1.0.
  if (voxel_depth <= kWeightEpsilon) {
    return 1.0f;
  }
  // After the truncation band -> 0 weight
  if (voxel_depth - measured_depth >= truncation_distance) {
    return 0.0f;
  }
  // Inverse square weight otherwise.
  return 1.0f / (voxel_depth * voxel_depth);
}

float WeightingFunction::computeTsdfDistancePenalty(
    float measured_depth, float voxel_depth, float truncation_distance) const {
  // Big tsdf distances measurements are susceptible to viewpoint changes.
  // Therefore we decrease the weight outside the truncation distance.
  // Note: can help to reduce holes in the floor reconstruction.
  const float tsdf_distance = measured_depth - voxel_depth;
  if (std::abs(tsdf_distance) >= truncation_distance) {
    // TODO(remos): Find optimal factor or make dependent on tsdf distance
    return 0.1f;
  }
  return 1.0f;
}

// Useful so we can print the weighting function types.
std::ostream& operator<<(std::ostream& os,
                         const WeightingFunctionType& weighting_function_type) {
  switch (weighting_function_type) {
    case WeightingFunctionType::kConstantWeight:
      os << "kConstantWeight";
      break;
    case WeightingFunctionType::kConstantDropoffWeight:
      os << "kConstantDropoffWeight";
      break;
    case WeightingFunctionType::kInverseSquareWeight:
      os << "kInverseSquareWeight";
      break;
    case WeightingFunctionType::kInverseSquareDropoffWeight:
      os << "kInverseSquareDropoffWeight";
      break;
    case WeightingFunctionType::kInverseSquareTsdfDistancePenalty:
      os << "kInverseSquareTsdfDistancePenalty";
    default:
      break;
  }
  return os;
}

std::string to_string(const WeightingFunctionType& weighting_function_type) {
  std::ostringstream ss;
  ss << weighting_function_type;
  return ss.str();
}

}  // namespace nvblox
