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
    default:
      break;
  }
  return os;
}

}  // namespace nvblox
