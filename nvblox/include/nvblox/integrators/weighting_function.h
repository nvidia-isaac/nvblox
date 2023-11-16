#pragma once

#include <cuda_runtime.h>

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/voxels.h"

namespace nvblox {

enum class WeightingFunctionType {
  kConstantWeight,
  kConstantDropoffWeight,
  kInverseSquareWeight,
  kInverseSquareDropoffWeight,
  kInverseSquareTsdfDistancePenalty
};

inline std::ostream& operator<<(
    std::ostream& os, const WeightingFunctionType& weighting_function_type);
inline std::string to_string(
    const WeightingFunctionType& weighting_function_type);

/** Class encapsulating the supported weighting types. Unfortunately CUDA does
 * not support polymorphism/inheritance in any way that's usable in this
 * context, so we have to go with a monolithic class. */
class WeightingFunction {
 public:
  __host__ __device__ inline WeightingFunction(WeightingFunctionType type);
  __host__ __device__ ~WeightingFunction() = default;

  /// Returns the weight of the given voxel, depending on the type.

  /// Computes the weight of this observation, depending on the selected sensor
  /// model.
  /// @param measured_depth The depth measured by the depth sensor
  /// @param voxel_depth The depth of the voxel being updated
  /// @param truncation_distance The truncation distance
  /// @return The weight that this observation should get.
  __host__ __device__ inline float operator()(float measured_depth,
                                              float voxel_depth,
                                              float truncation_distance) const;

  /// A getter
  /// The type of the weighting to be used when the operator() is called.
  /// @return The type of the weighting function
  __host__ __device__ inline const WeightingFunctionType& type() const;

  /// A setter
  /// See type()
  /// @param type The type of weighting to be used
  __host__ __device__ inline void type(const WeightingFunctionType type);

 private:
  __host__ __device__ inline float computeDropoff(
      float measured_depth, float voxel_depth, float truncation_distance) const;

  __host__ __device__ inline float computeInverseSquare(
      float measured_depth, float voxel_depth, float truncation_distance) const;

  __host__ __device__ inline float computeTsdfDistancePenalty(
      float measured_depth, float voxel_depth, float truncation_distance) const;

  WeightingFunctionType type_;

  constexpr static float constant_weight = 1.0f;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/weighting_function_impl.h"
