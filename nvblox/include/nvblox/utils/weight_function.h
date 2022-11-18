#pragma once

#include <cmath>

namespace nvblox {

__host__ __device__ inline float tsdf_constant_weight(const float& sdf) {
  return 1.0f;
}

__host__ __device__ inline float tsdf_linear_weight(const float& sdf,
                                                    const float& trunc) {
  float epsilon = trunc * 0.5f;
  float a = 0.5f * trunc / (trunc - epsilon);
  float b = 0.5f / (trunc - epsilon);
  if (sdf < epsilon) return 1.0f;
  if ((epsilon <= sdf) && (sdf <= trunc)) return (a - b * sdf + 0.5f);
  return 0.5f;
}

__host__ __device__ inline float tsdf_exp_weight(const float& sdf,
                                                 const float& trunc) {
  float epsilon = trunc * 0.5f;
  if (sdf < epsilon) return 1.0f;
  if ((epsilon <= sdf) && (sdf <= trunc)) {
    float d = -trunc * (sdf - epsilon) * (sdf - epsilon);
    return exp(d);
  } else {
    float d = -trunc * (trunc - epsilon) * (trunc - epsilon);
    return exp(d);
  }
}

}  // namespace nvblox
