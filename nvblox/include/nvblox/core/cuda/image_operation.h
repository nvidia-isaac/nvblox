#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#include "nvblox/core/oslidar.h"

namespace nvblox {
namespace cuda {

__host__ __device__ inline int idivup(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// NOTE(jjiao): we cannot directly give the value from CPU and avoid
// dangerous operations, like bug1: int w_ = lidar.num_azimuth_divisions();
// bug2: int h_ = lidar.num_elevation_divisions();
// bug3: cannot print in the loop
// we cannot correctly use values of w_ and h_
__global__ void computeNormalImageOSLidar(const float* depth_image,
                                          const float* height_image,
                                          float* normal_image, const int w,
                                          const int h,
                                          const float rads_per_pixel_azimuth,
                                          const float rads_per_pixel_elevation);

void getNormalImageOSLidar(OSLidar& lidar);

}  // namespace cuda
}  // namespace nvblox
