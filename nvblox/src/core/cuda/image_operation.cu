#include "nvblox/core/cuda/error_check.cuh"
#include "nvblox/core/cuda/image_operation.h"

namespace nvblox {
namespace cuda {
__global__ void computeNormalImageOSLidar(
    const float* depth_image, const float* height_image, float* normal_image,
    const int w, const int h, const float rads_per_pixel_azimuth,
    const float rads_per_pixel_elevation) {
  const float end_azimuth_rad = 3.1415926535f;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int u_stride = blockDim.x;
  int v_stride = 1;
  for (int u = tid; u < w; u += u_stride) {
    for (int v = 0; v < h; v += v_stride) {
      normal_image[3 * (v * w + u)] = 0.0f;
      normal_image[3 * (v * w + u) + 1] = 0.0f;
      normal_image[3 * (v * w + u) + 2] = 0.0f;

      float sign = 1.0f;
      int uu, vv;
      if (u == w - 1) {
        uu = 0;
      } else {
        uu = u + 1;
      }
      if (v == h - 1) {
        vv = 0;
        sign *= -1.0f;
      } else {
        vv = v + 1;
      }

      float d = image::access<float>(v, u, w, depth_image);
      float d1 = image::access<float>(v, uu, w, depth_image);
      float d2 = image::access<float>(vv, u, w, depth_image);
      // on the boundary, not continous
      if (abs(d - d1) > 0.9 * d) continue;
      // on the boundary, not continous
      if (abs(d - d2) > 0.9 * d) continue;

      float px, py, pz;
      float px1, py1, pz1;
      float px2, py2, pz2;

      {
        float depth = image::access<float>(v, u, w, depth_image);
        float height = image::access<float>(v, u, w, height_image);
        float r = sqrt(depth * depth - height * height);
        float azimuth_angle_rad = end_azimuth_rad - u * rads_per_pixel_azimuth;
        px = r * cos(azimuth_angle_rad);
        py = r * sin(azimuth_angle_rad);
        pz = height;
      }

      {
        float depth = image::access<float>(v, uu, w, depth_image);
        float height = image::access<float>(v, uu, w, height_image);
        float r = sqrt(depth * depth - height * height);
        float azimuth_angle_rad = end_azimuth_rad - uu * rads_per_pixel_azimuth;
        px1 = r * cos(azimuth_angle_rad);
        py1 = r * sin(azimuth_angle_rad);
        pz1 = height;
      }

      {
        float depth = image::access<float>(vv, u, w, depth_image);
        float height = image::access<float>(vv, u, w, height_image);
        float r = sqrt(depth * depth - height * height);
        float azimuth_angle_rad = end_azimuth_rad - u * rads_per_pixel_azimuth;
        px2 = r * cos(azimuth_angle_rad);
        py2 = r * sin(azimuth_angle_rad);
        pz2 = height;
      }

      float nx, ny, nz;
      {
        nx = sign * (py - py2) * (pz - pz1) - (py - py1) * (pz - pz2);
        ny = sign * (px - px1) * (pz - pz2) - (px - px2) * (pz - pz1);
        nz = sign * (px - px2) * (py - py1) - (px - px1) * (py - py2);
        float l = sqrt(nx * nx + ny * ny + nz * nz);
        if (l == 0.0f) {
          continue;
        } else {
          nx /= l;
          ny /= l;
          nz /= l;
        }
        // printf("%f %f, %f, %f\n", l, nx, ny, nz);
      }
      normal_image[3 * (v * w + u)] = nx;
      normal_image[3 * (v * w + u) + 1] = ny;
      normal_image[3 * (v * w + u) + 2] = nz;
    }
  }
}

// OSLidar
void getNormalImageOSLidar(OSLidar& lidar) {
  int w = lidar.num_azimuth_divisions();
  int h = lidar.num_elevation_divisions();
  float* normal_frame_cuda;
  checkCudaErrors(cudaMalloc(&normal_frame_cuda, sizeof(float) * w * h * 3));
  int block_size = idivup(w, 2);
  int grid_size = 1;
  computeNormalImageOSLidar<<<grid_size, block_size>>>(
      lidar.getDepthFrameCUDA(), lidar.getHeightFrameCUDA(), normal_frame_cuda,
      lidar.num_azimuth_divisions(), lidar.num_elevation_divisions(),
      lidar.rads_per_pixel_azimuth(), lidar.rads_per_pixel_elevation());
  lidar.setNormalFrameCUDA(normal_frame_cuda);
}

void freeNormalImageOSLidar(OSLidar& lidar) {
  float* normal_image = lidar.getNormalFrameCUDA();
  checkCudaErrors(cudaFree(normal_image));
}

}  // namespace cuda
}  // namespace nvblox