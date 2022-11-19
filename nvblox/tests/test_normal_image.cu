#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../executables/include/nvblox/datasets/external/stb_image.h"

#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

const float FACTOR = 1000.0f;
const float OFFSET = 10.0f;

struct LidarIntrinsics {
  int width = 2048;
  int height = 128;
  float horizontal_fov = 6.28319;
  float vertical_fov = 0.73584;
  float start_azimuth = 0.0;
  float end_azimuth = 6.28319;
  float start_elevation = 1.19763;
  float end_elevation = 1.93347;
  float rads_per_pixel_azimuth = horizontal_fov * 1.0 / (width - 1);
  float rads_per_pixel_elevation = vertical_fov * 1.0 / (height - 1);
};

std::vector<float> load16BitImage(const std::string& filename,
                                  const float factor, const float offset) {
  int width, height, num_channels;
  uint16_t* image_data =
      stbi_load_16(filename.c_str(), &width, &height, &num_channels, 0);

  std::vector<float> float_image_data(height * width);
  for (int lin_idx = 0; lin_idx < float_image_data.size(); lin_idx++) {
    float_image_data[lin_idx] =
        static_cast<float>(image_data[lin_idx]) / factor - offset;
  }
  stbi_image_free(image_data);
  return float_image_data;
}

__host__ __device__ inline int idivup(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

template <typename ElementType>
__host__ __device__ inline ElementType access(int row_idx, int col_idx,
                                              int cols,
                                              const ElementType* data) {
  return data[row_idx * cols + col_idx];
}

__global__ void computeNormalImage(float* depth_image, float* height_image,
                                   float* normal_image, const int width,
                                   const int height,
                                   const LidarIntrinsics lidar_intrinsics) {
  // method 1:
  // int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // int stride = blockDim.x;
  // for (int u = tid; u < width; u += stride) {
  //   for (int v = 0; v < height; v++) {
  //     int uu, vv;
  //     if (u == width - 1) uu = 0;
  //     if (v == height - 1) vv = 0;
  //     normal_image_x[v * width + u] =
  //         depth_image[v * width + u] - depth_image[vv * width + uu];
  //   }
  // }

  // method 2:
  // int tid = threadIdx.x;
  // int u_stride = blockDim.x;
  // int v_stride = 128 / 16;
  // for (int u = tid; u < width; u += u_stride) {
  //   for (int v = blockIdx.x * v_stride; v < (blockIdx.x + 1) * v_stride; v++)
  //   {
  //     int uu, vv;
  //     if (u == width - 1) uu = 0;
  //     if (v == height - 1) vv = 0;
  //     normal_image_x[v * width + u] =
  //         depth_image[v * width + u] - depth_image[vv * width + uu];
  //   }
  // }

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x;
  for (int u = tid; u < width; u += stride) {
    for (int v = 0; v < height; v++) {
      // int uu, vv;
      // if (u == width - 1) uu = 0;
      // if (v == height - 1) vv = 0;
      // normal_image_x[v * width + u] =
      //     depth_image[v * width + u] - depth_image[vv * width + uu];
      normal_image[3 * (v * width + u)] = 1.0;
      normal_image[3 * (v * width + u) + 1] = 0.0;
      normal_image[3 * (v * width + u) + 2] = 0.0;
    }
  }
}

int main(int argc, char** argv) {
  int width = 2048;
  int height = 128;

  // start: read the image
  std::vector<float> depth_image = load16BitImage(
      std::string(
          "/Spy/dataset/mapping_results/nvblox/20220216_garden_day/seq-01/"
          "frame-000000.depth.png"),
      FACTOR, 0.0f);
  std::vector<float> height_image = load16BitImage(
      std::string(
          "/Spy/dataset/mapping_results/nvblox/20220216_garden_day/seq-01/"
          "frame-000000.height.png"),
      FACTOR, OFFSET);

  float* depth_image_cuda;
  float* height_image_cuda;
  cudaMalloc((void**)&depth_image_cuda, sizeof(float) * 2048 * 128);
  cudaMemcpy(depth_image_cuda, depth_image.data(), sizeof(float) * 2048 * 128,
             cudaMemcpyHostToDevice);
  cudaMalloc((void**)&height_image_cuda, sizeof(float) * 2048 * 128);
  cudaMemcpy(height_image_cuda, height_image.data(), sizeof(float) * 2048 * 128,
             cudaMemcpyHostToDevice);

  LidarIntrinsics lidar_intrinsics;
  printf("OSLidar intrinsics--------------------\n");
  printf("width: %d\n", lidar_intrinsics.width);
  printf("height: %d\n", lidar_intrinsics.height);
  printf("horizontal_fov_rad: %f\n", lidar_intrinsics.horizontal_fov);
  printf("vertical_fov_rad: %f\n", lidar_intrinsics.vertical_fov);
  printf("start_elevation: %f\n", lidar_intrinsics.start_elevation);
  printf("end_elevation: %f\n", lidar_intrinsics.end_elevation);
  printf("rads_per_pixel_azimuth: %f\n",
         lidar_intrinsics.rads_per_pixel_azimuth);
  printf("rads_per_pixel_elevation: %f\n",
         lidar_intrinsics.rads_per_pixel_elevation);

  //////////////////////////////// function 1: compute normal images
  int block_size = 256;
  int grid_size = 1;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  float* normal_image_cuda;
  cudaMalloc((void**)&normal_image_cuda, sizeof(float) * 2048 * 128 * 3);
  computeNormalImage<<<grid_size, block_size>>>(
      depth_image_cuda, height_image_cuda, normal_image_cuda, width, height,
      lidar_intrinsics);

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float msecTotal = 1.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  std::cout << "runtime: " << msecTotal << "ms" << std::endl;

  float* normal_image;
  normal_image = (float*)malloc(sizeof(float) * 2048 * 128 * 3);
  cudaMemcpy(normal_image, normal_image_cuda, sizeof(float) * 2048 * 128 * 3,
             cudaMemcpyDeviceToHost);

  // end: free the memory
  cudaFree(depth_image_cuda);
  cudaFree(height_image_cuda);
  cudaFree(normal_image_cuda);

  //////////////////////////////// converted into PCL
  pcl::PointCloud<pcl::PointXYZINormal> cloud;
  for (int u = 0; u < width; u++) {
    for (int v = 0; v < height; v++) {
      float height = access<float>(v, u, width, height_image.data());
      float depth = access<float>(v, u, width, depth_image.data());
      if (height <= 1e-4 || depth <= 1e-4) continue;
      float* normal = new float[3];
      normal[0] = normal_image[3 * (v * width + u)];
      normal[1] = normal_image[3 * (v * width + u) + 1];
      normal[2] = normal_image[3 * (v * width + u) + 2];
      float r = sqrt(depth * depth - height * height);
      float azimuth_angle_rad =
          M_PI - u * lidar_intrinsics.rads_per_pixel_azimuth;
      pcl::PointXYZINormal point;
      point.x = r * cos(azimuth_angle_rad);
      point.y = r * sin(azimuth_angle_rad);
      point.z = height;
      point.normal_x = normal[0];
      point.normal_y = normal[1];
      point.normal_z = normal[2];
      cloud.push_back(point);
    }
  }
  pcl::PCDWriter pcd_writer;
  pcd_writer.write(
      "/Spy/dataset/mapping_results/nvblox/20220216_garden_day/"
      "test_xyz_normal.pcd",
      cloud);

  return 0;
}