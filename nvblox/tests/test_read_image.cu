#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../executables/include/nvblox/datasets/external/stb_image.h"

#include <iostream>
#include <vector>

const float FACTOR = 1000.0f;
const float OFFSET = 0.0f;

std::vector<float> load16BitDepthImage(const std::string& filename) {
  int width, height, num_channels;
  uint16_t* image_data =
      stbi_load_16(filename.c_str(), &width, &height, &num_channels, 0);

  std::vector<float> float_image_data(height * width);
  for (int lin_idx = 0; lin_idx < float_image_data.size(); lin_idx++) {
    float_image_data[lin_idx] =
        static_cast<float>(image_data[lin_idx]) / FACTOR - OFFSET;
  }
  stbi_image_free(image_data);
  return float_image_data;
}

__global__ void test(float* depth_image_cuda) {
  int cnt = 0;
  for (int i = 0; i < 2048 * 128; i++) {
    if (depth_image_cuda[i] != depth_image_cuda[i]) cnt++;
    // printf("%f ", depth_image_cuda[i]);
  }
  // printf("cnt: %d\n", cnt);
}

int main(int argc, char** argv) {
  std::vector<float> depth_image = load16BitDepthImage(std::string(
      "/Spy/dataset/nvblox/20220216_garden_day/seq-01/frame-000000.depth.png"));
  // std::vector<float> z_image = load16BitDepthImage(std::string(
  //     "/Spy/dataset/nvblox/20220216_garden_day/seq-01/frame-000000.z.png"));
  float* depth_image_cuda;
  cudaMalloc((void**)&depth_image_cuda, sizeof(float) * 2048 * 128);
  cudaMemcpy(depth_image_cuda, depth_image.data(), sizeof(float) * 2048 * 128,
             cudaMemcpyHostToDevice);

  test<<<1, 1>>>(depth_image_cuda);
  // std::cout << cnt << std::endl;
  cudaFree(depth_image_cuda);
  return 0;
}