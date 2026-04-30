/*
Copyright 2026 NVIDIA CORPORATION

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

#include "nvblox/projective_texturing/texture_atlas.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

namespace {

// CUDA kernel to copy a camera image into a region of the atlas.
// Each thread copies one pixel.
__global__ void copyImageToAtlasKernel(const Color* src, int src_width,
                                       int src_height, Color* dst,
                                       int dst_width, int x_offset,
                                       int y_offset) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= src_width || row >= src_height) return;

  int src_idx = row * src_width + col;
  int dst_idx = (row + y_offset) * dst_width + (col + x_offset);
  dst[dst_idx] = src[src_idx];
}

}  // namespace

void TextureAtlas::buildAtlasAsync(const std::vector<CameraView>& views,
                                   const CudaStream& stream) {
  if (views.empty()) {
    LOG(WARNING) << "TextureAtlas::build: no views provided";
    num_cameras_ = 0;
    return;
  }

  timing::Timer timer("projective_texture_mapper/build_atlas");

  num_cameras_ = static_cast<int>(views.size());
  regions_.resize(num_cameras_);

  // Pack all camera images side-by-side horizontally into a single atlas.
  // For single camera, the atlas equals the image (offset=0, scale=1).
  // max_height is the height of the tallest image (handles different image
  // resolutions).
  int total_width = 0;
  int max_height = 0;
  for (const auto& view : views) {
    total_width += view.color_image.width();
    max_height = std::max(max_height, view.color_image.height());
  }

  atlas_ = ColorImage(max_height, total_width, MemoryType::kDevice);

  // Clear atlas to black (needed for multi-camera when images have
  // different heights -- ensures unused regions are black)
  cudaMemsetAsync(atlas_.dataPtr(), 0, atlas_.numel() * sizeof(Color), stream);

  // Copy each image into its region of the atlas
  int x_offset = 0;
  for (int i = 0; i < num_cameras_; ++i) {
    const auto& img = views[i].color_image;
    int w = img.width();
    int h = img.height();

    // Compute UV offset and scale for this camera region
    regions_[i].offset =
        Vector2f(static_cast<float>(x_offset) / total_width, 0.0f);
    regions_[i].scale = Vector2f(static_cast<float>(w) / total_width,
                                 static_cast<float>(h) / max_height);

    // Copy image data into atlas region
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    copyImageToAtlasKernel<<<grid, block, 0, stream>>>(
        img.dataConstPtr(), w, h, atlas_.dataPtr(), total_width, x_offset, 0);

    x_offset += w;
  }

  checkCudaErrors(cudaGetLastError());
}

Vector2f TextureAtlas::uvOffset(int camera_index) const {
  CHECK_GE(camera_index, 0);
  CHECK_LT(camera_index, num_cameras_);
  return regions_[camera_index].offset;
}

Vector2f TextureAtlas::uvScale(int camera_index) const {
  CHECK_GE(camera_index, 0);
  CHECK_LT(camera_index, num_cameras_);
  return regions_[camera_index].scale;
}

}  // namespace nvblox
