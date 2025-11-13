/*
Copyright 2023 NVIDIA CORPORATION

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
#include "nvblox/integrators/esdf_slicer.h"

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/utils/cuda_kernel_utils.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

__global__ void populateSliceFromLayerKernel(
    const Index3DDeviceHashMapType<EsdfBlock> block_hash,
    AxisAlignedBoundingBox aabb, float block_size, float* image, int rows,
    int cols, float slice_height, float resolution, float unobserved_value) {
  const float voxel_size = block_size / EsdfBlock::kVoxelsPerSide;
  const int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;

  if (pixel_col >= cols || pixel_row >= rows) {
    return;
  }

  // Figure out where this pixel should map to.
  // Get voxel centers by adding half a voxel size.
  Vector3f voxel_position(
      aabb.min().x() + voxel_size / 2.0f + resolution * pixel_col,
      aabb.min().y() + voxel_size / 2.0f + resolution * pixel_row,
      slice_height);

  Index3D block_index, voxel_index;

  getBlockAndVoxelIndexFromPositionInLayer(block_size, voxel_position,
                                           &block_index, &voxel_index);

  // Get the relevant block.
  EsdfBlock* block_ptr = nullptr;
  auto it = block_hash.find(block_index);
  if (it != block_hash.end()) {
    block_ptr = it->second;
  } else {
    image::access(pixel_row, pixel_col, cols, image) = unobserved_value;
    return;
  }

  // Get the relevant pixel.
  const EsdfVoxel* voxel =
      &block_ptr->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  float distance = unobserved_value;
  if (voxel->observed) {
    distance = voxel_size * std::sqrt(voxel->squared_distance_vox);
    if (voxel->is_inside) {
      distance = -distance;
    }
  }
  image::access(pixel_row, pixel_col, cols, image) = distance;
}

constexpr int8_t kOccupancyGridUnknownValue = -1;

// Calling rules:
// - Should be called with 2D grid of thread-blocks where the total number of
// threads in each dimension exceeds the number of pixels in each image
// dimension ie:
// - blockIdx.x * blockDim.x + threadIdx.x > cols
// - blockIdx.y * blockDim.y + threadIdx.y > rows
// - We assume that theres enough space in the pointcloud to store a point per
// pixel if required.
__global__ void occupancyGridFromSliceImageKernel(
    const float* slice_image_ptr,         // NOLINT
    const int rows,                       // NOLINT
    const int cols,                       // NOLINT
    const float unobserved_value,         // NOLINT
    int8_t* occupancy_grid_device_ptr) {  // NOLINT
  // Get the pixel addressed by this thread.
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx >= cols || row_idx >= rows) {
    return;
  }

  // Access the slice
  const float pixel_value =
      image::access(row_idx, col_idx, cols, slice_image_ptr);

  constexpr float kEps = 1e-2;
  constexpr int8_t kOccupiedValue = 100;

  // If the distance is under epsilon (near zero), we're in occupied space so
  // set it to kOccupiedValue (via implicit cast from Bool(True) to int(1)).
  int8_t value = (pixel_value < kEps) * kOccupiedValue;
  // If the value is approximately equal to the constant signifying unknown,
  // set it to unknown in ros (-1).
  if (fabsf(pixel_value - unobserved_value) < kEps) {
    value = kOccupancyGridUnknownValue;
  }

  // Write the point to the output
  image::access(row_idx, col_idx, cols, occupancy_grid_device_ptr) = value;
}

EsdfSlicer::EsdfSlicer() : EsdfSlicer(std::make_shared<CudaStreamOwning>()) {}

EsdfSlicer::EsdfSlicer(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

AxisAlignedBoundingBox EsdfSlicer::getAabbOfLayerAtHeight(
    const EsdfLayer& layer, const float slice_height) {
  // Get the bounding box of the layer at this height
  // To do this, we get all the block indices, figure out which intersect
  // with our desired height, and select the min and max in the X and Y
  // direction.
  timing::Timer get_aabb_timer("/esdf_slicer/get_aabb");

  // Figure out the index of the desired height.
  Index3D desired_z_block_index;
  Index3D desired_z_voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(
      layer.block_size(), Vector3f(0.0f, 0.0f, slice_height),
      &desired_z_block_index, &desired_z_voxel_index);

  // Get a bounding box for the whole layer
  AxisAlignedBoundingBox aabb;
  aabb.setEmpty();
  for (const Index3D& block_index : layer.getAllBlockIndices()) {
    // Skip all other heights of block.
    if (block_index.z() != desired_z_block_index.z()) {
      continue;
    }
    // Extend the AABB by the dimensions of this block.
    aabb.extend(getAABBOfBlock(layer.block_size(), block_index));
  }
  return aabb;
}

AxisAlignedBoundingBox EsdfSlicer::getCombinedAabbOfLayersAtHeight(
    const EsdfLayer& layer_1, const EsdfLayer& layer_2,
    float layer_1_slice_height, float layer_2_slice_height) {
  // Combined (enclosing) AABB
  const AxisAlignedBoundingBox aabb_1 =
      getAabbOfLayerAtHeight(layer_1, layer_1_slice_height);
  const AxisAlignedBoundingBox aabb_2 =
      getAabbOfLayerAtHeight(layer_2, layer_2_slice_height);
  return aabb_1.merged(aabb_2);
}

void EsdfSlicer::sliceLayerToDistanceImage(const EsdfLayer& layer,
                                           float slice_height,
                                           float unobserved_value,
                                           AxisAlignedBoundingBox* aabb,
                                           Image<float>* output_image) {
  CHECK_NOTNULL(aabb);
  *aabb = getAabbOfLayerAtHeight(layer, slice_height);
  sliceLayerToDistanceImage(layer, slice_height, unobserved_value, *aabb,
                            output_image);
}

void EsdfSlicer::sliceLayerToDistanceImage(const EsdfLayer& layer,
                                           float slice_height,
                                           float unobserved_value,
                                           const AxisAlignedBoundingBox& aabb,
                                           Image<float>* output_image) {
  if (aabb.isEmpty()) {
    return;
  }
  timing::Timer slice_layer_timer("/esdf_slicer/slice_layer");

  const float block_size = layer.block_size();
  constexpr int kVoxelsPerSide = VoxelBlock<EsdfVoxel>::kVoxelsPerSide;
  const float voxel_size = block_size / kVoxelsPerSide;

  Vector3f bounding_size = aabb.sizes();
  // Width = cols, height = rows
  int width = static_cast<int>(std::ceil(bounding_size.x() / voxel_size));
  int height = static_cast<int>(std::ceil(bounding_size.y() / voxel_size));

  // Create an image on the device to fit the aabb.
  Image<float> image(height, width, MemoryType::kDevice);

  // Fill in the float image.
  populateSliceFromLayer(layer, aabb, slice_height, unobserved_value,
                         voxel_size, &image);

  *output_image = std::move(image);
  checkCudaErrors(cudaPeekAtLastError());
}

void EsdfSlicer::sliceLayersToCombinedDistanceImage(
    const EsdfLayer& layer_1, const EsdfLayer& layer_2,
    float layer_1_slice_height, float layer_2_slice_height,
    float unobserved_value, AxisAlignedBoundingBox* aabb,
    Image<float>* output_image) {
  CHECK_NOTNULL(aabb);
  CHECK_NOTNULL(output_image);
  *aabb = getCombinedAabbOfLayersAtHeight(
      layer_1, layer_2, layer_1_slice_height, layer_2_slice_height);
  sliceLayersToCombinedDistanceImage(layer_1, layer_2, layer_1_slice_height,
                                     layer_2_slice_height, unobserved_value,
                                     *aabb, output_image);
}

void EsdfSlicer::sliceLayersToCombinedDistanceImage(
    const EsdfLayer& layer_1, const EsdfLayer& layer_2,
    float layer_1_slice_height, float layer_2_slice_height,
    float unobserved_value, const AxisAlignedBoundingBox& aabb,
    Image<float>* output_image) {
  CHECK_NOTNULL(output_image);
  CHECK_EQ(layer_1.voxel_size(), layer_2.voxel_size());

  Image<float> slice_image_1(MemoryType::kDevice);
  sliceLayerToDistanceImage(layer_1, layer_1_slice_height, unobserved_value,
                            aabb, &slice_image_1);
  sliceLayerToDistanceImage(layer_2, layer_2_slice_height, unobserved_value,
                            aabb, output_image);

  // Get the minimal distance between the two slices
  image::elementWiseMinInPlaceGPUAsync(slice_image_1, output_image,
                                       *cuda_stream_);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

void EsdfSlicer::occupancyGridFromSliceImage(const Image<float>& slice_image,
                                             signed char* occupancy_grid_data,
                                             const float unobserved_value) {
  CHECK_NOTNULL(occupancy_grid_data);

  // Can happen that this function is called before the ESDF contains data.
  // Return without doing anything if that happens.
  if (slice_image.numel() <= 0 || slice_image.rows() <= 0 ||
      slice_image.cols() <= 0) {
    return;
  }

  const int width = slice_image.cols();
  const int height = slice_image.rows();

  // Allocate device-side scratch pad
  occupancy_grid_device_.reserveAsync(width * height, *cuda_stream_);

  // Call CUDA kernel to convert from float distance to int8 occupancy.
  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(
      slice_image.cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
      slice_image.rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
      1);
  occupancyGridFromSliceImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                                      *cuda_stream_>>>(
      slice_image.dataConstPtr(),    // NOLINT
      slice_image.rows(),            // NOLINT
      slice_image.cols(),            // NOLINT
      unobserved_value,              // NOLINT
      occupancy_grid_device_.data()  // NOLINT
  );
  checkCudaErrors(cudaPeekAtLastError());

  // Copy into the message
  checkCudaErrors(cudaMemcpyAsync(
      occupancy_grid_data, occupancy_grid_device_.data(),
      width * height * sizeof(int8_t), cudaMemcpyDefault, *cuda_stream_));
  cuda_stream_->synchronize();
}

void EsdfSlicer::populateSliceFromLayer(const EsdfLayer& layer,
                                        const AxisAlignedBoundingBox& aabb,
                                        float slice_height,
                                        float unobserved_value,
                                        float resolution,
                                        Image<float>* output_image) {
  CHECK(output_image->memory_type() == MemoryType::kDevice ||
        output_image->memory_type() == MemoryType::kUnified)
      << "Output needs to be accessible on device";
  // NOTE(alexmillane): At the moment we assume that the image is pre-allocated
  // to be the right size.
  CHECK_EQ(output_image->rows(),
           static_cast<int>(std::ceil(aabb.sizes().y() / resolution)));
  CHECK_EQ(output_image->cols(),
           static_cast<int>(std::ceil(aabb.sizes().x() / resolution)));
  if (output_image->numel() <= 0) {
    return;
  }
  const float voxel_size = layer.voxel_size();

  // Create a GPU hash of the ESDF.
  GPULayerView<EsdfBlock>& gpu_layer_view =
      layer.getGpuLayerView(*cuda_stream_);

  // Pass in the GPU hash and AABB and let the kernel figure it out.
  constexpr int kThreadDim = 16;
  const int rounded_rows = divideRoundUp(output_image->rows(), kThreadDim);
  const int rounded_cols = divideRoundUp(output_image->cols(), kThreadDim);
  dim3 block_dim(rounded_cols, rounded_rows);
  dim3 thread_dim(kThreadDim, kThreadDim);

  populateSliceFromLayerKernel<<<block_dim, thread_dim, 0, *cuda_stream_>>>(
      gpu_layer_view.getHash().impl_,  // NOLINT
      aabb,                            // NOLINT
      layer.block_size(),              // NOLINT
      output_image->dataPtr(),         // NOLINT
      output_image->rows(),            // NOLINT
      output_image->cols(),            // NOLINT
      slice_height,                    // NOLINT
      resolution,                      // NOLINT
      unobserved_value                 // NOLINT
  );
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox
