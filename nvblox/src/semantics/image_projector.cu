#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include "nvblox/core/hash.h"
#include "nvblox/semantics/image_projector.h"

namespace nvblox {

DepthImageBackProjector::DepthImageBackProjector()
    : DepthImageBackProjector(std::make_shared<CudaStreamOwning>()) {}

DepthImageBackProjector::DepthImageBackProjector(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

__global__ void projectImageKernel(const Camera camera, const float* image,
                                   const int rows, const int cols,
                                   const float max_back_projection_distance_m,
                                   Vector3f* pointcloud, int* pointcloud_size) {
  // Each thread does a single pixel
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  float depth = image::access(row_idx, col_idx, cols, image);
  if (depth <= 0.0f || depth > max_back_projection_distance_m) {
    return;
  }

  Index2D u_C(col_idx, row_idx);

  // Unproject from the image.
  const Vector3f p_C = camera.unprojectFromPixelIndices(u_C, depth);

  // Insert into the pointcloud.
  pointcloud[atomicAdd(pointcloud_size, 1)] = p_C;
}

void DepthImageBackProjector::backProjectOnGPU(
    const DepthImage& image, const Camera& camera, Pointcloud* pointcloud_C_ptr,
    const float max_back_projection_distance_m) {
  CHECK_NOTNULL(pointcloud_C_ptr);
  CHECK(pointcloud_C_ptr->memory_type() == MemoryType::kDevice ||
        pointcloud_C_ptr->memory_type() == MemoryType::kUnified);

  // Create the max number of output points.
  pointcloud_C_ptr->resizeAsync(image.numel(), *cuda_stream_);

  // Reset the counter.
  if (pointcloud_size_device_ == nullptr || pointcloud_size_host_ == nullptr) {
    pointcloud_size_device_ = make_unified<int>(MemoryType::kDevice);
    pointcloud_size_host_ = make_unified<int>(MemoryType::kHost);
  }
  pointcloud_size_device_.setZero();

  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(image.cols() / kThreadsPerThreadBlock.x + 1,
                        image.rows() / kThreadsPerThreadBlock.y + 1, 1);
  projectImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0, *cuda_stream_>>>(
      camera, image.dataConstPtr(), image.rows(), image.cols(),
      max_back_projection_distance_m, pointcloud_C_ptr->dataPtr(),
      pointcloud_size_device_.get());
  checkCudaErrors(cudaPeekAtLastError());

  pointcloud_size_device_.copyToAsync(pointcloud_size_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  pointcloud_C_ptr->resize(*pointcloud_size_host_);
}

struct GetVoxelCenter {
  const float voxel_size;

  GetVoxelCenter(float _voxel_size) : voxel_size(_voxel_size) {}

  __host__ __device__ Vector3f operator()(const Vector3f& x) const {
    return (x / voxel_size).array().floor() * voxel_size + voxel_size / 2.0f;
  }
};

void DepthImageBackProjector::pointcloudToVoxelCentersOnGPU(
    const Pointcloud& pointcloud_L, float voxel_size,
    Pointcloud* voxel_center_pointcloud_L) {
  CHECK_NOTNULL(voxel_center_pointcloud_L);
  CHECK(voxel_center_pointcloud_L->memory_type() == MemoryType::kDevice ||
        voxel_center_pointcloud_L->memory_type() == MemoryType::kUnified);

  if (pointcloud_L.empty()) {
    return;
  }

  // Create an array of voxel centers matching the nearest voxel for each point.
  voxel_center_pointcloud_L->resize(pointcloud_L.size());
  thrust::transform(thrust::device, pointcloud_L.points().begin(),
                    pointcloud_L.points().end(),
                    voxel_center_pointcloud_L->points().begin(),
                    GetVoxelCenter(voxel_size));

  // Sort points to bring duplicates together.
  thrust::sort(thrust::device, voxel_center_pointcloud_L->points().begin(),
               voxel_center_pointcloud_L->points().end(),
               VectorCompare<Vector3f>());

  // Find unique points and erase redundancies. The iterator will point to
  // the new last index.
  auto iterator = thrust::unique(thrust::device,
                                 voxel_center_pointcloud_L->points().begin(),
                                 voxel_center_pointcloud_L->points().end());

  // Figure out the new size.
  size_t new_size = iterator - voxel_center_pointcloud_L->points().begin();
  voxel_center_pointcloud_L->resize(new_size);
}

}  // namespace nvblox
