#include "nvblox/core/internal/error_check.h"
#include "nvblox/sensors/pointcloud.h"
#include "nvblox/utils/logging.h"

namespace nvblox {

Pointcloud::Pointcloud(int num_points, MemoryType memory_type)
    : points_(unified_vector<Vector3f>(num_points, memory_type)) {}

Pointcloud::Pointcloud(MemoryType memory_type)
    // zero-sized allocation in order to store memory_type in the underlying
    // vector
    : points_(unified_vector<Vector3f>(0, memory_type)) {}

void Pointcloud::copyFrom(const Pointcloud& other) {
  points_.copyFrom(other.points_);
}

void Pointcloud::copyFromAsync(const Pointcloud& other,
                               const CudaStream cuda_stream) {
  points_.copyFromAsync(other.points_, cuda_stream);
}

void Pointcloud::copyFrom(const std::vector<Vector3f>& points) {
  points_.copyFrom(points);
}

void Pointcloud::copyFromAsync(const std::vector<Vector3f>& points,
                               const CudaStream cuda_stream) {
  points_.copyFromAsync(points, cuda_stream);
}

void Pointcloud::copyFrom(const unified_vector<Vector3f>& points) {
  points_.copyFrom(points);
}

void Pointcloud::copyFromAsync(const unified_vector<Vector3f>& points,
                               const CudaStream cuda_stream) {
  points_.copyFromAsync(points, cuda_stream);
}

// Pointcloud operations

__global__ void transformPointcloudKernel(const Transform T_out_in,
                                          int pointcloud_size,
                                          const Vector3f* pointcloud_in,
                                          Vector3f* pointcloud_out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= pointcloud_size) {
    return;
  }

  pointcloud_out[index] = T_out_in * pointcloud_in[index];
}

void transformPointcloudOnGPU(const Transform& T_out_in,
                              const Pointcloud& pointcloud_in,
                              Pointcloud* pointcloud_out_ptr) {
  // Calls the streamed version after creating a stream
  CudaStreamOwning cuda_stream;
  transformPointcloudOnGPU(T_out_in, pointcloud_in, pointcloud_out_ptr,
                           &cuda_stream);
}

void transformPointcloudOnGPU(const Transform& T_out_in,
                              const Pointcloud& pointcloud_in,
                              Pointcloud* pointcloud_out_ptr,
                              CudaStream* cuda_stream_ptr) {
  CHECK_NOTNULL(pointcloud_out_ptr);
  CHECK_NOTNULL(cuda_stream_ptr);
  CHECK(pointcloud_out_ptr->memory_type() == MemoryType::kDevice ||
        pointcloud_out_ptr->memory_type() == MemoryType::kUnified);

  if (pointcloud_in.empty()) {
    return;
  }
  pointcloud_out_ptr->resizeAsync(pointcloud_in.size(), *cuda_stream_ptr);

  constexpr int kThreadsPerThreadBlock = 512;
  const int num_blocks(pointcloud_in.size() / kThreadsPerThreadBlock + 1);
  transformPointcloudKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                              *cuda_stream_ptr>>>(
      T_out_in, pointcloud_in.size(), pointcloud_in.dataConstPtr(),
      pointcloud_out_ptr->dataPtr());
  cuda_stream_ptr->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox
