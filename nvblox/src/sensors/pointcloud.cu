#include "nvblox/sensors/pointcloud.h"

#include "nvblox/utils/logging.h"

#include "nvblox/core/internal/error_check.h"

namespace nvblox {

Pointcloud::Pointcloud(int num_points, MemoryType memory_type)
    : points_(unified_vector<Vector3f>(num_points, memory_type)) {}

Pointcloud::Pointcloud(MemoryType memory_type) {}

Pointcloud::Pointcloud(const Pointcloud& other)
    : points_(other.points_, other.memory_type()) {
  LOG(WARNING) << "Deep copy of Pointcloud.";
}

Pointcloud::Pointcloud(const Pointcloud& other, MemoryType memory_type)
    : points_(other.points_, memory_type) {}

Pointcloud& Pointcloud::operator=(const Pointcloud& other) {
  LOG(WARNING) << "Deep copy of Pointcloud.";
  points_ = unified_vector<Vector3f>(other.points_, other.memory_type());
  return *this;
}

Pointcloud::Pointcloud(const std::vector<Vector3f>& points,
                       MemoryType memory_type)
    : points_(points, memory_type) {}

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
  CHECK_NOTNULL(pointcloud_out_ptr);
  CHECK(pointcloud_out_ptr->memory_type() == MemoryType::kDevice ||
        pointcloud_out_ptr->memory_type() == MemoryType::kUnified);

  if (pointcloud_in.empty()) {
    return;
  }
  pointcloud_out_ptr->resize(pointcloud_in.size());

  cudaStream_t cuda_stream;
  checkCudaErrors(cudaStreamCreate(&cuda_stream));
  constexpr int kThreadsPerThreadBlock = 512;
  const int num_blocks(pointcloud_in.size() / kThreadsPerThreadBlock + 1);
  transformPointcloudKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                              cuda_stream>>>(T_out_in, pointcloud_in.size(),
                                             pointcloud_in.dataConstPtr(),
                                             pointcloud_out_ptr->dataPtr());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaStreamDestroy(cuda_stream));
}

}  // namespace nvblox
