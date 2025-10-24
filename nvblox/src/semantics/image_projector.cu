#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include "nvblox/core/hash.h"
#include "nvblox/semantics/image_projector.h"
#include "nvblox/semantics/internal/cuda/impl/image_projector_impl.cuh"

namespace nvblox {

DepthImageBackProjector::DepthImageBackProjector()
    : DepthImageBackProjector(std::make_shared<CudaStreamOwning>()) {}

DepthImageBackProjector::DepthImageBackProjector(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

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
  thrust::transform(thrust::device.on(*cuda_stream_),
                    pointcloud_L.points().begin(), pointcloud_L.points().end(),
                    voxel_center_pointcloud_L->points().begin(),
                    GetVoxelCenter(voxel_size));

  // Sort points to bring duplicates together.
  thrust::sort(thrust::device.on(*cuda_stream_),
               voxel_center_pointcloud_L->points().begin(),
               voxel_center_pointcloud_L->points().end(),
               VectorCompare<Vector3f>());

  // Find unique points and erase redundancies. The iterator will point to
  // the new last index.
  auto iterator = thrust::unique(thrust::device.on(*cuda_stream_),
                                 voxel_center_pointcloud_L->points().begin(),
                                 voxel_center_pointcloud_L->points().end());

  // Figure out the new size.
  size_t new_size = iterator - voxel_center_pointcloud_L->points().begin();
  voxel_center_pointcloud_L->resize(new_size);
}

}  // namespace nvblox
