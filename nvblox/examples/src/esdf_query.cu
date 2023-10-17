/*
Copyright 2022 NVIDIA CORPORATION

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
#include <gflags/gflags.h>
#include <iostream>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

class EsdfQueryExample {
 public:
  void createMap();

  void queryMap(size_t num_queries);

 private:
  // Voxel size to use.
  float voxel_size_ = 0.10;

  // Mapping class which contains all the relevant layers and integrators.
  std::unique_ptr<Mapper> mapper_;

  // A simulation scene, used in place of real data.
  primitives::Scene scene_;
};

void EsdfQueryExample::createMap() {
  mapper_.reset(new Mapper(voxel_size_, MemoryType::kDevice));

  // Create a map that's a box with a sphere in the middle.
  scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-5.5f, -5.5f, -0.5f),
                                         Vector3f(5.5f, 5.5f, 5.5f));
  scene_.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
  scene_.addGroundLevel(0.0f);
  scene_.addCeiling(5.0f);
  scene_.addPrimitive(std::make_unique<primitives::Cube>(
      Vector3f(0.0f, 0.0f, 2.0f), Vector3f(2.0f, 2.0f, 2.0f)));
  scene_.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));

  // Get the ground truth TSDF from this scene (otherwise integrate individual
  // frames).
  // We need to create a temp layer unfortunately.
  TsdfLayer gt_tsdf(voxel_size_, MemoryType::kHost);
  scene_.generateLayerFromScene(4 * voxel_size_, &gt_tsdf);

  mapper_->tsdf_layer() = std::move(gt_tsdf);

  // Set the max computed distance to 5 meters.
  mapper_->esdf_integrator().max_esdf_distance_m(5.0f);
  // Generate the ESDF from everything in the TSDF.
  mapper_->updateFullEsdf();

  // Output a ply pointcloud of all of the distances to compare the query
  // points to.
  io::outputVoxelLayerToPly(mapper_->esdf_layer(), "esdf_query_map.ply");
}

__global__ void queryDistancesKernel(
    size_t num_queries, Index3DDeviceHashMapType<EsdfBlock> block_hash,
    float block_size, Vector3f* query_locations, float* query_distances,
    Vector3f* query_nearest_points) {
  constexpr int kNumVoxelsPerBlock = 8;
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  // Figure out which point this thread should be querying.
  size_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= num_queries) {
    return;
  }
  Vector3f query_location = query_locations[query_index];

  // Get the correct block from the hash.
  EsdfVoxel* esdf_voxel;
  if (!getVoxelAtPosition<EsdfVoxel>(block_hash, query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
    query_distances[query_index] = -100.0f;
    query_nearest_points[query_index] = query_location;
  } else {
    // Get the distance of the relevant voxel.
    query_distances[query_index] =
        voxel_size * sqrt(esdf_voxel->squared_distance_vox);
    // If it's inside an obstacle then set the distance to negative.
    if (esdf_voxel->is_inside) {
      query_distances[query_index] = -query_distances[query_index];
    }
    // TODO(helen): quick hack to get ~approx location of parent, should be
    // within a voxel of where it actually should be.
    // The parent location is relative to the location of the current voxel.
    query_nearest_points[query_index] =
        query_location +
        voxel_size * esdf_voxel->parent_direction.cast<float>();
  }
}

void EsdfQueryExample::queryMap(size_t num_queries) {
  // First, create an vector of random query points within the AABB.
  std::vector<Vector3f> query_points(num_queries);

  for (size_t i = 0; i < num_queries; i++) {
    query_points[i] = scene_.aabb().sample();
  }

  // Move this vector to the GPU.
  device_vector<Vector3f> query_points_device;
  query_points_device.copyFrom(query_points);
  // Create the output vectors which live in device memory.
  device_vector<float> distances_device(num_queries);
  device_vector<Vector3f> nearest_point_device(num_queries);

  // GPU hash transfer timer
  timing::Timer hash_transfer_timer("query/hash_transfer");
  GPULayerView<EsdfBlock> gpu_layer_view =
      mapper_->esdf_layer().getGpuLayerView();
  hash_transfer_timer.Stop();

  // Call a kernel.
  timing::Timer kernel_timer("query/kernel");

  constexpr int kNumThreads = 512;
  int num_blocks = num_queries / kNumThreads + 1;

  // Call the kernel.
  queryDistancesKernel<<<num_blocks, kNumThreads>>>(
      num_queries, gpu_layer_view.getHash().impl_,
      mapper_->esdf_layer().block_size(), query_points_device.data(),
      distances_device.data(), nearest_point_device.data());
  checkCudaErrors(cudaDeviceSynchronize());
  kernel_timer.Stop();

  // Get the data back out. You can call toVector() even on device vectors
  // to get them back out on the host.
  // This just outputs a pointcloud of all the query points & distances.
  io::PlyWriter writer("esdf_query_results.ply");
  std::vector<float> distances_host = distances_device.toVector();
  writer.setPoints(&query_points);
  writer.setIntensities(&distances_host);
  writer.write();
}

}  // namespace nvblox

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  size_t num_queries = 1e6;
  nvblox::EsdfQueryExample example;
  LOG(INFO) << "Creating the map from primitives.";
  example.createMap();
  LOG(INFO) << "Running queries.";
  example.queryMap(num_queries);
  LOG(INFO) << "Time taken to run " << num_queries << " queries: ";

  std::cout << nvblox::timing::Timing::Print();
}
