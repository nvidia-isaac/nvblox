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
#include <fstream>
#include <iostream>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

class SphereBenchmark {
 public:
  SphereBenchmark();

  void runBenchmark();
  bool outputMesh(const std::string& ply_output_path);

 private:
  // Settings. Do not modify or the benchmark isn't comparable.
  static constexpr float kVoxelSize = 0.05;
  static constexpr float kBlockSize =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSize;
  static constexpr int kNumTrajectoryPoints = 80;
  static constexpr float kSphereRadius = 2.0f;
  static constexpr float kTrajectoryRadius = 4.0f;
  static constexpr float kMaxEnvironmentDimension = 5.0f;

  // Actual layers.
  TsdfLayer tsdf_layer_;
  EsdfLayer esdf_layer_;
  MeshLayer mesh_layer_;

  // Simulated camera.
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

SphereBenchmark::SphereBenchmark()
    : tsdf_layer_(kVoxelSize, MemoryType::kDevice),
      esdf_layer_(kVoxelSize, MemoryType::kUnified),
      mesh_layer_(kBlockSize, MemoryType::kUnified),
      camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

// C++ <17 requires declaring static constexpr variables
// In C++17 this is no longer required, as static constexpr also implies inline
constexpr float SphereBenchmark::kVoxelSize;
constexpr float SphereBenchmark::kBlockSize;
constexpr int SphereBenchmark::kNumTrajectoryPoints;
constexpr float SphereBenchmark::kSphereRadius;
constexpr float SphereBenchmark::kTrajectoryRadius;
constexpr float SphereBenchmark::kMaxEnvironmentDimension;

void SphereBenchmark::runBenchmark() {
  // Create an integrator with default settings.
  ProjectiveTsdfIntegrator integrator;
  MeshIntegrator mesh_integrator;
  EsdfIntegrator esdf_integrator;
  esdf_integrator.max_esdf_distance_m(4.0f);

  // Scene is bounded to the dimensions above.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(
      Vector3f(-kMaxEnvironmentDimension, -kMaxEnvironmentDimension, 0.0f),
      Vector3f(kMaxEnvironmentDimension, kMaxEnvironmentDimension,
               kMaxEnvironmentDimension));
  // Create a scene with a ground plane and a sphere.
  scene.addGroundLevel(0.0f);
  scene.addCeiling(kMaxEnvironmentDimension);
  scene.addPrimitive(std::make_unique<primitives::Sphere>(
      Vector3f(0.0f, 0.0f, kSphereRadius), kSphereRadius));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-kMaxEnvironmentDimension, kMaxEnvironmentDimension,
                           -kMaxEnvironmentDimension, kMaxEnvironmentDimension);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kSphereRadius);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene.generateDepthImageFromScene(
        camera_, T_S_C, 2 * kMaxEnvironmentDimension, &depth_frame);

    std::vector<Index3D> updated_blocks;
    // Integrate this depth image.
    {
      timing::Timer integration_timer("benchmark/integrate_tsdf");
      integrator.integrateFrame(depth_frame, T_S_C, camera_, &tsdf_layer_,
                                &updated_blocks);
    }

    // Integrate the mesh.
    {
      timing::Timer mesh_timer("benchmark/integrate_mesh");
      mesh_integrator.integrateBlocksGPU(tsdf_layer_, updated_blocks,
                                         &mesh_layer_);
    }

    // Integrate the ESDF.
    {
      timing::Timer esdf_timer("benchmark/integrate_esdf");
      esdf_integrator.integrateBlocks(tsdf_layer_, updated_blocks,
                                      &esdf_layer_);
    }
  }
}

bool SphereBenchmark::outputMesh(const std::string& ply_output_path) {
  timing::Timer timer_write("mesh/write");
  return io::outputMeshLayerToPly(mesh_layer_, ply_output_path);
}

}  // namespace nvblox

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  nvblox::warmupCuda();

  std::string output_mesh_path = "";
  if (argc >= 2) {
    output_mesh_path = argv[1];
  }

  nvblox::SphereBenchmark benchmark;
  benchmark.runBenchmark();

  if (!output_mesh_path.empty()) {
    benchmark.outputMesh(output_mesh_path);
  }

  std::cout << nvblox::timing::Timing::Print();

  return 0;
}
