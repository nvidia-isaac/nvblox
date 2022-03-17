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
#include <gtest/gtest.h>
#include <cmath>

#include "nvblox/core/camera.h"
#include "nvblox/core/image.h"
#include "nvblox/core/interpolation_3d.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/csv.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/projective_tsdf_integrator_cpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

constexpr float kInterpolatedSurfaceDistanceEpsilon = 1e-6;

// Parameters which define a passing test which measures the difference between
// CPU and GPU integration. We accept a test if
// <(kAcceptablePercentageOverThreshold)% voxels have less than
// (kGPUvsCPUDifferenceThresholdM)meters difference.
// NOTE(alexmillane): Obviously there are some slight differences in the voxel
// distance values in a small number of voxels.
constexpr float kGPUvsCPUDifferenceThresholdM = 1e-3;      // 1mm
constexpr float kAcceptablePercentageOverThreshold = 0.4;  // 0.4%

class TsdfIntegratorTest : public ::testing::Test {
 protected:
  TsdfIntegratorTest()
      : layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.2;
  TsdfLayer layer_;

  // How much error we expect on the surface
  constexpr static float surface_reconstruction_allowable_distance_error_vox_ =
      2.0f;
  constexpr static float surface_reconstruction_allowable_distance_error_m_ =
      surface_reconstruction_allowable_distance_error_vox_ * voxel_size_m_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

class TsdfIntegratorTestParameterized
    : public TsdfIntegratorTest,
      public ::testing::WithParamInterface<DeviceType> {
  // Add a param
};

// ProjectiveTsdfIntegrator child that gives the tests access to the internal
// functions.
class TestProjectiveTsdfIntegrator : public ProjectiveTsdfIntegrator {
 public:
  TestProjectiveTsdfIntegrator() : ProjectiveTsdfIntegrator() {}
  FRIEND_TEST(TsdfIntegratorTest, BlocksInView);
};

struct Plane {
  Plane(const Vector3f& _p, const Vector3f& _n) : p(_p), n(_n){};

  static Plane RandomAtPoint(const Vector3f& p) {
    return Plane(p, Vector3f::Random().normalized());
  }

  const Vector3f p;
  const Vector3f n;
};

DepthImage matrixToDepthImage(const Eigen::MatrixXf& mat) {
  DepthImage depth_frame(mat.rows(), mat.cols(), MemoryType::kUnified);
  for (int col_idx = 0; col_idx < mat.cols(); col_idx++) {
    for (int row_idx = 0; row_idx < mat.rows(); row_idx++) {
      depth_frame(row_idx, col_idx) = mat(row_idx, col_idx);
    }
  }
  return depth_frame;
}

Eigen::MatrixX3f backProjectToPlaneVectorized(
    const Eigen::MatrixX2f& uv_coordinates, const Plane& plane,
    const Camera& camera) {
  CHECK((uv_coordinates.col(0).array() >= 0.0f).all() &&
        (uv_coordinates.col(0).array() < camera.width()).all());
  CHECK((uv_coordinates.col(1).array() >= 0.0f).all() &&
        (uv_coordinates.col(1).array() < camera.height()).all());
  // Plane-ray intersection
  Eigen::ArrayX3f rays_matrix(uv_coordinates.rows(), 3);
  rays_matrix.col(0) =
      (uv_coordinates.col(0).array() - camera.cu()) / camera.fu();
  rays_matrix.col(1) =
      (uv_coordinates.col(1).array() - camera.cv()) / camera.fv();
  rays_matrix.col(2) = 1.0f;
  const Eigen::ArrayXf t_matrix =
      plane.p.dot(plane.n) *
      (rays_matrix.col(0) * plane.n.x() + rays_matrix.col(1) * plane.n.y() +
       rays_matrix.col(2) * plane.n.z())
          .inverse();
  // Each pixel's 3D point
  return rays_matrix.colwise() * t_matrix;
}

DepthImage getDepthImage(const Plane& plane, const Camera& camera) {
  CHECK(plane.p.z() > 0.0f);
  // Enumerate all pixel locations.
  Eigen::MatrixX2f uv_coordinates(camera.height() * camera.width(), 2);
  int linear_idx = 0;
  for (int u = 0; u < camera.width(); u++) {
    for (int v = 0; v < camera.height(); v++) {
      uv_coordinates(linear_idx, 0) = u;
      uv_coordinates(linear_idx, 1) = v;
      ++linear_idx;
    }
  }
  // Back project and get depth frame
  const Eigen::MatrixX3f points_C =
      backProjectToPlaneVectorized(uv_coordinates, plane, camera);
  Eigen::MatrixXf depths = (points_C.col(2).array());
  depths.resize(camera.height(), camera.width());
  return matrixToDepthImage(depths);
}

Eigen::MatrixX2f getRandomPixelLocations(const int num_samples,
                                         const Camera& camera) {
  // Note: Eigen's Random() generates numbers between -1.0 and 1.0 -> hence the
  // abs().
  Eigen::MatrixX2f uv_coordinates =
      Eigen::MatrixX2f::Random(num_samples, 2).array().abs();
  constexpr int border_px = 20;
  uv_coordinates.col(0) =
      (uv_coordinates.col(0) *
       static_cast<float>(camera.width() - 1 - 2 * border_px))
          .array() +
      border_px;
  uv_coordinates.col(1) =
      (uv_coordinates.col(1) *
       static_cast<float>(camera.height() - 1 - 2 * border_px))
          .array() +
      border_px;
  return uv_coordinates;
}

TEST_P(TsdfIntegratorTestParameterized, ReconstructPlane) {
  // Make sure this is deterministic.
  std::srand(0);

  // Get the params
  const DeviceType device_type = GetParam();

  // Plane centered at (0,0,depth) with random (slight) slant
  const Plane plane =
      Plane(Vector3f(0.0f, 0.0f, 5.0f),
            Vector3f(test_utils::randomFloatInRange(-0.25, 0.25),
                     test_utils::randomFloatInRange(-0.25, 0.25), -1.0f));

  // Get a depth map of our view of the plane.
  const DepthImage depth_frame = getDepthImage(plane, camera_);

  std::string filepath = "./depth_frame.csv";
  io::writeToCsv(filepath, depth_frame);

  // Integrate into a layer
  std::unique_ptr<ProjectiveTsdfIntegrator> integrator_ptr;
  if (device_type == DeviceType::kCPU) {
    integrator_ptr = std::make_unique<ProjectiveTsdfIntegratorCPU>();
  } else {
    integrator_ptr = std::make_unique<ProjectiveTsdfIntegrator>();
  }
  const Transform T_L_C = Transform::Identity();

  integrator_ptr->truncation_distance_vox(10.0f);
  integrator_ptr->integrateFrame(depth_frame, T_L_C, camera_, &layer_);

  // Sample some points on the plane, within the camera view.
  constexpr int kNumberOfPointsToCheck = 1000;
  const Eigen::MatrixX2f u_random_C =
      getRandomPixelLocations(kNumberOfPointsToCheck, camera_);
  const Eigen::MatrixX3f p_check_L =
      backProjectToPlaneVectorized(u_random_C, plane, camera_);

  // Get the distance of these surface points
  std::vector<Vector3f> points_L;
  points_L.reserve(p_check_L.rows());
  for (int i = 0; i < p_check_L.rows(); i++) {
    points_L.push_back(p_check_L.row(i));
  }
  std::vector<float> distances;
  std::vector<bool> success_flags;
  interpolation::interpolateOnCPU(points_L, layer_, &distances, &success_flags);
  EXPECT_EQ(success_flags.size(), kNumberOfPointsToCheck);
  EXPECT_EQ(distances.size(), success_flags.size());

  // Check that something actually got integrated
  float max_distance = std::numeric_limits<float>::min();
  auto lambda = [&max_distance](const Index3D& block_index,
                                const Index3D& voxel_index,
                                const TsdfVoxel* voxel) {
    if (voxel->distance > max_distance) max_distance = voxel->distance;
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer_, lambda);
  const float nothing_integrator_indicator = voxel_size_m_;
  EXPECT_GT(max_distance, nothing_integrator_indicator);

  // Check that all interpolations worked and that the distance is close to zero
  int num_failures = 0;
  int num_bad_flags = 0;
  for (int i = 0; i < distances.size(); i++) {
    EXPECT_TRUE(success_flags[i]);
    EXPECT_NEAR(distances[i], 0.0f,
                surface_reconstruction_allowable_distance_error_m_);
    if (std::abs(distances[i]) >
        surface_reconstruction_allowable_distance_error_m_) {
      num_failures++;
    }
  }
  LOG(INFO) << "Num points greater than allowable distance: " << num_failures;
  LOG(INFO) << "num_bad_flags: " << num_bad_flags << " / " << distances.size();
}

TEST_F(TsdfIntegratorTestParameterized, SphereSceneTest) {
  constexpr float kSphereRadius = 2.0f;
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m_;
  // Maximum distance to consider for scene generation.
  constexpr float kMaxDist = 10.0;
  constexpr float kMinWeight = 1.0;

  // Tolerance for error.
  constexpr float kDistanceErrorTolerance = kTruncationDistanceMeters;

  // Scene is bounded to -5, -5, 0 to 5, 5, 5.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  // Create a scene with a ground plane and a sphere.
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

  // Get the ground truth SDF for it.
  TsdfLayer gt_layer(voxel_size_m_, MemoryType::kUnified);
  scene.generateSdfFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create an integrator.
  ProjectiveTsdfIntegratorCPU integrator_cpu;
  ProjectiveTsdfIntegrator integrator_gpu;
  integrator_cpu.truncation_distance_vox(kTruncationDistanceVox);
  integrator_gpu.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  // Two layers, one for CPU integration and one for GPU integration
  TsdfLayer layer_cpu(layer_.voxel_size(), MemoryType::kUnified);
  TsdfLayer layer_gpu(layer_.voxel_size(), MemoryType::kUnified);

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kTrajectoryHeight);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene.generateDepthImageFromScene(camera_, T_S_C, kMaxDist, &depth_frame);

    // Integrate this depth image.
    integrator_cpu.integrateFrame(depth_frame, T_S_C, camera_, &layer_cpu);
    integrator_gpu.integrateFrame(depth_frame, T_S_C, camera_, &layer_gpu);
  }

  // Now do some checks...
  // Check every voxel in the map.
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight >= kMinWeight) {
      // Get the corresponding point from the GT layer.
      const TsdfVoxel* gt_voxel = getVoxelAtBlockAndVoxelIndex<TsdfVoxel>(
          gt_layer, block_index, voxel_index);
      if (gt_voxel != nullptr) {
        EXPECT_NEAR(voxel->distance, gt_voxel->distance,
                    kDistanceErrorTolerance);
      }
    }
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer_cpu, lambda);
  callFunctionOnAllVoxels<TsdfVoxel>(layer_gpu, lambda);

  io::outputVoxelLayerToPly(layer_gpu, "test_tsdf_projective_gpu.ply");
  io::outputVoxelLayerToPly(layer_cpu, "test_tsdf_projective_cpu.ply");
  io::outputVoxelLayerToPly(gt_layer, "test_tsdf_projective_gt.ply");

  // Compare the layers
  ASSERT_GE(layer_cpu.numAllocatedBlocks(), layer_gpu.numAllocatedBlocks());
  size_t total_num_voxels_observed = 0;
  size_t num_voxels_over_threshold = 0;
  for (const Index3D& block_index : layer_gpu.getAllBlockIndices()) {
    const auto block_cpu = layer_cpu.getBlockAtIndex(block_index);
    const auto block_gpu = layer_gpu.getBlockAtIndex(block_index);
    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          const float diff = block_gpu->voxels[x][y][z].distance -
                             block_cpu->voxels[x][y][z].distance;
          // EXPECT_NEAR(diff, 0.0f, kGPUvsCPUDifferenceThresholdM);
          if (block_gpu->voxels[x][y][z].weight > 0.0f) {
            ++total_num_voxels_observed;
            if (std::abs(diff) > kGPUvsCPUDifferenceThresholdM) {
              ++num_voxels_over_threshold;
            }
          }
        }
      }
    }
  }

  const float percentage_over_threshold =
      100.0f * (static_cast<float>(num_voxels_over_threshold) /
                static_cast<float>(total_num_voxels_observed));
  EXPECT_LT(percentage_over_threshold, kAcceptablePercentageOverThreshold);
  LOG(INFO) << "Percentage of voxels with a difference greater than "
            << kGPUvsCPUDifferenceThresholdM << ": "
            << percentage_over_threshold << "%";
  LOG(INFO) << "total_num_voxels_observed: " << total_num_voxels_observed
            << std::endl;
  LOG(INFO) << "num_voxels_over_threshold: " << num_voxels_over_threshold
            << std::endl;
}

INSTANTIATE_TEST_CASE_P(DeviceTests, TsdfIntegratorTestParameterized,
                        ::testing::Values(DeviceType::kCPU, DeviceType::kGPU));

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
