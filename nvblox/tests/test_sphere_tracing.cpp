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

#include "nvblox/core/blox.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/voxels.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/utils/timing.h"

#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

class SphereTracingTest : public ::testing::Test {
 protected:
  SphereTracingTest()
      : layer_(
            std::make_shared<TsdfLayer>(voxel_size_m_, MemoryType::kUnified)) {
    constexpr float fu = 300;
    constexpr float fv = 300;
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr float cu = static_cast<float>(width) / 2.0f;
    constexpr float cv = static_cast<float>(height) / 2.0f;
    camera_ptr_ = std::make_unique<Camera>(fu, fv, cu, cv, width, height);
  }

  // Layers
  constexpr static float voxel_size_m_ = 0.05f;
  std::shared_ptr<TsdfLayer> layer_;

  // Integration/Ground-truth distance computation
  constexpr static float truncation_distance_vox_ = 4.0f;
  constexpr static float truncation_distance_m_ =
      truncation_distance_vox_ * voxel_size_m_;

  // Test camera
  std::unique_ptr<Camera> camera_ptr_;

  // Scenes
  constexpr static float scene_sphere_radius_ = 2.0f;
  const Vector3f scene_sphere_center_ = Vector3f(0.0f, 0.0f, 2.0f);
  primitives::Scene getSphereInBoxScene() {
    // Scene is bounded to -6, -6, 0 to 6, 6, 6.
    // NOTE(alexmillane): I increased the size to 6 here because if the scene
    // has walls on the very edge, the distance just through the wall is not
    // allocated, making the ray marching go straight through the wall. Note
    // that in reality, we will always have some thickness to the wall (the
    // integrator ensures that).
    primitives::Scene scene;
    scene.aabb() = AxisAlignedBoundingBox(Vector3f(-6.0f, -6.0f, -1.0f),
                                          Vector3f(6.0f, 6.0f, 6.0f));
    scene.addGroundLevel(0.0f);
    scene.addCeiling(5.0f);
    scene.addPrimitive(std::make_unique<primitives::Sphere>(
        scene_sphere_center_, scene_sphere_radius_));
    scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
    return scene;
  }
  Transform getRandomViewpointInSphereInBoxScene() {
    // - position
    //    - inside the (-5, -5, 0) to (5, 5, 5) bounds,
    //    - and not inside the sphere in the center
    // - orientation
    //    - random
    auto is_inside_sphere = [this](const Vector3f& p) -> bool {
      return (p - scene_sphere_center_).norm() <= scene_sphere_radius_;
    };
    Vector3f p_L = Vector3f::Zero();
    while (is_inside_sphere(p_L)) {
      p_L = Vector3f(test_utils::randomFloatInRange(-4.75, 4.75),
                     test_utils::randomFloatInRange(-4.75, 4.75),
                     test_utils::randomFloatInRange(0.25, 4.75));
    }
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(
        Eigen::AngleAxisf(test_utils::randomFloatInRange(-M_PI, M_PI),
                          test_utils::getRandomUnitVector3f())
            .toRotationMatrix());
    T_S_C.pretranslate(p_L);
    return T_S_C;
  }
  void getSphereSceneReconstruction(const primitives::Scene& scene,
                                    TsdfLayer* layer_ptr) {
    // Objects to perform the reconstruction
    ProjectiveTsdfIntegrator integrator;
    integrator.truncation_distance_vox(truncation_distance_vox_);
    DepthImage depth_image(camera_ptr_->height(), camera_ptr_->width(),
                           MemoryType::kUnified);

    // Fuse in viewpoints
    constexpr float kTrajectoryRadius = 4.0f;
    constexpr float kTrajectoryHeight = 2.0f;
    constexpr int kNumTrajectoryPoints = 80;
    const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);
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
      SphereTracer::Params params;
      scene.generateDepthImageFromScene(
          *camera_ptr_, T_S_C, params.maximum_ray_length_m, &depth_image);

      // Integrate this depth image.
      integrator.integrateFrame(depth_image, T_S_C, *camera_ptr_, layer_ptr);
    }
  }
};

class SphereTracingInScaledDistanceFieldTest
    : public SphereTracingTest,
      public ::testing::WithParamInterface<float> {
 protected:
  // Yo dawg I heard you like params
};

TEST_P(SphereTracingInScaledDistanceFieldTest, PlaneTest) {
  // Get the test param
  const float distance_scaling = GetParam();

  // Test params
  constexpr float kAllowableErrorInDistanceAlongRay = voxel_size_m_;

  constexpr float kVolumeHalfSize = 5.0f;
  constexpr float kMinDistanceFromPlane = 1.0f;
  constexpr float kMinDistanceFromVolumeEdges = 0.5;

  // Scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(
      Vector3f(-kVolumeHalfSize, -kVolumeHalfSize, -kVolumeHalfSize),
      Vector3f(kVolumeHalfSize, kVolumeHalfSize, kVolumeHalfSize));
  // Create a scene with a plane in the center.
  Vector3f plane_center(0.0f, 0.0f, 0.0f);
  Vector3f plane_normal(-1.0f, 0.0f, 0.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Plane>(plane_center, plane_normal));

  // Get the ground truth SDF for it.
  const float truncation_distance_m = 5.0;
  scene.generateSdfFromScene(truncation_distance_m, layer_.get());

  // Scale the distance for all voxels in the layer
  auto scaling_lambda = [distance_scaling](const Index3D&, const Index3D&,
                                           TsdfVoxel* voxel) {
    voxel->distance *= distance_scaling;
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer_.get(), scaling_lambda);

  // Sphere tracer
  SphereTracer sphere_tracer_gpu;

  // Test a number of random rays
  constexpr int kNumTests = 1000;
  int num_converged = 0;
  int num_error_too_high = 0;
  for (int i = 0; i < kNumTests; i++) {
    // Get a random point on the plane (and in volume)
    Vector3f p_plane_L(0.0f,  // NOLINT
                       test_utils::randomFloatInRange(
                           -(kVolumeHalfSize - kMinDistanceFromVolumeEdges),
                           (kVolumeHalfSize - kMinDistanceFromVolumeEdges)),
                       test_utils::randomFloatInRange(
                           -(kVolumeHalfSize - kMinDistanceFromVolumeEdges),
                           (kVolumeHalfSize - kMinDistanceFromVolumeEdges)));

    // Get a random point in the volume (positive side of the plane)
    Vector3f p_volume_L(test_utils::randomFloatInRange(
                            -(kVolumeHalfSize - kMinDistanceFromVolumeEdges),
                            -kMinDistanceFromPlane),
                        test_utils::randomFloatInRange(
                            -(kVolumeHalfSize - kMinDistanceFromVolumeEdges),
                            (kVolumeHalfSize - kMinDistanceFromVolumeEdges)),
                        test_utils::randomFloatInRange(
                            -(kVolumeHalfSize - kMinDistanceFromVolumeEdges),
                            (kVolumeHalfSize - kMinDistanceFromVolumeEdges)));

    // Ray between these points
    const Ray ray{(p_plane_L - p_volume_L).normalized(), p_volume_L};

    // March
    float t;
    const bool marcher_converged =
        sphere_tracer_gpu.castOnGPU(ray, *layer_, truncation_distance_m_, &t);
    EXPECT_TRUE(marcher_converged);
    if (!marcher_converged) {
      continue;
    }

    // GT
    float distance_gt;
    Vector3f ray_intersection_gt;
    constexpr float kMaxDist = 20.0f;
    EXPECT_TRUE(scene.getRayIntersection(ray.origin, ray.direction, kMaxDist,
                                         &ray_intersection_gt, &distance_gt));

    // Check
    const float error = std::abs(t - distance_gt);
    bool error_too_high = error > kAllowableErrorInDistanceAlongRay;

    // Tally
    if (marcher_converged) ++num_converged;
    if (error_too_high) ++num_error_too_high;
  }

  constexpr float kAllowableErrorTooHighPercentage = 0.5f;
  float percentage_error_too_high = num_error_too_high * 100.0f / kNumTests;
  EXPECT_LT(percentage_error_too_high, kAllowableErrorTooHighPercentage);

  LOG(INFO) << "num_converged: " << num_converged << " / " << kNumTests;
  LOG(INFO) << "num_error_too_high: " << num_error_too_high << " / "
            << kNumTests;
}

INSTANTIATE_TEST_CASE_P(PlaneTests, SphereTracingInScaledDistanceFieldTest,
                        ::testing::Values(0.9f, 1.0f, 1.1f));

enum class DistanceFieldType { kGroundTruth, kReconstruction };

class SphereTracingInSphereSceneTest
    : public SphereTracingTest,
      public ::testing::WithParamInterface<DistanceFieldType> {
 protected:
  // Yo dawg I heard you like params
};

TEST_P(SphereTracingInSphereSceneTest, SphereSceneTests) {
  // Scene
  primitives::Scene scene = getSphereInBoxScene();

  // Distance Field type
  const DistanceFieldType field_type = GetParam();

  // Distance field (GT or reconsruction)
  if (field_type == DistanceFieldType::kGroundTruth) {
    std::cout << "Testing on Ground Truth distance field." << std::endl;
    scene.generateSdfFromScene(truncation_distance_m_, layer_.get());
  } else {
    std::cout << "Testing on reconstructed distance field." << std::endl;
    getSphereSceneReconstruction(scene, layer_.get());
  }

  // Sphere tracer
  SphereTracer sphere_tracer_gpu;

  // Declare the images here so we have access to them after the tests
  std::shared_ptr<const DepthImage> depth_image_sphere_traced_ptr;
  DepthImage depth_frame_gt(camera_ptr_->height(), camera_ptr_->width(),
                            MemoryType::kUnified);
  DepthImage diff;

  constexpr int kNUmImages = 10;
  for (int i = 0; i < kNUmImages; i++) {
    // Random view points
    const Transform T_S_C = getRandomViewpointInSphereInBoxScene();

    // Generate a sphere traced image
    timing::Timer render_timer("render");
    depth_image_sphere_traced_ptr = sphere_tracer_gpu.renderImageOnGPU(
        *camera_ptr_, T_S_C, *layer_, truncation_distance_m_,
        MemoryType::kUnified);
    render_timer.Stop();

    // Generate a GT image
    scene.generateDepthImageFromScene(
        *camera_ptr_, T_S_C, sphere_tracer_gpu.params().maximum_ray_length_m,
        &depth_frame_gt);

    // Error image
    test_utils::getDifferenceImageOnGPU(*depth_image_sphere_traced_ptr,
                                        depth_frame_gt, &diff);

    // Count the number of error pixels.
    // NOTE(alexmillane): We ignore rays that do not converge (this occurs in
    // small regions of the reconstruction test).
    constexpr float kErrorPixelThreshold = 4.0f * voxel_size_m_;
    int num_error_pixels = 0;
    int num_rays_converged = 0;
    for (int i = 0; i < diff.numel(); i++) {
      if ((*depth_image_sphere_traced_ptr)(i) > 0.0f) {
        ++num_rays_converged;
        if (diff(i) > kErrorPixelThreshold) {
          ++num_error_pixels;
        }
      }
    }
    const float percentage_rays_converged =
        static_cast<float>(num_rays_converged) * 100.0f / diff.numel();
    const float percentage_error_pixels =
        static_cast<float>(num_error_pixels) * 100.0f / diff.numel();
    std::cout << "percentage of rays converged "
              << " = " << percentage_rays_converged << "\%" << std::endl;
    std::cout << "number of pixels with error > " << kErrorPixelThreshold
              << ": " << num_error_pixels << " = " << percentage_error_pixels
              << "\%" << std::endl;

    constexpr float kPercentagePixelsFailThreshold = 2.0f;  // percent!
    EXPECT_LT(percentage_error_pixels, kPercentagePixelsFailThreshold);

    // In the groundtruth distance field we expect all rays to converge
    if (field_type == DistanceFieldType::kGroundTruth) {
      EXPECT_GT(percentage_rays_converged, 99.5f);
    }
  }

  std::cout << timing::Timing::Print() << std::endl;

  // Write Images
  io::writeToCsv("sphere_tracing_image.txt", *depth_image_sphere_traced_ptr);
  io::writeToCsv("sphere_tracing_gt.txt", depth_frame_gt);
  io::writeToCsv("sphere_tracing_diff.txt", diff);

  // Write Scene
  // io::outputVoxelLayerToPly(*layer_, "sphere_tracing_scene.ply");
}

INSTANTIATE_TEST_CASE_P(SphereSceneTests, SphereTracingInSphereSceneTest,
                        ::testing::Values(DistanceFieldType::kGroundTruth,
                                          DistanceFieldType::kReconstruction));

void generateErrorImageFromSubsampledImage(const DepthImage& original,
                                           const DepthImage& subsampled,
                                           DepthImage* diff) {
  CHECK_EQ(original.rows(), diff->rows());
  CHECK_EQ(original.cols(), diff->cols());
  CHECK_EQ(original.rows() % subsampled.rows(), 0);
  CHECK_EQ(original.cols() % subsampled.cols(), 0);

  const int subsampling_factor = original.rows() / subsampled.rows();
  CHECK_EQ(original.cols() / subsampled.cols(), subsampling_factor);

  for (int row_idx = 0; row_idx < original.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < original.cols(); col_idx++) {
      const Vector2f u_px_subsampled(
          static_cast<float>(col_idx) / static_cast<float>(subsampling_factor) +
              0.5,
          static_cast<float>(row_idx) / static_cast<float>(subsampling_factor) +
              0.5);

      const float original_depth = original(row_idx, col_idx);
      const bool original_success = original_depth > 0.0f;

      float subsampled_depth;
      const bool subsampled_success = interpolation::interpolate2DLinear(
          subsampled, u_px_subsampled, &subsampled_depth);

      if (original_success && subsampled_success) {
        const float depth_error = std::abs(original_depth - subsampled_depth);
        (*diff)(row_idx, col_idx) = depth_error;
      } else {
        (*diff)(row_idx, col_idx) = -1.0f;
      }
    }
  }
}

float getPercentageErrorPixels(const DepthImage& diff,
                               const float error_threshold) {
  int num_error_pixels = 0;
  for (int i = 0; i < diff.numel(); i++) {
    if ((diff)(i) > 0.0f) {
      if (diff(i) > error_threshold) {
        ++num_error_pixels;
      }
    }
  }
  return static_cast<float>(num_error_pixels) /
         static_cast<float>(diff.numel()) * 100.0f;
}

TEST_F(SphereTracingTest, SubsamplingTest) {
  // Scene
  primitives::Scene scene = getSphereInBoxScene();

  // Distance field (GT or reconsruction)
  getSphereSceneReconstruction(scene, layer_.get());

  // Sphere tracer
  SphereTracer sphere_tracer_gpu;

  // Declare the images here so we have access to them after the tests
  DepthImage depth_image_full;
  DepthImage depth_image_half;
  DepthImage depth_image_quarter;
  DepthImage diff(camera_ptr_->rows(), camera_ptr_->cols(),
                  MemoryType::kUnified);

  constexpr int kNUmImages = 10;
  for (int i = 0; i < kNUmImages; i++) {
    // Random view points
    const Transform T_S_C = getRandomViewpointInSphereInBoxScene();

    // Generate a sphere traced image
    timing::Timer render_timer_full("render/full");
    depth_image_full = *sphere_tracer_gpu.renderImageOnGPU(
        *camera_ptr_, T_S_C, *layer_, truncation_distance_m_,
        MemoryType::kUnified);
    render_timer_full.Stop();

    timing::Timer render_timer_half("render/half");
    depth_image_half = *sphere_tracer_gpu.renderImageOnGPU(
        *camera_ptr_, T_S_C, *layer_, truncation_distance_m_,
        MemoryType::kUnified, 2);
    render_timer_half.Stop();

    timing::Timer render_timer_quarter("render/quarter");
    depth_image_quarter = *sphere_tracer_gpu.renderImageOnGPU(
        *camera_ptr_, T_S_C, *layer_, truncation_distance_m_,
        MemoryType::kUnified, 4);
    render_timer_quarter.Stop();

    // Errors
    constexpr float kErrorPixelThreshold = 0.5;
    generateErrorImageFromSubsampledImage(depth_image_full, depth_image_half,
                                          &diff);
    const float percentage_error_pixels_half =
        getPercentageErrorPixels(diff, kErrorPixelThreshold);

    generateErrorImageFromSubsampledImage(depth_image_full, depth_image_quarter,
                                          &diff);
    const float percentage_error_pixels_quarter =
        getPercentageErrorPixels(diff, kErrorPixelThreshold);

    std::cout << "percentage_error_pixels_half: "
              << percentage_error_pixels_half << std::endl;
    std::cout << "percentage_error_pixels_quarter: "
              << percentage_error_pixels_quarter << std::endl;

    EXPECT_LT(percentage_error_pixels_half, 2.0f);
    EXPECT_LT(percentage_error_pixels_half, 4.0f);
  }

  std::cout << timing::Timing::Print() << std::endl;

  // Write Images
  io::writeToCsv("sphere_tracing_image_full.csv", depth_image_full);
  io::writeToCsv("sphere_tracing_image_half.csv", depth_image_half);
  io::writeToCsv("sphere_tracing_image_quarter.csv", depth_image_quarter);
  io::writeToCsv("sphere_tracing_image_subsampling_diff.csv", diff);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  warmupCuda();
  return RUN_ALL_TESTS();
}
