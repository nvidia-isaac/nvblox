#include <gtest/gtest.h>

#include "nvblox/io/image_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/semantics/image_projector.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

TEST(ImageProjectorTest, ExtractVoxelCentersOnGPUTest) {
  constexpr float kVoxelSize = 0.1;

  Pointcloud pointcloud(MemoryType::kUnified);
  constexpr int kNumElements = 10000;
  for (size_t i = 0; i < kNumElements; i++) {
    pointcloud.push_back(Vector3f(0.1f, 0.2f, i / 2));
  }

  DepthImageBackProjector image_back_projector;

  Pointcloud voxel_centers(MemoryType::kUnified);
  image_back_projector.pointcloudToVoxelCentersOnGPU(pointcloud, kVoxelSize,
                                                     &voxel_centers);

  EXPECT_EQ(voxel_centers.size(), kNumElements / 2);

  // Check that points coming out are all close to a voxel center
  for (const Vector3f& p : voxel_centers.points()) {
    const Vector3f corner_position =
        (p.array() / kVoxelSize).cast<int>().cast<float>() * kVoxelSize;
    const Vector3f distance_to_corner = p - corner_position;
    const Vector3f distance_to_center =
        distance_to_corner.array() - (kVoxelSize / 2.0f);
    constexpr float kMaximumDistanceToCenter = 1e-3;
    EXPECT_LT(distance_to_center.cwiseAbs().maxCoeff(),
              kMaximumDistanceToCenter);
  }
}

TEST(ImageProjectorTest, BackProjection) {
  // Sphere in a box scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-6.0f, -6.0f, -1.0f),
                                        Vector3f(6.0f, 6.0f, 6.0f));
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  constexpr static float scene_sphere_radius = 2.0f;
  const Vector3f scene_sphere_center = Vector3f(0.0f, 0.0f, 2.0f);
  scene.addPrimitive(std::make_unique<primitives::Sphere>(scene_sphere_center,
                                                          scene_sphere_radius));
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

  const float kMaxDistance = 2.0f;
  constexpr float kVoxelSize = 0.05;
  TsdfLayer tsdf_layer(kVoxelSize, MemoryType::kUnified);
  scene.generateLayerFromScene(kMaxDistance, &tsdf_layer);

  // Camera
  constexpr float fu = 300;
  constexpr float fv = 300;
  constexpr int width = 640;
  constexpr int height = 480;
  constexpr float cu = static_cast<float>(width) / 2.0f;
  constexpr float cv = static_cast<float>(height) / 2.0f;
  Camera camera(fu, fv, cu, cv, width, height);

  // Pose
  Transform T_L_C = Transform::Identity();
  T_L_C.prerotate(
      Eigen::AngleAxisf(M_PI / 2.0f, Vector3f::UnitY()).toRotationMatrix());
  T_L_C.pretranslate(Vector3f(-4.5, 0.0, 2.0));

  // Generate the depth.
  DepthImage depth_image(height, width, MemoryType::kUnified);
  constexpr float kMaxDist = 20.0f;
  scene.generateDepthImageFromScene(camera, T_L_C, kMaxDist, &depth_image);

  //
  DepthImageBackProjector image_back_projector;
  Pointcloud pointcloud_C(MemoryType::kDevice);
  image_back_projector.backProjectOnGPU(depth_image, camera, &pointcloud_C);

  // Get points back to the CPU
  const std::vector<Vector3f> points_C = pointcloud_C.points().toVector();

  // Transform points (on the CPU)
  std::vector<Vector3f> points_L;
  for (const Vector3f& p_C : points_C) {
    points_L.push_back(T_L_C * p_C);
  }

  // Check that these evaluate to approximate zero distance
  std::vector<TsdfVoxel> voxels;
  std::vector<bool> success_flags;
  tsdf_layer.getVoxels(points_L, &voxels, &success_flags);

  // Check points are near zero in the distance field
  const float kMaximumDistance = kVoxelSize;
  for (size_t i = 0; i < voxels.size(); i++) {
    EXPECT_TRUE(success_flags[i]);
    EXPECT_LT(voxels[i].distance, kMaximumDistance);
  }

  // Debug output
  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("backprojector_depth_image.png", depth_image);
    io::PlyWriter ply_writer("backprojector_pointcloud.ply");
    ply_writer.setPoints(&points_L);
    ply_writer.write();
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
