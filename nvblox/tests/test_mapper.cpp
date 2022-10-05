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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/bounding_spheres.h"
#include "nvblox/core/mapper.h"
#include "nvblox/core/types.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/primitives/scene.h"

using namespace nvblox;

primitives::Scene getSphereInABoxScene(const Vector3f& sphere_center,
                                       const float sphere_radius) {
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-11.0f, -11.0f, -1.0f),
                                        Vector3f(11.0f, 11.0f, 11.0f));
  scene.addGroundLevel(0.0f);
  scene.addCeiling(10.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(sphere_center, sphere_radius));
  scene.addPlaneBoundaries(-10.0f, 10.0f, -10.0f, 10.0f);
  return scene;
}

bool allMeshPointsOnSphere(const MeshLayer& mesh_layer, const Vector3f& center,
                           const float sphere_radius) {
  Mesh mesh = Mesh::fromLayer(mesh_layer);
  for (const Vector3f p : mesh.vertices) {
    constexpr float kVertexEps = 0.01;
    if (std::abs((p - center).norm() - sphere_radius) > kVertexEps) {
      return false;
    }
  }
  return true;
}

TEST(MapperTest, ClearOutsideSphere) {
  // Create a scene with a sphere
  const Vector3f sphere_center(0.0f, 0.0f, 5.0f);
  const float sphere_radius = 2.0f;
  primitives::Scene scene = getSphereInABoxScene(sphere_center, sphere_radius);

  constexpr float voxel_size_m = 0.1;
  RgbdMapper mapper(voxel_size_m, MemoryType::kUnified);

  scene.generateSdfFromScene(1.0, &mapper.tsdf_layer());
  EXPECT_GT(mapper.tsdf_layer().numAllocatedBlocks(), 0);

  mapper.generateMesh();
  mapper.generateEsdf();

  // allocate color, just so we can clear later
  for (const Index3D& idx : mapper.tsdf_layer().getAllBlockIndices()) {
    mapper.color_layer().allocateBlockAtIndex(idx);
  }

  // Not all mesh points are on the sphere (walls are there).
  EXPECT_FALSE(
      allMeshPointsOnSphere(mapper.mesh_layer(), sphere_center, sphere_radius));

  // Clearing outside of sphere
  mapper.clearOutsideRadius(sphere_center, sphere_radius);

  EXPECT_EQ(mapper.tsdf_layer().numAllocatedBlocks(),
            mapper.esdf_layer().numAllocatedBlocks());
  EXPECT_EQ(mapper.tsdf_layer().numAllocatedBlocks(),
            mapper.color_layer().numAllocatedBlocks());

  // Test resulting mesh
  EXPECT_TRUE(
      allMeshPointsOnSphere(mapper.mesh_layer(), sphere_center, sphere_radius));

  constexpr bool kOutputMesh = false;
  if (kOutputMesh) {
    io::outputMeshLayerToPly(mapper.mesh_layer(), "mapper_test.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
