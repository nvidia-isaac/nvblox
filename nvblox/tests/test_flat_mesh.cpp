/*
Copyright 2026 NVIDIA CORPORATION

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

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/flat_mesh_integrator.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/tests/utils.h"
#include "nvblox/utils/timing.h"

using namespace nvblox;

class FlatMeshTest : public ::testing::Test {
 protected:
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);

    tsdf_layer_.reset(new TsdfLayer(voxel_size_, MemoryType::kUnified));
    color_layer_.reset(new ColorLayer(voxel_size_, MemoryType::kUnified));

    // Scene: sphere in a box (same as test_mesh.cpp).
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                           Vector3f(3.0f, 3.0f, 3.0f));
    scene_.addPrimitive(
        std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 1.5f), 1.0));
  }

  // Match test_mesh.cpp voxel size for consistent grid alignment.
  float voxel_size_ = 0.10;

  TsdfLayer::Ptr tsdf_layer_;
  ColorLayer::Ptr color_layer_;

  ColorFlatMeshIntegrator flat_mesh_integrator_;
  ColorMeshIntegrator mesh_integrator_;

  primitives::Scene scene_;
};

// Test that an empty input produces an empty mesh without crashing.
TEST_F(FlatMeshTest, EmptyInput) {
  ColorMesh mesh;
  std::vector<Index3D> empty_indices;

  // Should not crash.
  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, empty_indices, &mesh);

  EXPECT_EQ(mesh.vertices.size(), 0);
  EXPECT_EQ(mesh.vertex_normals.size(), 0);
  EXPECT_EQ(mesh.triangles.size(), 0);
  EXPECT_EQ(mesh.vertex_appearances.size(), 0);
}

// Test that a blank (no data) TSDF layer produces an empty mesh.
TEST_F(FlatMeshTest, BlankMap) {
  ColorMesh mesh;
  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_EQ(block_indices.size(), 0);

  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, block_indices, &mesh);
  EXPECT_EQ(mesh.vertices.size(), 0);
}

// Test geometry extraction from a plane and verify vertex positions and
// normals.
TEST_F(FlatMeshTest, PlaneGeometry) {
  // Reset scene to just a plane at x=0 pointing in -x direction.
  primitives::Scene plane_scene;
  plane_scene.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                              Vector3f(3.0f, 3.0f, 3.0f));
  plane_scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0, 0.0, 0.0), Vector3f(-1, 0, 0)));

  plane_scene.generateLayerFromScene(4 * voxel_size_, tsdf_layer_.get());

  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_GT(block_indices.size(), 0);

  ColorMesh mesh(MemoryType::kUnified);
  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, block_indices, &mesh);

  // Must produce vertices.
  EXPECT_GT(mesh.vertices.size(), 0);

  // Sizes must be consistent: vertices == normals, triangles == vertices
  // (trivial index buffer).
  EXPECT_EQ(mesh.vertices.size(), mesh.vertex_normals.size());
  EXPECT_EQ(mesh.vertices.size(), mesh.triangles.size());

  // Vertex count must be a multiple of 3 (triangle list).
  EXPECT_EQ(mesh.vertices.size() % 3, 0);

  // Verify vertex positions lie on the plane (x ≈ 0) and normals point in -x.
  // Same tolerance as test_mesh.cpp PlaneMesh test.
  constexpr float kFloatEpsilon = 1e-4;
  for (size_t i = 0; i < mesh.vertices.size(); i++) {
    EXPECT_NEAR(mesh.vertices[i].x(), 0.0, kFloatEpsilon);
    EXPECT_NEAR(mesh.vertex_normals[i].x(), -1.0, kFloatEpsilon);
    EXPECT_NEAR(mesh.vertex_normals[i].y(), 0.0, kFloatEpsilon);
    EXPECT_NEAR(mesh.vertex_normals[i].z(), 0.0, kFloatEpsilon);
  }

  // Verify trivial index buffer: triangles[i] == i.
  for (size_t i = 0; i < mesh.triangles.size(); i++) {
    EXPECT_EQ(mesh.triangles[i], static_cast<int>(i));
  }
}

// Test that flat mesh geometry matches the existing MeshIntegrator output.
// The flat mesh doesn't do vertex welding, so we compare bounding boxes and
// verify all vertices lie near the sphere surface.
TEST_F(FlatMeshTest, GeometryMatchesMeshIntegrator) {
  scene_.generateLayerFromScene(4 * voxel_size_, tsdf_layer_.get());

  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_GT(block_indices.size(), 0);

  // --- Flat mesh ---
  ColorMesh flat_mesh(MemoryType::kUnified);
  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, block_indices,
                                        &flat_mesh);
  ASSERT_GT(flat_mesh.vertices.size(), 0);

  // --- Block-based mesh (reference) ---
  float block_size = tsdf_layer_->block_size();
  ColorMeshLayer mesh_layer(block_size, MemoryType::kUnified);
  mesh_integrator_.weld_vertices(false);
  EXPECT_TRUE(mesh_integrator_.integrateMeshFromDistanceField(*tsdf_layer_,
                                                              &mesh_layer));

  CudaStreamOwning stream;
  auto ref_mesh = mesh_layer.getMesh(stream);
  ASSERT_GT(ref_mesh->vertices.size(), 0);

  // Copy reference mesh vertices to host (getMesh returns device memory).
  unified_vector<Vector3f> ref_vertices_host(MemoryType::kHost);
  ref_vertices_host.copyFromAsync(ref_mesh->vertices, stream);
  stream.synchronize();

  // Compare bounding boxes: should be very similar.
  auto computeBBox = [](const Vector3f* data,
                        size_t size) -> std::pair<Vector3f, Vector3f> {
    Vector3f min_pt = Vector3f::Constant(std::numeric_limits<float>::max());
    Vector3f max_pt = Vector3f::Constant(std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < size; i++) {
      min_pt = min_pt.cwiseMin(data[i]);
      max_pt = max_pt.cwiseMax(data[i]);
    }
    return {min_pt, max_pt};
  };

  auto [flat_min, flat_max] =
      computeBBox(flat_mesh.vertices.data(), flat_mesh.vertices.size());
  auto [ref_min, ref_max] =
      computeBBox(ref_vertices_host.data(), ref_vertices_host.size());

  // Bounding boxes should match within a voxel.
  for (int d = 0; d < 3; d++) {
    EXPECT_NEAR(flat_min[d], ref_min[d], voxel_size_);
    EXPECT_NEAR(flat_max[d], ref_max[d], voxel_size_);
  }

  // All flat mesh vertices should lie near the sphere surface (radius 1.0,
  // center (0, 0, 1.5)). Tolerance is the voxel size since marching cubes
  // interpolates between voxel corners.
  const Vector3f sphere_center(0.0f, 0.0f, 1.5f);
  const float sphere_radius = 1.0f;
  for (size_t i = 0; i < flat_mesh.vertices.size(); i++) {
    float dist = (flat_mesh.vertices[i] - sphere_center).norm();
    EXPECT_NEAR(dist, sphere_radius, voxel_size_)
        << "Vertex " << i << " at distance " << dist << " from sphere center";
  }

  if (FLAGS_nvblox_test_file_output) {
    io::outputColorMeshLayerToPly(mesh_layer, "test_flat_mesh_ref.ply");
  }
}

// Test repeatability: running twice on the same input produces the same output.
TEST_F(FlatMeshTest, Repeatability) {
  scene_.generateLayerFromScene(4 * voxel_size_, tsdf_layer_.get());

  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_GT(block_indices.size(), 0);

  ColorMesh mesh1(MemoryType::kUnified);
  ColorMesh mesh2(MemoryType::kUnified);

  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, block_indices, &mesh1);
  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, block_indices, &mesh2);

  ASSERT_EQ(mesh1.vertices.size(), mesh2.vertices.size());
  ASSERT_EQ(mesh1.vertex_normals.size(), mesh2.vertex_normals.size());
  ASSERT_EQ(mesh1.triangles.size(), mesh2.triangles.size());

  // Atomic-based triangle allocation is deterministic for same input on the
  // same GPU, but we sort to make the comparison order-independent regardless.
  auto threed_less = [](const Vector3f& a, const Vector3f& b) -> bool {
    if (a.x() != b.x()) return a.x() < b.x();
    if (a.y() != b.y()) return a.y() < b.y();
    return a.z() < b.z();
  };

  std::vector<Vector3f> verts1(mesh1.vertices.begin(), mesh1.vertices.end());
  std::vector<Vector3f> verts2(mesh2.vertices.begin(), mesh2.vertices.end());
  std::sort(verts1.begin(), verts1.end(), threed_less);
  std::sort(verts2.begin(), verts2.end(), threed_less);

  for (size_t i = 0; i < verts1.size(); i++) {
    EXPECT_TRUE((verts1[i].array() == verts2[i].array()).all())
        << "Mismatch at vertex " << i;
  }
}

// Test the appearance (color) overload: geometry + color in a single pass.
TEST_F(FlatMeshTest, WithAppearance) {
  scene_.generateLayerFromScene(4 * voxel_size_, tsdf_layer_.get());

  // Fill the color layer with a solid purple color at all voxel locations
  // that have TSDF data.
  const Color kTestColor = Color::Purple();
  for (const Index3D& block_idx : tsdf_layer_->getAllBlockIndices()) {
    ColorBlock::Ptr color_block = color_layer_->allocateBlockAtIndex(block_idx);
    callFunctionOnAllVoxels<ColorVoxel>(color_block.get(),
                                        [&](const Index3D&, ColorVoxel* voxel) {
                                          voxel->color = kTestColor;
                                          voxel->weight = 1.0f;
                                        });
  }

  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_GT(block_indices.size(), 0);

  ColorMesh mesh(MemoryType::kUnified);
  flat_mesh_integrator_.integrateBlocks(*tsdf_layer_, *color_layer_,
                                        block_indices, &mesh);

  ASSERT_GT(mesh.vertices.size(), 0);
  ASSERT_GT(mesh.vertex_appearances.size(), 0);
  EXPECT_EQ(mesh.vertices.size(), mesh.vertex_appearances.size());

  // Verify that most vertices got the test color (some edge voxels may have
  // default appearance due to missing neighbor blocks in the color layer).
  int num_matching = 0;
  for (size_t i = 0; i < mesh.vertex_appearances.size(); i++) {
    if (mesh.vertex_appearances[i] == kTestColor) {
      num_matching++;
    }
  }
  // At least 50% should match (conservative; in practice nearly all will).
  EXPECT_GT(num_matching, static_cast<int>(mesh.vertex_appearances.size()) / 2)
      << "Only " << num_matching << " of " << mesh.vertex_appearances.size()
      << " vertices have the expected color";
}

// Test auto-grow: start with a tiny buffer, verify the mesh is still complete.
// The kernel uses `continue` (not `return`) on overflow, so the triangle
// counter always reflects the exact true demand and one retry suffices.
TEST_F(FlatMeshTest, AutoGrow) {
  scene_.generateLayerFromScene(4 * voxel_size_, tsdf_layer_.get());

  std::vector<Index3D> block_indices = tsdf_layer_->getAllBlockIndices();
  ASSERT_GT(block_indices.size(), 0);

  // First run with a large buffer to get the true triangle count.
  ColorFlatMeshIntegrator large_integrator;
  large_integrator.max_num_triangles(2000000);
  ColorMesh large_mesh(MemoryType::kUnified);
  large_integrator.integrateBlocks(*tsdf_layer_, block_indices, &large_mesh);
  ASSERT_GT(large_mesh.vertices.size(), 0);

  const size_t expected_vertices = large_mesh.vertices.size();

  // Run with a tiny buffer that will definitely overflow. The kernel counts
  // all triangles even when the buffer is full (continue instead of return),
  // so the auto-grow knows the exact demand and one retry always suffices.
  ColorFlatMeshIntegrator small_integrator;
  small_integrator.max_num_triangles(10);
  ColorMesh small_mesh(MemoryType::kUnified);
  small_integrator.integrateBlocks(*tsdf_layer_, block_indices, &small_mesh);

  // After auto-grow, the output should match the large-buffer run.
  EXPECT_EQ(small_mesh.vertices.size(), expected_vertices)
      << "Auto-grow should produce the same output as a large buffer. "
      << "Small: " << small_mesh.vertices.size()
      << " Large: " << expected_vertices;

  // The integrator's max_num_triangles should have grown past the initial
  // value.
  EXPECT_GT(small_integrator.max_num_triangles(), 10);
}

// Helper: copy device vertices from a ColorMesh to a host vector.
static std::vector<Vector3f> copyVerticesToHost(const ColorMesh& mesh) {
  CudaStreamOwning stream;
  unified_vector<Vector3f> host_verts(MemoryType::kHost);
  host_verts.copyFromAsync(mesh.vertices, stream);
  stream.synchronize();
  return std::vector<Vector3f>(host_verts.begin(), host_verts.end());
}

// Test frustum-culled flat mesh extraction via Mapper::updateFlatMesh
// with camera parameters. Creates geometry at two well-separated locations,
// points the camera at only one, and verifies the output mesh contains only
// geometry from the visible region.
TEST(FlatMeshFrustumCullingTest, OnlyVisibleBlocksMeshed) {
  constexpr float kVoxelSize = 0.10f;

  Mapper mapper(kVoxelSize, MemoryType::kUnified);

  // Place two spheres far apart so their blocks don't overlap:
  //   Sphere A at (0, 0, 1.5)  -- camera will look at this one
  //   Sphere B at (20, 0, 1.5) -- outside camera FOV
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, -1.0f),
                                        Vector3f(23.0f, 3.0f, 4.0f));
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 1.5f), 1.0));
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(20.0f, 0.0f, 1.5f), 1.0));

  scene.generateLayerFromScene(4 * kVoxelSize,
                               mapper.layers().getPtr<TsdfLayer>());

  // Fill color layer so the appearance overload works.
  const Color kTestColor = Color::Purple();
  TsdfLayer& tsdf = mapper.tsdf_layer();
  ColorLayer& colors = mapper.color_layer();
  for (const Index3D& idx : tsdf.getAllBlockIndices()) {
    ColorBlock::Ptr cb = colors.allocateBlockAtIndex(idx);
    callFunctionOnAllVoxels<ColorVoxel>(cb.get(),
                                        [&](const Index3D&, ColorVoxel* voxel) {
                                          voxel->color = kTestColor;
                                          voxel->weight = 1.0f;
                                        });
  }

  // Full (un-culled) extraction -- the internal mesh is device memory, so
  // we copy vertices to host for inspection.
  mapper.updateFlatMesh<ColorVoxel>();
  const ColorMesh& mesh_ref = mapper.flat_color_mesh();
  ASSERT_GT(mesh_ref.vertices.size(), 0);

  const size_t full_vertex_count = mesh_ref.vertices.size();
  std::vector<Vector3f> full_verts = copyVerticesToHost(mesh_ref);

  bool has_near_a = false;
  bool has_near_b = false;
  const Vector3f center_a(0.0f, 0.0f, 1.5f);
  const Vector3f center_b(20.0f, 0.0f, 1.5f);
  for (const auto& v : full_verts) {
    if ((v - center_a).norm() < 2.0f) has_near_a = true;
    if ((v - center_b).norm() < 2.0f) has_near_b = true;
  }
  EXPECT_TRUE(has_near_a) << "Full mesh should contain sphere A vertices";
  EXPECT_TRUE(has_near_b) << "Full mesh should contain sphere B vertices";

  // Camera at (0,0,-5) looking along +z (camera convention: z-forward).
  // With max_depth=10, the frustum reaches z=5, covering sphere A at z=1.5
  // but nowhere near sphere B at x=20.
  Camera camera(300.0f, 300.0f, 320.0f, 240.0f, 640, 480);
  Transform T_L_C = Transform::Identity();
  T_L_C.pretranslate(Vector3f(0.0f, 0.0f, -5.0f));
  constexpr float kMaxDepth = 10.0f;

  mapper.updateFlatMesh<ColorVoxel>(camera, T_L_C, kMaxDepth);
  ASSERT_GT(mesh_ref.vertices.size(), 0);

  // Culled mesh should have fewer vertices than the full mesh.
  EXPECT_LT(mesh_ref.vertices.size(), full_vertex_count);

  std::vector<Vector3f> culled_verts = copyVerticesToHost(mesh_ref);

  // All culled vertices should be near sphere A, none near sphere B.
  for (size_t i = 0; i < culled_verts.size(); i++) {
    float dist_to_b = (culled_verts[i] - center_b).norm();
    EXPECT_GT(dist_to_b, 5.0f)
        << "Vertex " << i << " is too close to sphere B (should be culled)";
  }

  // Verify the culled mesh has correct structure.
  EXPECT_EQ(mesh_ref.vertices.size(), mesh_ref.vertex_normals.size());
  EXPECT_EQ(mesh_ref.vertices.size(), mesh_ref.triangles.size());
  EXPECT_EQ(mesh_ref.vertices.size(), mesh_ref.vertex_appearances.size());
  EXPECT_EQ(mesh_ref.vertices.size() % 3, 0u);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
