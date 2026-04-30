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

#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/projective_texturing/camera_view.h"
#include "nvblox/projective_texturing/projective_texture_mapper.h"
#include "nvblox/projective_texturing/texture_atlas.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace test {

// ============================================================================
// Single-camera fixture: stream + camera + 100x100 depth/color images.
// Used by all tests that need to project vertices into a camera.
// ============================================================================

class ProjectiveTextureMapperTest : public ::testing::Test {
 protected:
  static constexpr int kImageSize = 100;
  static constexpr float kFocalLength = 50.0f;
  static constexpr float kPrincipalPoint = 50.0f;
  static constexpr float kDepthEverywhere = 1.0f;

  void SetUp() override {
    stream_ = std::make_shared<CudaStreamOwning>();
    camera_ = Camera(kFocalLength, kFocalLength, kPrincipalPoint,
                     kPrincipalPoint, kImageSize, kImageSize);
    T_C_W_ = Transform::Identity();

    color_image_ = std::make_unique<ColorImage>(kImageSize, kImageSize,
                                                MemoryType::kDevice);
    depth_image_ = std::make_unique<DepthImage>(kImageSize, kImageSize,
                                                MemoryType::kDevice);

    std::vector<float> host_depth(kImageSize * kImageSize, kDepthEverywhere);
    cudaMemcpy(depth_image_->dataPtr(), host_depth.data(),
               host_depth.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  ColorMesh createTriangleMesh(const Vector3f& v0, const Vector3f& v1,
                               const Vector3f& v2) {
    ColorMesh mesh(MemoryType::kDevice);

    std::vector<Vector3f> host_verts = {v0, v1, v2};
    std::vector<Vector3f> host_normals = {
        Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, 0, -1)};
    std::vector<int> host_tris = {0, 1, 2};
    std::vector<Color> host_colors = {Color(255, 0, 0), Color(0, 255, 0),
                                      Color(0, 0, 255)};

    mesh.resizeAsync(3, *stream_);

    cudaMemcpyAsync(mesh.vertices.data(), host_verts.data(),
                    3 * sizeof(Vector3f), cudaMemcpyHostToDevice, *stream_);
    cudaMemcpyAsync(mesh.vertex_normals.data(), host_normals.data(),
                    3 * sizeof(Vector3f), cudaMemcpyHostToDevice, *stream_);
    cudaMemcpyAsync(mesh.triangles.data(), host_tris.data(), 3 * sizeof(int),
                    cudaMemcpyHostToDevice, *stream_);
    cudaMemcpyAsync(mesh.vertex_appearances.data(), host_colors.data(),
                    3 * sizeof(Color), cudaMemcpyHostToDevice, *stream_);
    stream_->synchronize();
    return mesh;
  }

  CameraView makeView() {
    return CameraView(camera_, T_C_W_, *color_image_, *depth_image_);
  }

  std::vector<Vector2f> mapAndReadbackUVs(ColorMesh* mesh,
                                          std::vector<CameraView> views) {
    ProjectiveTextureMapper mapper;
    mapper.buildAtlasAsync(std::move(views), *stream_);
    mapper.mapMesh(mesh, *stream_);
    stream_->synchronize();

    std::vector<Vector2f> host_uvs(mesh->vertex_uvs.size());
    cudaMemcpy(host_uvs.data(), mesh->vertex_uvs.data(),
               host_uvs.size() * sizeof(Vector2f), cudaMemcpyDeviceToHost);
    return host_uvs;
  }

  std::shared_ptr<CudaStreamOwning> stream_;
  Camera camera_;
  Transform T_C_W_;
  std::unique_ptr<ColorImage> color_image_;
  std::unique_ptr<DepthImage> depth_image_;
};

// ============================================================================
// TextureAtlas tests
// ============================================================================

TEST_F(ProjectiveTextureMapperTest, TextureAtlasSingleCamera) {
  TextureAtlas atlas;
  atlas.buildAtlasAsync({makeView()}, *stream_);
  stream_->synchronize();

  EXPECT_EQ(atlas.numCameras(), 1);
  EXPECT_EQ(atlas.atlasImage().width(), kImageSize);
  EXPECT_EQ(atlas.atlasImage().height(), kImageSize);

  Vector2f offset = atlas.uvOffset(0);
  Vector2f scale = atlas.uvScale(0);
  EXPECT_FLOAT_EQ(offset.x(), 0.0f);
  EXPECT_FLOAT_EQ(offset.y(), 0.0f);
  EXPECT_FLOAT_EQ(scale.x(), 1.0f);
  EXPECT_FLOAT_EQ(scale.y(), 1.0f);
}

TEST_F(ProjectiveTextureMapperTest, TextureAtlasMultiCamera) {
  auto color2 =
      std::make_unique<ColorImage>(kImageSize, kImageSize, MemoryType::kDevice);
  auto depth2 =
      std::make_unique<DepthImage>(kImageSize, kImageSize, MemoryType::kDevice);
  CameraView view1 = makeView();
  CameraView view2(camera_, T_C_W_, *color2, *depth2);

  TextureAtlas atlas;
  atlas.buildAtlasAsync({view1, view2}, *stream_);
  stream_->synchronize();

  EXPECT_EQ(atlas.numCameras(), 2);
  EXPECT_EQ(atlas.atlasImage().width(), 2 * kImageSize);
  EXPECT_EQ(atlas.atlasImage().height(), kImageSize);

  EXPECT_FLOAT_EQ(atlas.uvOffset(0).x(), 0.0f);
  EXPECT_FLOAT_EQ(atlas.uvScale(0).x(), 0.5f);
  EXPECT_FLOAT_EQ(atlas.uvOffset(1).x(), 0.5f);
  EXPECT_FLOAT_EQ(atlas.uvScale(1).x(), 0.5f);
}

// ============================================================================
// Single-camera projection tests
// ============================================================================

TEST_F(ProjectiveTextureMapperTest, VertexInFrontOfCameraGetsValidUV) {
  ColorMesh mesh = createTriangleMesh(Vector3f(-0.1f, -0.1f, 0.5f),
                                      Vector3f(0.1f, -0.1f, 0.5f),
                                      Vector3f(0.0f, 0.1f, 0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});
  ASSERT_EQ(host_uvs.size(), 3u);

  for (int i = 0; i < 3; ++i) {
    EXPECT_GE(host_uvs[i].x(), 0.0f) << "Vertex " << i;
    EXPECT_LE(host_uvs[i].x(), 1.0f) << "Vertex " << i;
    EXPECT_GE(host_uvs[i].y(), 0.0f) << "Vertex " << i;
    EXPECT_LE(host_uvs[i].y(), 1.0f) << "Vertex " << i;
  }
}

TEST_F(ProjectiveTextureMapperTest, VertexBehindCameraGetsInvalidUV) {
  ColorMesh mesh = createTriangleMesh(Vector3f(-0.1f, -0.1f, -0.5f),
                                      Vector3f(0.1f, -0.1f, -0.5f),
                                      Vector3f(0.0f, 0.1f, -0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(host_uvs[i].x(), -1.0f) << "Vertex " << i;
    EXPECT_FLOAT_EQ(host_uvs[i].y(), -1.0f) << "Vertex " << i;
  }
}

TEST_F(ProjectiveTextureMapperTest, VertexOutsideFrustumGetsInvalidUV) {
  ColorMesh mesh = createTriangleMesh(Vector3f(10.0f, 0.0f, 0.5f),
                                      Vector3f(10.1f, 0.0f, 0.5f),
                                      Vector3f(10.05f, 0.1f, 0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(host_uvs[i].x(), -1.0f) << "Vertex " << i;
    EXPECT_FLOAT_EQ(host_uvs[i].y(), -1.0f) << "Vertex " << i;
  }
}

TEST_F(ProjectiveTextureMapperTest, OccludedVertexGetsInvalidUV) {
  // Depth image is 1.0m everywhere. Triangle at z=2.0m is behind the surface.
  ColorMesh mesh = createTriangleMesh(Vector3f(-0.1f, -0.1f, 2.0f),
                                      Vector3f(0.1f, -0.1f, 2.0f),
                                      Vector3f(0.0f, 0.1f, 2.0f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(host_uvs[i].x(), -1.0f) << "Vertex " << i;
    EXPECT_FLOAT_EQ(host_uvs[i].y(), -1.0f) << "Vertex " << i;
  }
}

TEST_F(ProjectiveTextureMapperTest, AllOrNothingPerTriangle) {
  // 2 vertices in frustum, 1 outside. All-or-nothing: all get kInvalidUV.
  ColorMesh mesh = createTriangleMesh(Vector3f(0.0f, 0.0f, 0.5f),
                                      Vector3f(0.05f, 0.0f, 0.5f),
                                      Vector3f(50.0f, 0.0f, 0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(host_uvs[i].x(), -1.0f)
        << "Vertex " << i << " should be invalid (all-or-nothing)";
    EXPECT_FLOAT_EQ(host_uvs[i].y(), -1.0f)
        << "Vertex " << i << " should be invalid (all-or-nothing)";
  }
}

// ============================================================================
// UV accuracy test
// ============================================================================

TEST_F(ProjectiveTextureMapperTest, UVAccuracyAtImageCenter) {
  // Camera: fx=50, fy=50, cx=50, cy=50, 100x100 pixels.
  // Vertex at (0, 0, 1.0) -> pixel (50, 50) -> UV (0.5, 0.5).
  // Depth is 1.0m everywhere; z=1.0 is on the surface, not occluded
  // (1.0 is NOT > 1.0 + tolerance).
  ColorMesh mesh = createTriangleMesh(Vector3f(-0.01f, -0.01f, 1.0f),
                                      Vector3f(0.01f, -0.01f, 1.0f),
                                      Vector3f(0.0f, 0.01f, 1.0f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(host_uvs[i].x(), 0.5f, 0.02f)
        << "Vertex " << i << " UV.x expected near 0.5";
    EXPECT_NEAR(host_uvs[i].y(), 0.5f, 0.02f)
        << "Vertex " << i << " UV.y expected near 0.5";
  }

  // Vertex 2 at (0, 0.01, 1.0):
  //   pixel_x = 50*0.0/1.0 + 50 = 50   -> u = 50/100 = 0.5
  //   pixel_y = 50*0.01/1.0 + 50 = 50.5 -> v = 50.5/100 = 0.505
  EXPECT_NEAR(host_uvs[2].x(), 0.5f, 0.001f) << "Vertex 2 UV.x";
  EXPECT_NEAR(host_uvs[2].y(), 0.505f, 0.001f) << "Vertex 2 UV.y";
}

// ============================================================================
// Occlusion tolerance boundary test
// ============================================================================

TEST_F(ProjectiveTextureMapperTest, OcclusionToleranceBoundary) {
  // Depth is 1.0m everywhere. Default tolerance = 0.02m.
  // z=1.01: within tolerance (1.01 <= 1.0 + 0.02) -> valid
  {
    ColorMesh mesh = createTriangleMesh(Vector3f(-0.01f, -0.01f, 1.01f),
                                        Vector3f(0.01f, -0.01f, 1.01f),
                                        Vector3f(0.0f, 0.01f, 1.01f));
    auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});
    for (int i = 0; i < 3; ++i) {
      EXPECT_GE(host_uvs[i].x(), 0.0f)
          << "Vertex " << i << " within tolerance should be valid";
      EXPECT_LE(host_uvs[i].x(), 1.0f)
          << "Vertex " << i << " within tolerance should be valid";
    }
  }

  // z=1.03: beyond tolerance (1.03 > 1.0 + 0.02) -> occluded
  {
    ColorMesh mesh = createTriangleMesh(Vector3f(-0.01f, -0.01f, 1.03f),
                                        Vector3f(0.01f, -0.01f, 1.03f),
                                        Vector3f(0.0f, 0.01f, 1.03f));
    auto host_uvs = mapAndReadbackUVs(&mesh, {makeView()});
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(host_uvs[i].x(), -1.0f)
          << "Vertex " << i << " beyond tolerance should be occluded";
      EXPECT_FLOAT_EQ(host_uvs[i].y(), -1.0f)
          << "Vertex " << i << " beyond tolerance should be occluded";
    }
  }
}

// ============================================================================
// Multi-camera tests
// ============================================================================

TEST_F(ProjectiveTextureMapperTest, MultiCameraBestViewSelection) {
  // Camera 0: at origin looking +Z (from fixture).
  // Camera 1: shifted 2m right in world frame.
  // Triangle at (0, 0, 0.5) directly faces camera 0 -> UV in left half.
  Transform T_C1_W = Transform::Identity();
  T_C1_W.translate(Vector3f(-2.0f, 0.0f, 0.0f));

  auto color2 =
      std::make_unique<ColorImage>(kImageSize, kImageSize, MemoryType::kDevice);
  auto depth2 =
      std::make_unique<DepthImage>(kImageSize, kImageSize, MemoryType::kDevice);
  std::vector<float> host_depth2(kImageSize * kImageSize, 5.0f);
  cudaMemcpy(depth2->dataPtr(), host_depth2.data(),
             host_depth2.size() * sizeof(float), cudaMemcpyHostToDevice);

  CameraView view0 = makeView();
  CameraView view1(camera_, T_C1_W, *color2, *depth2);

  ColorMesh mesh = createTriangleMesh(Vector3f(-0.05f, -0.05f, 0.5f),
                                      Vector3f(0.05f, -0.05f, 0.5f),
                                      Vector3f(0.0f, 0.05f, 0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {view0, view1});
  ASSERT_EQ(host_uvs.size(), 3u);

  // Atlas is 200 wide. Camera 0 = left half [0, 0.5). Camera 1 = right half.
  for (int i = 0; i < 3; ++i) {
    EXPECT_GE(host_uvs[i].x(), 0.0f) << "Vertex " << i;
    EXPECT_LT(host_uvs[i].x(), 0.5f)
        << "Vertex " << i << " should be in camera 0 region (left half)";
    EXPECT_GE(host_uvs[i].y(), 0.0f) << "Vertex " << i;
    EXPECT_LE(host_uvs[i].y(), 1.0f) << "Vertex " << i;
  }
}

TEST_F(ProjectiveTextureMapperTest, MultiCameraFallbackToSecondBest) {
  // Camera 0: narrow FOV (10x10), can see triangle center but not all vertices.
  // Camera 1: wide FOV (100x100), sees all vertices.
  // Fallback: camera 0 fails all-or-nothing -> falls back to camera 1.
  constexpr int kNarrowSize = 10;
  Camera narrow_camera(kFocalLength, kFocalLength, 5.0f, 5.0f, kNarrowSize,
                       kNarrowSize);
  auto color_narrow = std::make_unique<ColorImage>(kNarrowSize, kNarrowSize,
                                                   MemoryType::kDevice);
  auto depth_narrow = std::make_unique<DepthImage>(kNarrowSize, kNarrowSize,
                                                   MemoryType::kDevice);
  std::vector<float> host_depth_narrow(kNarrowSize * kNarrowSize, 5.0f);
  cudaMemcpy(depth_narrow->dataPtr(), host_depth_narrow.data(),
             host_depth_narrow.size() * sizeof(float), cudaMemcpyHostToDevice);

  auto color_wide =
      std::make_unique<ColorImage>(kImageSize, kImageSize, MemoryType::kDevice);
  auto depth_wide =
      std::make_unique<DepthImage>(kImageSize, kImageSize, MemoryType::kDevice);
  std::vector<float> host_depth_wide(kImageSize * kImageSize, 5.0f);
  cudaMemcpy(depth_wide->dataPtr(), host_depth_wide.data(),
             host_depth_wide.size() * sizeof(float), cudaMemcpyHostToDevice);

  CameraView view0(narrow_camera, T_C_W_, *color_narrow, *depth_narrow);
  CameraView view1(camera_, T_C_W_, *color_wide, *depth_wide);

  ColorMesh mesh = createTriangleMesh(Vector3f(-0.15f, -0.15f, 0.5f),
                                      Vector3f(0.15f, -0.15f, 0.5f),
                                      Vector3f(0.0f, 0.15f, 0.5f));

  auto host_uvs = mapAndReadbackUVs(&mesh, {view0, view1});
  ASSERT_EQ(host_uvs.size(), 3u);

  for (int i = 0; i < 3; ++i) {
    EXPECT_GE(host_uvs[i].x(), 0.0f)
        << "Vertex " << i << " should have valid UV after fallback";
    EXPECT_LE(host_uvs[i].x(), 1.0f)
        << "Vertex " << i << " should have valid UV after fallback";
    EXPECT_GE(host_uvs[i].y(), 0.0f)
        << "Vertex " << i << " should have valid UV after fallback";
    EXPECT_LE(host_uvs[i].y(), 1.0f)
        << "Vertex " << i << " should have valid UV after fallback";
  }

  // UVs should land in camera 1's atlas region (right half).
  // Atlas is 110 wide (10 + 100). Camera 1 offset = 10/110.
  float camera1_offset_x = static_cast<float>(kNarrowSize) /
                           static_cast<float>(kNarrowSize + kImageSize);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GE(host_uvs[i].x(), camera1_offset_x)
        << "Vertex " << i << " should be in camera 1 region";
  }
}

}  // namespace test
}  // namespace nvblox

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
