/*
Copyright 2024 NVIDIA CORPORATION

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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/tsdf_zero_crossings_extractor.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/tsdf_zero_crossings_extractor_cpu.h"

using namespace nvblox;

namespace {
bool IsContainedInVector(const Vector3f& actual,
                         const std::vector<Vector3f>& expected,
                         float tolerance = 1e-5f) {
  for (const auto& vec : expected) {
    if ((actual - vec).norm() <= tolerance) {
      return true;
    }
  }
  return false;
}
}  // namespace

class ZeroCrossingsFromAboveSimplePlane : public ::testing::Test {
 protected:
  constexpr static float voxel_size_m = 0.1;
  primitives::Scene scene;
  AxisAlignedBoundingBox aabb;

  void SetUp() override {
    aabb = AxisAlignedBoundingBox(Vector3f(0.0, 0.0, -2.0),
                                  Vector3f(0.2, 0.2, 2.0));
    scene.addGroundLevel(-1.0f);
    scene.aabb() = aabb;
  }

  constexpr static int expected_num_crossings = 4;
  // Expect the crossings to be at z=-1 & the x,y midpoints of each voxel
  std::vector<Vector3f> expected_crossings_locations = {{0.05, 0.05, -1.0},
                                                        {0.15, 0.05, -1.0},
                                                        {0.05, 0.15, -1.0},
                                                        {0.15, 0.15, -1.0}};

  // Choose a low tolerance as we expect a near perfect result on a plane.
  const float tolerance = 0.000001;
};

TEST_F(ZeroCrossingsFromAboveSimplePlane, CPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);

  TsdfZeroCrossingsExtractorCPU tsdf_zero_crossings_extractor_cpu;
  EXPECT_TRUE(
      tsdf_zero_crossings_extractor_cpu.computeZeroCrossingsFromAboveOnCPU(
          mapper.tsdf_layer()));
  const std::vector<Vector3f> actual_crossings_locations_cpu =
      tsdf_zero_crossings_extractor_cpu.getZeroCrossingsHost();

  // Check each returned crossing is part of the expected crossings
  for (const auto& actual_crossing : actual_crossings_locations_cpu) {
    EXPECT_TRUE(
        IsContainedInVector(actual_crossing, expected_crossings_locations));
  }

  const int actual_num_crossings_cpu = actual_crossings_locations_cpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_cpu);
}

TEST_F(ZeroCrossingsFromAboveSimplePlane, GPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);

  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor(
      std::make_shared<CudaStreamOwning>());

  std::optional<std::vector<Vector3f>> maybe_zero_crossings =
      tsdf_zero_crossings_extractor.computeZeroCrossingsFromAboveOnGPU(
          mapper.tsdf_layer());

  ASSERT_TRUE(maybe_zero_crossings.has_value());
  const std::vector<Vector3f> actual_crossings_locations_gpu =
      maybe_zero_crossings.value();

  const int actual_num_crossings_gpu = actual_crossings_locations_gpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_gpu);

  // Check each returned crossing is part of the expected crossings
  for (const auto& actual_crossing : actual_crossings_locations_gpu) {
    EXPECT_TRUE(
        IsContainedInVector(actual_crossing, expected_crossings_locations));
  }
}

class ZeroCrossingsFromAboveSimplePlaneAtBoundary : public ::testing::Test {
 protected:
  constexpr static float voxel_size_m = 0.1;
  // To test if the boundaries are caught we place the plane into the lower
  // half of the lowest voxels (< voxel_size_m/2)
  const float ground_level_height = 0.04;
  constexpr static float kMinTsdfWeight = 0.1;
  constexpr static int max_crossings = 1000;
  primitives::Scene scene;
  AxisAlignedBoundingBox aabb;

  void SetUp() override {
    aabb = AxisAlignedBoundingBox(Vector3f(0.0, 0.0, -2.0),
                                  Vector3f(0.2, 0.2, 2.0));
    scene.addGroundLevel(ground_level_height);
    scene.aabb() = aabb;
  }

  constexpr static int expected_num_crossings = 4;
  // Expect the crossings to be at z= & the x,y midpoints of each voxel
  std::vector<Vector3f> expected_crossings_locations = {{0.05, 0.05, 0.04},
                                                        {0.15, 0.05, 0.04},
                                                        {0.05, 0.15, 0.04},
                                                        {0.15, 0.15, 0.04}};

  // Choose a low tolerance as we expect a near perfect result on a plane.
  const float tolerance = 0.000001;
};

TEST_F(ZeroCrossingsFromAboveSimplePlaneAtBoundary, GPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);

  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor(
      std::make_shared<CudaStreamOwning>());
  std::optional<std::vector<Vector3f>> maybe_zero_crossings =
      tsdf_zero_crossings_extractor.computeZeroCrossingsFromAboveOnGPU(
          mapper.tsdf_layer());
  ASSERT_TRUE(maybe_zero_crossings.has_value());
  const std::vector<Vector3f> actual_crossings_locations_gpu =
      maybe_zero_crossings.value();

  const int actual_num_crossings_gpu = actual_crossings_locations_gpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_gpu);

  // Check each returned crossing is part of the expected crossings
  for (const auto& actual_crossing : actual_crossings_locations_gpu) {
    EXPECT_TRUE(
        IsContainedInVector(actual_crossing, expected_crossings_locations));
  }
}

class ZeroCrossingsFromAboveSimpleSphere : public ::testing::Test {
 protected:
  constexpr static float voxel_size_m = 0.1;
  primitives::Scene scene;
  AxisAlignedBoundingBox aabb;
  const float sphere_radius = 0.1;
  const Vector3f sphere_origin = Vector3f(0.0, 0.0, 0.0);

  void SetUp() override {
    scene.addPrimitive(
        std::make_unique<primitives::Sphere>(sphere_origin, sphere_radius));
  }

  constexpr static int expected_num_crossings = 4;
  // X& Y locations are expected to be at half the voxel length.
  // x^2 + y^2 + z^2 = r^2
  // z = 0.07071
  const float half_voxel_size_m = 0.5 * voxel_size_m;
  std::vector<Vector3f> expected_crossings_loc = {
      {half_voxel_size_m, half_voxel_size_m, 0.07071},
      {half_voxel_size_m, -half_voxel_size_m, 0.07071},
      {-half_voxel_size_m, -half_voxel_size_m, 0.07071},
      {-half_voxel_size_m, half_voxel_size_m, 0.07071}};

  // Choose a high tolerance because we interpolate due to the curved
  // nature of a sphere.
  const float tolerance = 0.004;
};

TEST_F(ZeroCrossingsFromAboveSimpleSphere, CPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);
  mapper.updateColorMesh(UpdateFullLayer::kYes);

  TsdfZeroCrossingsExtractorCPU tsdf_zero_crossings_extractor_cpu;
  EXPECT_TRUE(
      tsdf_zero_crossings_extractor_cpu.computeZeroCrossingsFromAboveOnCPU(
          mapper.tsdf_layer()));
  const std::vector<Vector3f> actual_crossings_loc_cpu =
      tsdf_zero_crossings_extractor_cpu.getZeroCrossingsHost();

  const int actual_num_crossings_cpu = actual_crossings_loc_cpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_cpu);

  // Check each returned crossing is part of the expected crossings
  for (const auto& actual_crossing : actual_crossings_loc_cpu) {
    EXPECT_TRUE(IsContainedInVector(actual_crossing, expected_crossings_loc,
                                    tolerance));
  }
}

TEST_F(ZeroCrossingsFromAboveSimpleSphere, GPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);
  mapper.updateColorMesh(UpdateFullLayer::kYes);

  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor(
      std::make_shared<CudaStreamOwning>());

  std::optional<std::vector<Vector3f>> maybe_zero_crossings =
      tsdf_zero_crossings_extractor.computeZeroCrossingsFromAboveOnGPU(
          mapper.tsdf_layer());

  ASSERT_TRUE(maybe_zero_crossings.has_value());
  const std::vector<Vector3f> actual_crossings_loc_gpu =
      maybe_zero_crossings.value();

  const int actual_num_crossings_gpu = actual_crossings_loc_gpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_gpu);

  // Check each returned crossing is part of the expected crossings
  for (const auto& actual_crossing : actual_crossings_loc_gpu) {
    EXPECT_TRUE(IsContainedInVector(actual_crossing, expected_crossings_loc,
                                    tolerance));
  }
}

TEST_F(ZeroCrossingsFromAboveSimpleSphere, GPUTestMaxCrossingsExceeded) {
  const int max_crossings_too_low = 2;
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);
  mapper.updateColorMesh(UpdateFullLayer::kYes);

  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor(
      std::make_shared<CudaStreamOwning>());
  tsdf_zero_crossings_extractor.max_crossings(max_crossings_too_low);

  // If exceeded, resize the buffer to fit all the crossings.
  EXPECT_TRUE(tsdf_zero_crossings_extractor.computeZeroCrossingsFromAboveOnGPU(
      mapper.tsdf_layer()));
}

class ZeroCrossingsFromAboveEmptyScene : public ::testing::Test {
 protected:
  constexpr static float voxel_size_m = 0.1;
  primitives::Scene scene;
  AxisAlignedBoundingBox aabb;

  void SetUp() override {}

  constexpr static int expected_num_crossings = 0;

  // Choose a low tolerance as we expect a near perfect result on a plane.
  const float tolerance = 0.000001;
};

TEST_F(ZeroCrossingsFromAboveEmptyScene, GPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);
  mapper.updateColorMesh(UpdateFullLayer::kYes);

  TsdfZeroCrossingsExtractor tsdf_zero_crossings_extractor(
      std::make_shared<CudaStreamOwning>());

  std::optional<std::vector<Vector3f>> maybe_zero_crossings =
      tsdf_zero_crossings_extractor.computeZeroCrossingsFromAboveOnGPU(
          mapper.tsdf_layer());
  ASSERT_TRUE(maybe_zero_crossings.has_value());

  const std::vector<Vector3f> actual_crossings_loc_gpu =
      maybe_zero_crossings.value();

  const int actual_num_crossings_gpu = actual_crossings_loc_gpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_gpu);
}

TEST_F(ZeroCrossingsFromAboveEmptyScene, CPUTest) {
  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);
  scene.generateLayerFromScene(1.0, &tsdf_layer_host);

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);
  mapper.updateColorMesh(UpdateFullLayer::kYes);

  TsdfZeroCrossingsExtractorCPU tsdf_zero_crossings_extractor_cpu;
  EXPECT_TRUE(
      tsdf_zero_crossings_extractor_cpu.computeZeroCrossingsFromAboveOnCPU(
          mapper.tsdf_layer()));
  const std::vector<Vector3f> actual_crossings_loc_cpu =
      tsdf_zero_crossings_extractor_cpu.getZeroCrossingsHost();

  const int actual_num_crossings_cpu = actual_crossings_loc_cpu.size();
  EXPECT_EQ(expected_num_crossings, actual_num_crossings_cpu);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
