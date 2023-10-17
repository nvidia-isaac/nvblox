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
#include <gtest/gtest.h>

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/executables/fuser.h"

using namespace nvblox;

TEST(FuserTest, CommandLineFlags) {
  // Fake flags
  char* argv[] = {
      (char*)"CommandLineFlags_test",
      (char*)"--voxel_size=1.0",
      (char*)"--num_frames=2",
      (char*)"--timing_output_path=3",
      (char*)"--esdf_output_path=4",
      (char*)"--mesh_output_path=5",
      (char*)"--map_output_path=6",
      (char*)"--projective_frame_subsampling=7",
      (char*)"--color_frame_subsampling=8",
      (char*)"--mesh_frame_subsampling=9",
      (char*)"--esdf_frame_subsampling=10",
      (char*)"--projective_integrator_max_integration_distance_m=11.0",
      (char*)"--projective_integrator_truncation_distance_vox=12.0",
      (char*)"--projective_integrator_max_weight=13.0",
      (char*)"--free_region_occupancy_probability=0.2",
      (char*)"--occupied_region_occupancy_probability=0.8",
      (char*)"--unobserved_region_occupancy_probability=0.6",
      (char*)"--occupied_region_half_width_m=1.0",
      (char*)"--mesh_integrator_min_weight=14.0",
      (char*)"--mesh_integrator_weld_vertices=false",
      (char*)"--esdf_integrator_min_weight=16.0",
      (char*)"--esdf_integrator_max_site_distance_vox=17.0",
      (char*)"--esdf_integrator_max_distance_m=18.0",
      (char*)"--weighting_scheme_constant_dropoff=true",
      NULL,
  };
  int argc = (sizeof(argv) / sizeof(*(argv))) - 1;
  char** argv_ptr = argv;
  gflags::ParseCommandLineFlags(&argc, &argv_ptr, true);

  std::unique_ptr<Fuser> fuser =
      datasets::threedmatch::createFuser("./data/3dmatch", 1);

  // Check that the params made it in
  constexpr float kEps = 1.0e-6;

  // Layer params
  EXPECT_NEAR(fuser->voxel_size_m_, 1.0f, kEps);
  EXPECT_NEAR(fuser->static_mapper().tsdf_layer().voxel_size(), 1.0f, kEps);

  // Dataset
  EXPECT_EQ(fuser->num_frames_to_integrate_, 2);

  // Output paths
  EXPECT_EQ(fuser->timing_output_path_, "3");
  EXPECT_EQ(fuser->esdf_output_path_, "4");
  EXPECT_EQ(fuser->mesh_output_path_, "5");
  EXPECT_EQ(fuser->map_output_path_, "6");

  // Subsampling
  EXPECT_EQ(fuser->projective_frame_subsampling_, 7);
  EXPECT_EQ(fuser->color_frame_subsampling_, 8);
  EXPECT_EQ(fuser->mesh_frame_subsampling_, 9);
  EXPECT_EQ(fuser->esdf_frame_subsampling_, 10);

  // Projective integrator
  EXPECT_NEAR(fuser->static_mapper().tsdf_integrator().max_weight(), 13.0f,
              kEps);
  EXPECT_NEAR(fuser->static_mapper().lidar_tsdf_integrator().max_weight(),
              13.0f, kEps);
  EXPECT_NEAR(fuser->static_mapper().color_integrator().max_weight(), 13.0f,
              kEps);
  EXPECT_NEAR(
      fuser->static_mapper().tsdf_integrator().max_integration_distance_m(),
      11.0f, kEps);
  EXPECT_NEAR(fuser->static_mapper()
                  .occupancy_integrator()
                  .max_integration_distance_m(),
              11.0f, kEps);
  EXPECT_NEAR(
      fuser->static_mapper().color_integrator().max_integration_distance_m(),
      11.0f, kEps);
  EXPECT_NEAR(
      fuser->static_mapper().tsdf_integrator().truncation_distance_vox(), 12.0f,
      kEps);
  EXPECT_NEAR(
      fuser->static_mapper().occupancy_integrator().truncation_distance_vox(),
      12.0f, kEps);
  EXPECT_NEAR(fuser->static_mapper()
                  .occupancy_integrator()
                  .free_region_occupancy_probability(),
              0.2, kEps);
  EXPECT_NEAR(fuser->static_mapper()
                  .occupancy_integrator()
                  .occupied_region_occupancy_probability(),
              0.8, kEps);
  EXPECT_NEAR(fuser->static_mapper()
                  .occupancy_integrator()
                  .unobserved_region_occupancy_probability(),
              0.6, kEps);
  EXPECT_NEAR(fuser->static_mapper()
                  .occupancy_integrator()
                  .occupied_region_half_width_m(),
              1.0, kEps);

  // Mesh integrator
  EXPECT_NEAR(fuser->static_mapper().mesh_integrator().min_weight(), 14.0f,
              kEps);
  EXPECT_EQ(fuser->static_mapper().mesh_integrator().weld_vertices(), false);

  // ESDF integrator
  EXPECT_NEAR(fuser->static_mapper().esdf_integrator().min_weight(), 16.0f,
              kEps);
  EXPECT_NEAR(fuser->static_mapper().esdf_integrator().max_site_distance_vox(),
              17.0f, kEps);
  EXPECT_NEAR(fuser->static_mapper().esdf_integrator().max_esdf_distance_m(),
              18.0f, kEps);

  // Weighting scheme
  EXPECT_EQ(fuser->static_mapper().tsdf_integrator().weighting_function_type(),
            WeightingFunctionType::kConstantDropoffWeight);
  EXPECT_EQ(fuser->static_mapper().color_integrator().weighting_function_type(),
            WeightingFunctionType::kConstantDropoffWeight);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}