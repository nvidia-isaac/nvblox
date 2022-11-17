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
#include <gflags/gflags.h>

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
      (char*)"--tsdf_frame_subsampling=7",
      (char*)"--color_frame_subsampling=8",
      (char*)"--mesh_frame_subsampling=9",
      (char*)"--esdf_frame_subsampling=10",
      (char*)"--tsdf_integrator_max_integration_distance_m=11.0",
      (char*)"--tsdf_integrator_truncation_distance_vox=12.0",
      (char*)"--tsdf_integrator_max_weight=13.0",
      (char*)"--mesh_integrator_min_weight=14.0",
      (char*)"--mesh_integrator_weld_vertices=false",
      (char*)"--color_integrator_max_integration_distance_m=15.0",
      (char*)"--esdf_integrator_min_weight=16.0",
      (char*)"--esdf_integrator_max_site_distance_vox=17.0",
      (char*)"--esdf_integrator_max_distance_m=18.0",
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
  CHECK_NEAR(fuser->voxel_size_m_, 1.0f, kEps);
  CHECK_NEAR(fuser->mapper().tsdf_layer().voxel_size(), 1.0f, kEps);

  // Dataset
  CHECK_EQ(fuser->num_frames_to_integrate_, 2);

  // Output paths
  CHECK_EQ(fuser->timing_output_path_, "3");
  CHECK_EQ(fuser->esdf_output_path_, "4");
  CHECK_EQ(fuser->mesh_output_path_, "5");
  CHECK_EQ(fuser->map_output_path_, "6");

  // Subsampling
  CHECK_EQ(fuser->tsdf_frame_subsampling_, 7);
  CHECK_EQ(fuser->color_frame_subsampling_, 8);
  CHECK_EQ(fuser->mesh_frame_subsampling_, 9);
  CHECK_EQ(fuser->esdf_frame_subsampling_, 10);

  // TSDF integrator
  CHECK_NEAR(fuser->mapper().tsdf_integrator().max_integration_distance_m(),
             11.0f, kEps);
  CHECK_NEAR(fuser->mapper().tsdf_integrator().truncation_distance_vox(), 12.0f,
             kEps);
  CHECK_NEAR(fuser->mapper().tsdf_integrator().max_weight(), 13.0f, kEps);

  // Mesh integrator
  CHECK_NEAR(fuser->mapper().mesh_integrator().min_weight(), 14.0f, kEps);
  CHECK_EQ(fuser->mapper().mesh_integrator().weld_vertices(), false);

  // Color integrator
  CHECK_NEAR(fuser->mapper().color_integrator().max_integration_distance_m(),
             15.0f, kEps);

  // ESDF integrator
  CHECK_NEAR(fuser->mapper().esdf_integrator().min_weight(), 16.0f, kEps);
  CHECK_NEAR(fuser->mapper().esdf_integrator().max_site_distance_vox(), 17.0f,
             kEps);
  CHECK_NEAR(fuser->mapper().esdf_integrator().max_distance_m(), 18.0f, kEps);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}