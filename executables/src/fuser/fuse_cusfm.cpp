/*
Copyright 2025 NVIDIA CORPORATION

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

// This binary runs Nvblox to reconstruct scene with color and depth images
// with poses. It outputs an 2d occupancy map at a given height.
//
// Usage:
// fuse_cusfm
//  --color_image_dir <color_dir>
//  --depth_image_dir <depth_dir>
//  --frames_meta_file <frames_meta_file>
//  --save_2d_occupancy_map_path <out_dir>/occupancy_map
//
// Recommended parameters for optimal performance:
//  --mapping_type_dynamic
//  --projective_integrator_max_integration_distance_m=2.5
//  --esdf_slice_min_height=0.09
//  --esdf_slice_max_height=0.65
//  --esdf_slice_height=0.3
//  --workspace_bounds_type=1
//  --workspace_bounds_min_height_m=-0.3
//  --workspace_bounds_max_height_m=2.0
//
// Other optional parameters:
//  --mesh_output_path <out_dir>/mesh.ply
//  --nouse_2d_esdf_mode
//  --num_frames 20
//  --fit_to_z0 (align all poses to z=0 plane for horizontal mesh)
//
// One can find an example cuSFM dataset here
// https://github.com/nvidia-isaac/pyCuSFM/tree/main/data/r2b_galileo
// and use this workflow to generate depth and mesh
// https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html#

#include <gflags/gflags.h>

#include "nvblox/datasets/cusfm_data.h"
#include "nvblox/fuser/fuser.h"
#include "nvblox/integrators/esdf_slicer.h"
#include "nvblox/integrators/occupancy_conversions.h"

DECLARE_bool(alsologtostderr);

DEFINE_string(save_2d_occupancy_map_path, "",
              "Required: path to save 2d occupancy map");

DEFINE_double(distance_map_unknown_value_optimistic, 1000.0,
              "The distance inside the distance map for unknown value.");

DEFINE_string(color_image_dir, "",
              "Required: override the default color image location");

DEFINE_string(depth_image_dir, "",
              "Required: override the default depth image location");

DEFINE_string(frames_meta_file, "",
              "Required: override the default frames_meta location");

DEFINE_bool(fit_to_z0, false,
            "Fit all poses to z=0 plane for horizontal mesh output");

namespace nvblox {

bool Save2dOccupancyMap(const std::string& output_occupancy_map_path,
                        CameraFuser& fuser) {
  // Grab an esdf slice and converts to 2d occupancy map
  nvblox::EsdfSlicer esdf_slicer;

  AxisAlignedBoundingBox aabb;
  Image<float> map_slice_image(MemoryType::kDevice);
  esdf_slicer.sliceLayerToDistanceImage(
      fuser.static_mapper()->esdf_layer(),
      fuser.static_mapper()->esdf_integrator().esdf_slice_height(),
      FLAGS_distance_map_unknown_value_optimistic /* unobserved value */, &aabb,
      &map_slice_image);

  const size_t width = map_slice_image.cols();
  const size_t height = map_slice_image.rows();

  if (width == 0 || height == 0) {
    LOG(ERROR) << "No map to save, skipping map save.";
    return false;
  }
  constexpr int8_t kOccupancyGridUnknownValue = -1;
  constexpr float kFreeThreshold = 0.25;
  constexpr float kOccupiedThreshold = 0.65;
  std::vector<int8_t> occupancy_grid;
  occupancy_grid.resize(width * height, kOccupancyGridUnknownValue);
  // Convert from ESDF to occupancy grid
  esdf_slicer.occupancyGridFromSliceImage(
      map_slice_image, occupancy_grid.data(),
      FLAGS_distance_map_unknown_value_optimistic /* unknown value */);
  std::string png_path = output_occupancy_map_path + ".png";
  nvblox::conversions::saveOccupancyGridAsPng(png_path, kFreeThreshold,
                                              kOccupiedThreshold, height, width,
                                              occupancy_grid);
  LOG(INFO) << "Writing occupancy map to: " << png_path;
  const size_t file_name_index = png_path.find_last_of("/\\");
  std::string image_name = png_path.substr(file_name_index + 1);
  std::string yaml_path = output_occupancy_map_path + ".yaml";
  nvblox::conversions::saveOccupancyGridYaml(
      yaml_path, image_name, fuser.static_mapper()->esdf_layer().voxel_size(),
      aabb.min().x(), aabb.min().y(), kFreeThreshold, kOccupiedThreshold);
  LOG(INFO) << "Writing occupancy map yaml meta data to: " << yaml_path;
  return true;
}

}  // namespace nvblox

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  CHECK(!FLAGS_color_image_dir.empty()) << "Please provide --color_image_dir";
  CHECK(!FLAGS_depth_image_dir.empty()) << "Please provide --depth_image_dir";
  CHECK(!FLAGS_frames_meta_file.empty()) << "Please provide --frames_meta_file";
  CHECK(!FLAGS_save_2d_occupancy_map_path.empty())
      << "Please provide --save_2d_occupancy_map_path";

  // Extract output directory from occupancy map path for z0 transform output
  std::string output_dir;
  if (FLAGS_fit_to_z0) {
    size_t last_slash = FLAGS_save_2d_occupancy_map_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
      output_dir = FLAGS_save_2d_occupancy_map_path.substr(0, last_slash);
    } else {
      output_dir = ".";  // Current directory as fallback
    }
    LOG(INFO) << "Z0 transform will be saved to: " << output_dir;
  }

  std::unique_ptr<nvblox::CameraFuser> fuser =
      nvblox::datasets::cusfm_data::createFuser(
          FLAGS_color_image_dir, FLAGS_depth_image_dir, FLAGS_frames_meta_file,
          true /* init from gflags */, FLAGS_fit_to_z0, output_dir);
  if (!fuser) {
    LOG(FATAL) << "Creation of the Fuser failed";
  }

  // Make sure the layers are the correct resolution.
  if (fuser->run() != EXIT_SUCCESS) {
    LOG(ERROR) << "Failed to run fuser";
    return EXIT_FAILURE;
  }

  if (!nvblox::Save2dOccupancyMap(FLAGS_save_2d_occupancy_map_path,
                                  *fuser.get())) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
