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
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <filesystem>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/lidarply_loader.h"
#include "nvblox/fuser/fuser.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/semantics/image_projector.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"
#include "nvblox/tests/utils.h"
#include "nvblox/utils/cuda_kernel_utils.h"
#include "nvblox/utils/timing.h"

using namespace nvblox;

constexpr float kTolerance = 1e-4;

bool isPointValid(const Vector3f& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
}

// GPU kernel to compute squared closest point distances
// For each point in points_a, finds the closest point in points_b
// and stores the squared distance
__global__ void computeSquaredClosestPointDistancesKernel(
    const Vector3f* points_a, const int num_points_a, const Vector3f* points_b,
    const int num_points_b, const float max_range,
    float* squared_distances_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points_a) {
    return;
  }

  const Vector3f point_a = points_a[idx];

  // Skip points beyond max_range (mark with 0.0f so they don't contribute to
  // sum)
  if (point_a.norm() > max_range) {
    squared_distances_out[idx] = 0.0f;
    return;
  }

  // Find the closest point in pointcloud_b
  float min_squared_distance = std::numeric_limits<float>::max();
  for (int j = 0; j < num_points_b; j++) {
    const Vector3f point_b = points_b[j];
    const float squared_distance = (point_a - point_b).squaredNorm();
    min_squared_distance = std::min(min_squared_distance, squared_distance);
  }
  NVBLOX_CHECK(min_squared_distance < std::numeric_limits<float>::max(),
               "Invalid squared distance");

  squared_distances_out[idx] = min_squared_distance;
}

// Helper function to calculate RMSE closest point distance between two
// pointclouds.
// For each point in pointcloud_a, finds the closest point in pointcloud_b
// and returns the RMSE distance.
float getRMSEClosestPointDistance(const Pointcloud& pointcloud_a,
                                  const Pointcloud& pointcloud_b,
                                  float max_range = 15.0f) {
  CHECK(pointcloud_a.memory_type() == MemoryType::kDevice &&
        pointcloud_b.memory_type() == MemoryType::kDevice);
  CHECK(!pointcloud_a.empty() && !pointcloud_b.empty());

  const int num_points_a = pointcloud_a.size();
  const int num_points_b = pointcloud_b.size();

  // Allocate device memory for squared distances
  unified_vector<float> squared_distances(num_points_a, MemoryType::kDevice);

  // Launch kernel to compute squared distances
  constexpr int kThreadsPerBlock = 256;
  int num_blocks = divideRoundUp(num_points_a, kThreadsPerBlock);
  computeSquaredClosestPointDistancesKernel<<<num_blocks, kThreadsPerBlock, 0,
                                              CudaStreamOwning()>>>(
      pointcloud_a.pointsConstPtr(), num_points_a,
      pointcloud_b.pointsConstPtr(), num_points_b, max_range,
      squared_distances.data());
  checkCudaErrors(cudaPeekAtLastError());

  // Use Thrust to sum the squared distances on the device.
  // Use thrust::device execution policy with raw pointers to avoid a
  // stdgpu::device_ptr<void> iterator_traits incompatibility with thrust.
  float total_squared_distance =
      thrust::reduce(thrust::device, squared_distances.data(),
                     squared_distances.data() + num_points_a, 0.0f);

  return std::sqrt(total_squared_distance / num_points_a);
}

class LidarMotionCompensationTest : public ::testing::Test {
 protected:
  std::string data_base_path_ = test_utils::getTestDataPath("data/lidarply/");
  std::string test_output_dir_ = "./data/test_output_data/";
};

std::unique_ptr<LidarFuser> getFuser(const std::string& data_base_path,
                                     int seq_id, bool use_motion_compensation) {
  // Create the fuser with the correct motion compensation setting.
  std::unique_ptr<LidarFuser> fuser =
      datasets::lidarply::createFuser(data_base_path, seq_id);
  fuser->use_lidar_motion_compensation_ = use_motion_compensation;
  return fuser;
}

TEST_F(LidarMotionCompensationTest, MotionCompensationComparison) {
  constexpr int kSeqID = 1;

  // Create a fuser that does not apply motion compensation.
  bool enable_motion_compensation = false;
  std::unique_ptr<LidarFuser> distorted_fuser =
      getFuser(data_base_path_, kSeqID, enable_motion_compensation);

  // Create a fuser that applies motion compensation to pointclouds.
  enable_motion_compensation = true;
  std::unique_ptr<LidarFuser> compensated_fuser =
      getFuser(data_base_path_, kSeqID, enable_motion_compensation);

  // Vectors to store the output pointclouds.
  std::vector<Pointcloud> distorted_pointclouds_L;
  std::vector<Pointcloud> compensated_pointclouds_L;

  // Warm up the device for better timing accuracy.
  warmupCuda();

  int num_frames = 2;
  for (int i = 0; i < num_frames; i++) {
    // Integrate the frame without motion compensation.
    timing::Timing::Reset();
    distorted_fuser->integrateFrame(i);
    std::cout << "Timing WITHOUT motion compensation:" << std::endl;
    std::cout << timing::Timing::Print() << std::endl;

    // Access the distorted data.
    const DepthImage& distorted_depth_image =
        distorted_fuser->multi_mapper_->getLastDepthFrameFromPointcloud();
    Lidar distorted_lidar = *distorted_fuser->getSensor();
    Transform distorted_T_L_C = *distorted_fuser->getSensorPose();

    // Integrate the frame with motion compensation.
    timing::Timing::Reset();
    compensated_fuser->integrateFrame(i);
    std::cout << "Timing WITH motion compensation:" << std::endl;
    std::cout << timing::Timing::Print() << std::endl;

    // Access the compensated data.
    const DepthImage& compensated_depth_image =
        compensated_fuser->multi_mapper_->getLastDepthFrameFromPointcloud();
    Lidar compensated_lidar = *compensated_fuser->getSensor();
    Transform compensated_T_L_C = *compensated_fuser->getSensorPose();

    // Only the actual depth data should be different.
    EXPECT_EQ(compensated_lidar, distorted_lidar);
    EXPECT_TRUE(compensated_T_L_C.isApprox(distorted_T_L_C, kTolerance));
    EXPECT_EQ(distorted_depth_image.rows(), compensated_depth_image.rows());
    EXPECT_EQ(distorted_depth_image.cols(), compensated_depth_image.cols());
    EXPECT_EQ(distorted_depth_image.memory_type(),
              compensated_depth_image.memory_type());

    // Back project the depth images to pointclouds
    DepthImageBackProjector image_back_projector;
    Pointcloud distorted_pointcloud_C(MemoryType::kDevice);
    Pointcloud compensated_pointcloud_C(MemoryType::kDevice);
    image_back_projector.backProjectOnGPU(
        distorted_depth_image, distorted_lidar, &distorted_pointcloud_C);
    image_back_projector.backProjectOnGPU(
        compensated_depth_image, compensated_lidar, &compensated_pointcloud_C);
    EXPECT_GT(distorted_pointcloud_C.size(), 0);
    EXPECT_GT(compensated_pointcloud_C.size(), 0);

    // Transform both pointclouds to the global frame
    distorted_pointclouds_L.push_back(Pointcloud(MemoryType::kDevice));
    compensated_pointclouds_L.push_back(Pointcloud(MemoryType::kDevice));
    transformPointcloudOnGPU(distorted_T_L_C, distorted_pointcloud_C,
                             &distorted_pointclouds_L[i]);
    transformPointcloudOnGPU(compensated_T_L_C, compensated_pointcloud_C,
                             &compensated_pointclouds_L[i]);

    // Visualization of distorted/compensated pointclouds and depth images.
    // Useful for visual inspection of the motion compensation results.
    // Example command to visualize the results:
    //     python -m nvblox_visualize_pointcloud
    //         <TEST_OUTPUT_DIR>/compensated_pointcloud_0.ply
    //         <TEST_OUTPUT_DIR>/compensated_pointcloud_1.ply
    //         --max-range 15.0
    // When comparing this with the visualization of the original pointclouds,
    // the compensated pointclouds should have a better alignment.
    if (FLAGS_nvblox_test_file_output) {
      // Create the test directory if it doesn't exist
      std::filesystem::create_directories(test_output_dir_);

      // Store the depth images for visual inspection
      std::string distorted_depth_image_path = test_output_dir_ +
                                               "distorted_depth_image_" +
                                               std::to_string(i) + ".png";
      std::string compensated_depth_image_path = test_output_dir_ +
                                                 "compensated_depth_image_" +
                                                 std::to_string(i) + ".png";
      io::writeToPng(distorted_depth_image_path, distorted_depth_image);
      io::writeToPng(compensated_depth_image_path, compensated_depth_image);

      // Store the pointclouds for visual inspection
      std::string distorted_pointcloud_path = test_output_dir_ +
                                              "distorted_pointcloud_" +
                                              std::to_string(i) + ".ply";
      std::string compensated_pointcloud_path = test_output_dir_ +
                                                "compensated_pointcloud_" +
                                                std::to_string(i) + ".ply";
      io::outputPointcloudToPly(distorted_pointclouds_L[i],
                                distorted_pointcloud_path);
      io::outputPointcloudToPly(compensated_pointclouds_L[i],
                                compensated_pointcloud_path);
    }

    // Synchronize the device to make sure the results are consistent.
    cudaDeviceSynchronize();
  }

  // Don't evaluate points beyond this range.
  constexpr float kMaxRange = 15.0f;

  for (int n = 0; n < num_frames - 1; n++) {
    // Calculate the RMSE closest point distance between
    // consecutive distorted and compensated pointclouds using GPU acceleration.
    float rmse_distorted = getRMSEClosestPointDistance(
        distorted_pointclouds_L[n], distorted_pointclouds_L[n + 1], kMaxRange);
    float rmse_compensated = getRMSEClosestPointDistance(
        compensated_pointclouds_L[n], compensated_pointclouds_L[n + 1],
        kMaxRange);
    LOG(INFO) << "RMSE distorted: " << rmse_distorted
              << " m, RMSE compensated: " << rmse_compensated << " m"
              << " (max range: " << kMaxRange << " m)";

    // Calculate improvement
    float improvement_m = rmse_distorted - rmse_compensated;
    float improvement_pct = 100.0f * improvement_m / rmse_distorted;
    LOG(INFO) << "Improvement: " << improvement_m << " m (" << improvement_pct
              << "%)";

    // The compensated pointclouds should have a lower RMSE than the distorted
    // pointclouds.
    EXPECT_LT(rmse_compensated, rmse_distorted)
        << "Motion compensation should reduce RMSE. Frame pair " << n << " to "
        << (n + 1);

    // Expect at least 5% improvement
    constexpr float kMinImprovementPct = 4.0f;
    EXPECT_GT(improvement_pct, kMinImprovementPct)
        << "Motion compensation should provide at least 5% improvement. "
        << "Actual: " << improvement_pct << "%";
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
