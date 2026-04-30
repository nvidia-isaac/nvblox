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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "nvblox/io/image_io.h"
#include "nvblox/semantics/mask_from_detections.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr int kNumRows = 480;
constexpr int kNumCols = 640;

class DepthMaskMode;
extern void findDepthModes(host_vector<DepthMaskMode>& depth_and_modes,
                           const CudaStream& cuda_stream);

template <typename T>
void fill(ImageView<T> view, const T value) {
  for (int y = 0; y < view.rows(); ++y) {
    for (int x = 0; x < view.cols(); ++x) {
      view(y, x) = value;
    }
  }
}

bool masksEqual(const MonoImage& m1, const MonoImage& m2) {
  bool is_equal = true;

  is_equal &= (m1.rows() == m2.rows());
  is_equal &= (m1.cols() == m2.cols());

  if (is_equal) {
    for (int y = 0; y < m1.rows(); ++y) {
      for (int x = 0; x < m1.cols(); ++x) {
        is_equal &= (m1(y, x) == m2(y, x));
      }
    }
  }

  return is_equal;
}

void testWithSyntheticDetections(
    const std::vector<ImageBoundingBox>& detection_bboxes,
    const std::vector<ImageBoundingBox>& object_bboxes,
    const std::vector<float>& depths, const float background_depth) {
  ASSERT_EQ(detection_bboxes.size(), object_bboxes.size());
  ASSERT_EQ(detection_bboxes.size(), depths.size());

  // Create depth image with background.
  DepthImage depth_image(kNumRows, kNumCols, MemoryType::kUnified);
  fill<float>(depth_image, background_depth);

  // Create detection mask.
  MonoImage detection_mask(kNumRows, kNumCols, MemoryType::kUnified);
  detection_mask.setZeroAsync(CudaStreamOwning());

  // Create ground-truth segmentation mask.
  MonoImage expected_mask(kNumRows, kNumCols, MemoryType::kUnified);
  expected_mask.setZeroAsync(CudaStreamOwning());

  // Draw depth and masks
  for (size_t i = 0; i < detection_bboxes.size(); ++i) {
    const auto bbox_object = object_bboxes.at(i);
    const auto bbox_detection = detection_bboxes.at(i);

    fill<float>(DepthImageView(depth_image).cropped(bbox_object), depths.at(i));
    fill<uint8_t>(MonoImageView(detection_mask).cropped(bbox_detection), 255);
    fill<uint8_t>(MonoImageView(expected_mask).cropped(bbox_object), 255);
  }

  constexpr float kDistanceThreshold = 0.05;
  MonoImage computed_mask(detection_mask.rows(), detection_mask.cols(),
                          MemoryType::kUnified);
  maskFromDetections(detection_bboxes, depth_image, kDistanceThreshold,
                     &computed_mask, CudaStreamOwning());

  io::writeToPng("detection_mask.png", detection_mask);
  io::writeToPng("computed_mask.png", computed_mask);
  io::writeToPng("expected_mask.png", expected_mask);
  io::writeToPng("depth.png", depth_image);

  EXPECT_TRUE(masksEqual(computed_mask, expected_mask));
}

TEST(MaskFromDetections, SingleDetection) {
  const ImageBoundingBox bbox_detection{Index2D(50, 50), Index2D(250, 250)};
  const ImageBoundingBox bbox_object{Index2D(60, 60), Index2D(240, 240)};
  constexpr float kBackgroundDepth = 9.0f;
  constexpr float kObjectDepth = 2.0f;

  testWithSyntheticDetections({bbox_detection}, {bbox_object}, {kObjectDepth},
                              kBackgroundDepth);
}

ImageBoundingBox createBbox(const float x, const float y, const float size_x,
                            float size_y) {
  return ImageBoundingBox{Index2D(x - size_x, y - size_x),
                          Index2D(x + size_x, y + size_y)};
}

ImageBoundingBox shrink(const ImageBoundingBox& bbox) {
  constexpr float kScale = 0.8;

  const Vector2f center =
      (bbox.max().cast<float>() + bbox.min().cast<float>()) / 2.F;

  const Vector2f scaled_min =
      (bbox.min().cast<float>() - center) * kScale + center;
  const Vector2f scaled_max =
      (bbox.max().cast<float>() - center) * kScale + center;

  ImageBoundingBox scaled(scaled_min.cast<int>(), scaled_max.cast<int>());
  return scaled;
}

TEST(MaskFromDetections, MultipleDetections) {
  constexpr float kBackgroundDepth = 9.0f;

  std::vector<ImageBoundingBox> detections;
  std::vector<ImageBoundingBox> objects;

  detections.push_back(createBbox(100, 100, 10, 20));
  objects.push_back(shrink(detections.back()));

  detections.push_back(createBbox(300, 100, 20, 30));
  objects.push_back(shrink(detections.back()));

  detections.push_back(createBbox(100, 300, 20, 20));
  objects.push_back(shrink(detections.back()));

  detections.push_back(createBbox(300, 300, 25, 25));
  objects.push_back(shrink(detections.back()));

  testWithSyntheticDetections(detections, objects, {2.F, 3.F, 4.F, 5.F},
                              kBackgroundDepth);
}

TEST(MaskFromDetections, RealData) {
  DepthImage depth_image(MemoryType::kHost);
  const std::string base_path =
      test_utils::getTestDataPath("data/human_dataset/");
  ASSERT_TRUE(io::readFromPng(base_path + "depth_image_1.png", &depth_image));

  const ImageBoundingBox detection(Index2D(141, 0), Index2D(262, 310));

  MonoImage mask(depth_image.rows(), depth_image.cols(), MemoryType::kHost);
  constexpr float kDistanceThreshold = 0.1;
  maskFromDetections({detection}, depth_image, kDistanceThreshold, &mask,
                     CudaStreamOwning());
  io::writeToPng("mask.png", mask);
}

TEST(MaskFromDetections, NoDetections) {
  constexpr float kBackgroundDepth = 9.0f;
  testWithSyntheticDetections({}, {}, {}, kBackgroundDepth);

  // Check that the computed mask in the case of no detections is unmasked.
  DepthImage depth(2, 2, MemoryType::kHost);
  MonoImage mask(2, 2, MemoryType::kHost);
  mask(0, 0) = 0;
  mask(0, 1) = 1;
  mask(1, 0) = 2;
  mask(1, 1) = 3;

  constexpr float kDistanceThreshold = 0.1;
  maskFromDetections({}, depth, kDistanceThreshold, &mask, CudaStreamOwning());

  EXPECT_EQ(mask(0, 0), 0);
  EXPECT_EQ(mask(0, 1), 0);
  EXPECT_EQ(mask(1, 0), 0);
  EXPECT_EQ(mask(1, 1), 0);
}

TEST(MaskFromDetections, Clipping) {
  // Bounding box that exceeds image bounds.
  ImageBoundingBox unclipped_bounding_box(Index2D(-1, -1), Index2D(2, 2));
  // Try to clip it in.
  ImageBoundingBox clipped_bounding_box =
      clipToImageBounds(unclipped_bounding_box, 2, 2);

  EXPECT_TRUE(clipped_bounding_box.min() == Index2D(0, 0));
  EXPECT_TRUE(clipped_bounding_box.max() == Index2D(1, 1));
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
