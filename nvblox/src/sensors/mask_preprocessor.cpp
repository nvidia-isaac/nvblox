/*
Copyright 2023 NVIDIA CORPORATION

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

#include <queue>

#include "nvblox/sensors/mask_preprocessor.h"
#include "nvblox/utils/timing.h"
namespace nvblox {
namespace image {

// Mode of exploration for a connected component. Also used as bitmask for
// when indicating visited pixels
enum class ExplorationMode : uint8_t {
  // First bit of mask image is used to indicate visited pixels.
  kDefault = 0x01,
  // Zeroing mask image during traversal. Second bit of mask image is used to
  // indicate visited pixels. Motivation: we need to first run with kDefault and
  // thereafter optionally with kErase. Using different bits for indicating
  // visited allows us to do so without having to reset all the visited bits
  // before the second run.
  kErase = 0x02
};

// Return true if pixel given by (row, col) is visited and false otherwise
inline bool isVisited(const int row, const int col, const ExplorationMode mode,
                      const MonoImage& mask) {
  // Pixel is visited if the bit indicated by "mode" is NOT set.
  return !(mask(row, col) & static_cast<uint8_t>(mode));
}

// Set the pixel given by (row,col) to visited
inline void setVisited(const int row, const int col, const ExplorationMode mode,
                       MonoImage* mask) {
  // Clear the bit given by the "mode" bitmask
  (*mask)(row, col) &= ~(static_cast<uint8_t>(mode));
}

// If the pixel given by (row, col) is 1) within bounds, 2) not already visited
// and 3) has active mask, then:
//  * set pixel to visited
//  * add pixel to queue
inline void maybeVisitPixel(const int row, const int col,
                            const ExplorationMode mode, MonoImage* mask,
                            std::queue<std::array<int, 2>>& queue) {
  if (col < 0 || col >= mask->width() || row < 0 || row >= mask->height()) {
    return;
  } else if ((*mask)(row, col) && !isVisited(row, col, mode, *mask)) {
    queue.push({row, col});
    setVisited(row, col, mode, mask);
  }
}

// Explore a connected component and return its size (number of pixels)
//
// @param start_row  Row where exploration starts
// @param start_col  Column where exploration starts
// @param queue      Used internally to keep track of next pixel to visit. Must
// be empty.
// @param mode       Mode of exploration
// @return  Number of pixels in component
inline int exploreComponent(const int start_row, const int start_col,
                            ExplorationMode mode, MonoImage* mask,
                            std::queue<std::array<int, 2>>& queue) {
  CHECK(queue.empty());
  CHECK(mask->height() > 0);
  CHECK(mask->width() > 0);

  int component_size = 0;

  // Initialize with starting pixel
  queue.push({start_row, start_col});
  setVisited(start_row, start_col, mode, mask);

  while (!queue.empty()) {
    // Pop queue with next pixel to visit
    const int row = queue.front()[0];
    const int col = queue.front()[1];
    queue.pop();

    // Optionally erase the mask as we go
    if (mode == ExplorationMode::kErase) {
      (*mask)(row, col) = 0;
    }

    // Try to visit the neighbouring pixels.
    maybeVisitPixel(row, col - 1, mode, mask, queue);  // West
    maybeVisitPixel(row, col + 1, mode, mask, queue);  // East
    maybeVisitPixel(row - 1, col, mode, mask, queue);  // North
    maybeVisitPixel(row + 1, col, mode, mask, queue);  // South

    ++component_size;
  }

  return component_size;
}

void findConnectedComponets(MonoImage& mask_host, const int size_threshold) {
  std::queue<std::array<int, 2>> queue;

  // Iterate over all pixels until we find an unvisited one where the mask is
  // active. That's the beginning of a new connected component
  timing::Timer explore_component_loop_timer(
      "image/remove_small_connected_components/explore_component_loop");
  for (int row = 0; row < mask_host.rows(); ++row) {
    for (int col = 0; col < mask_host.cols(); ++col) {
      if (mask_host(row, col) &&
          !isVisited(row, col, ExplorationMode::kDefault, mask_host)) {
        // Find the numver of pixels in this component
        int component_size = exploreComponent(
            row, col, ExplorationMode::kDefault, &mask_host, queue);

        // Erase the component if it's too small
        if (component_size < size_threshold) {
          exploreComponent(row, col, ExplorationMode::kErase, &mask_host,
                           queue);
        }
      }
    }
  }
  explore_component_loop_timer.Stop();
}

MaskPreprocessor::MaskPreprocessor(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream),
      npp_stream_context_(getNppStreamContext(*cuda_stream)) {}

void MaskPreprocessor::removeSmallConnectedComponents(const MonoImage& mask_in,
                                                      const int size_threshold,
                                                      MonoImage* mask_out) {
  timing::Timer remove_small_connected_components_timer(
      "image/remove_small_connected_components");

  // Simply copy the output if threshold is zero.
  if (size_threshold <= 0) {
    mask_out->copyFromAsync(mask_in, *cuda_stream_);
    cuda_stream_->synchronize();
    return;
  }

  NVBLOX_CHECK(mask_in.rows() % kDownScaleFactor == 0,
               "Mask rows must be divisible by kDownScaleFactor");
  NVBLOX_CHECK(mask_in.cols() % kDownScaleFactor == 0,
               "Mask cols must be divisible by kDownScaleFactor");

  // Allocate output mask if required,
  CHECK_GT(mask_in.rows(), 0);
  CHECK_GT(mask_in.cols(), 0);
  mask_out->resizeAsync(mask_in.rows(), mask_in.cols(), *cuda_stream_);

  // Downscale to save processing time in the coming steps.
  naiveDownscaleGPUAsync(mask_in, kDownScaleFactor, &mask_downscaled_,
                         *cuda_stream_);

  // Apply threshold to set all active pixels to 255 which is required by
  // the findConnectedComponets().
  constexpr uint8_t kThreshold = 0;
  constexpr uint8_t kSetToValue = 255;
  mask_thresholded_host_.resizeAsync(mask_downscaled_.rows(),
                                     mask_downscaled_.cols(), *cuda_stream_);
  setGreaterThanThresholdToValue(mask_downscaled_, kThreshold, kSetToValue,
                                 npp_stream_context_, &mask_thresholded_host_);
  cuda_stream_->synchronize();

  // Run the connected-component finder on CPU. This turned out to be faster
  // thant NPP's GPU implementation for the 640x480 images we're typically
  // processing.
  const int size_threshold_downscaled =
      size_threshold / (kDownScaleFactor * kDownScaleFactor);
  findConnectedComponets(mask_thresholded_host_, size_threshold_downscaled);

  // Finally, upscale to original resolution.
  image::upscaleGPUAsync(mask_thresholded_host_, kDownScaleFactor, mask_out,
                         *cuda_stream_);
  cuda_stream_->synchronize();
}

}  // namespace image
}  // namespace nvblox
