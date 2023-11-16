#include "nvblox/core/internal/cuda/atomic_float.cuh"
#include "nvblox/semantics/image_masker.h"

namespace nvblox {

template <typename ElementType>
__device__ inline void copyToUnmaskedOutput(
    const ElementType* input, const int row_idx, const int col_idx,
    const int cols, const ElementType masked_image_invalid_pixel,
    ElementType* unmasked_output, ElementType* masked_output) {
  image::access(row_idx, col_idx, cols, unmasked_output) =
      image::access(row_idx, col_idx, cols, input);
  image::access(row_idx, col_idx, cols, masked_output) =
      masked_image_invalid_pixel;
}

template <typename ElementType>
__device__ inline void copyToMaskedOutput(
    const ElementType* input, const int row_idx, const int col_idx,
    const int cols, const ElementType unmasked_image_invalid_pixel,
    ElementType* unmasked_output, ElementType* masked_output) {
  image::access(row_idx, col_idx, cols, unmasked_output) =
      unmasked_image_invalid_pixel;
  image::access(row_idx, col_idx, cols, masked_output) =
      image::access(row_idx, col_idx, cols, input);
}

__global__ void splitColorImageKernel(const Color* input, const uint8_t* mask,
                                      const int rows, const int cols,
                                      const Color masked_image_invalid_pixel,
                                      const Color unmasked_image_invalid_pixel,
                                      Color* unmasked_output,
                                      Color* masked_output,
                                      Color* masked_depth_overlay = nullptr) {
  // Each thread does a single pixel
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // Assuming that the mask lies directly on top of the input image
  const bool is_masked = image::access(row_idx, col_idx, cols, mask);

  // Overlay the mask onto the color image as debug output
  if (masked_depth_overlay) {
    const Color input_color = image::access(row_idx, col_idx, cols, input);
    image::access(row_idx, col_idx, cols, masked_depth_overlay) = Color(
        fmax(input_color.r, is_masked * 255u), input_color.g, input_color.b);
  }

  if (is_masked) {
    copyToMaskedOutput(input, row_idx, col_idx, cols,
                       unmasked_image_invalid_pixel, unmasked_output,
                       masked_output);
  } else {
    copyToUnmaskedOutput(input, row_idx, col_idx, cols,
                         masked_image_invalid_pixel, unmasked_output,
                         masked_output);
  }
}

__global__ void initializeImageKernel(const float value, const int rows,
                                      const int cols, float* image) {
  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }
  image::access(row_idx, col_idx, cols, image) = value;
}

template <unsigned int kPatchSize>
__global__ void getMinimumDepthKernel(const float* depth_input,
                                      const Transform T_CM_CD,
                                      const Camera depth_camera,
                                      const Camera mask_camera,
                                      float* min_depth_image) {
  static_assert((kPatchSize % 2) == 1,
                "Patch size of getMinimumDepthKernel must be odd.\n");

  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= depth_camera.rows()) || (col_idx >= depth_camera.cols())) {
    return;
  }

  // Unproject from the image.
  const Index2D u_CD(col_idx, row_idx);
  const float depth =
      image::access(row_idx, col_idx, depth_camera.cols(), depth_input);
  const Vector3f p_CD = depth_camera.unprojectFromPixelIndices(u_CD, depth);

  // Transform from the depth camera to the mask camera frame
  const Vector3f p_CM = T_CM_CD * p_CD;

  // Project on the mask image
  Eigen::Vector2f u_CM;
  if (mask_camera.project(p_CM, &u_CM)) {
    // Update the minimum depth values inside the image patch
    // TODO: think about replacing the for loops with a more efficient
    //       implementation
    for (int patch_row = 0; patch_row < kPatchSize; patch_row++) {
      for (int patch_col = 0; patch_col < kPatchSize; patch_col++) {
        const int absolute_col = u_CM.x() + patch_col - kPatchSize / 2;
        const int absolute_row = u_CM.y() + patch_row - kPatchSize / 2;

        if ((absolute_row >= 0) && (absolute_row < mask_camera.rows()) &&
            (absolute_col >= 0) && (absolute_col < mask_camera.cols())) {
          // Update the minimal depth values seen from the mask camera
          atomicMinFloat(&image::access(absolute_row, absolute_col,
                                        mask_camera.cols(), min_depth_image),
                         p_CM.z());
        }
      }
    }
  }
}

__global__ void splitDepthImageKernel(
    const float* depth_input, const uint8_t* mask, const Transform T_CM_CD,
    const Camera depth_camera, const Camera mask_camera,
    const float occlusion_threshold_m, const float masked_image_invalid_pixel,
    const float unmasked_image_invalid_pixel, const float* min_depth_image,
    float* unmasked_depth_output, float* masked_depth_output,
    Color* masked_depth_overlay = nullptr) {
  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= depth_camera.rows()) || (col_idx >= depth_camera.cols())) {
    return;
  }
  const float depth =
      image::access(row_idx, col_idx, depth_camera.cols(), depth_input);

  // Initialize the depth overlay image using scaled depth values
  // The depth overlay image can be used for visualization and debugging
  if (masked_depth_overlay) {
    constexpr float max_depth_display_m = 20.f;
    constexpr float scale_factor = 255u / max_depth_display_m;
    const uint8_t scaled_depth = fmin(scale_factor * depth, 255u);
    image::access(row_idx, col_idx, depth_camera.cols(), masked_depth_overlay) =
        Color(scaled_depth, scaled_depth, scaled_depth);
  }

  // If the depth is infinite, the input pixel is not masked
  if (std::isinf(depth)) {
    copyToUnmaskedOutput(depth_input, row_idx, col_idx, depth_camera.cols(),
                         masked_image_invalid_pixel, unmasked_depth_output,
                         masked_depth_output);
    return;
  }

  // Unproject from the image.
  const Index2D u_CD(col_idx, row_idx);
  const Vector3f p_CD = depth_camera.unprojectFromPixelIndices(u_CD, depth);

  // Transform from the depth camera to the mask camera frame
  const Vector3f p_CM = T_CM_CD * p_CD;

  // Project on the mask image
  Eigen::Vector2f u_CM;
  if (mask_camera.project(p_CM, &u_CM)) {
    // A point is considered to be occluded on the mask image only if it lies
    // more than the occlusion threshold behind the point occluding it.
    const bool is_occluded =
        image::access(u_CM.y(), u_CM.x(), mask_camera.cols(), min_depth_image) +
            occlusion_threshold_m <
        p_CM.z();
    const bool is_masked =
        image::access(u_CM.y(), u_CM.x(), mask_camera.cols(), mask);

    // A masked point is only valid if it is not occluded on the mask image.
    if (is_masked && !is_occluded) {
      copyToMaskedOutput(depth_input, row_idx, col_idx, depth_camera.cols(),
                         unmasked_image_invalid_pixel, unmasked_depth_output,
                         masked_depth_output);
      // Overlay the mask onto the depth overlay image
      if (masked_depth_overlay) {
        image::access(row_idx, col_idx, depth_camera.cols(),
                      masked_depth_overlay)
            .r = 255u;
      }
    } else {
      copyToUnmaskedOutput(depth_input, row_idx, col_idx, depth_camera.cols(),
                           masked_image_invalid_pixel, unmasked_depth_output,
                           masked_depth_output);
    }
  } else {
    // If the projection failed, the input pixel is not masked
    copyToUnmaskedOutput(depth_input, row_idx, col_idx, depth_camera.cols(),
                         masked_image_invalid_pixel, unmasked_depth_output,
                         masked_depth_output);
  }
}

inline Color* getOverlayDataPtr(ColorImage* overlay_image) {
  if (overlay_image) {
    return overlay_image->dataPtr();
  } else {
    return nullptr;
  }
}

ImageMasker::ImageMasker()
    : ImageMasker(std::make_shared<CudaStreamOwning>()) {}

ImageMasker::ImageMasker(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

void ImageMasker::splitImageOnGPU(const ColorImage& input,
                                  const MonoImage& mask,
                                  ColorImage* unmasked_output,
                                  ColorImage* masked_output,
                                  ColorImage* masked_color_overlay) {
  allocateOutput(input, unmasked_output, masked_output, masked_color_overlay);
  CHECK((input.rows() == mask.rows()) && (input.cols() == mask.cols()));

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(input.cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
                        input.rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
                        1);
  splitColorImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                          *cuda_stream_>>>(
      input.dataConstPtr(),                      // NOLINT
      mask.dataConstPtr(),                       // NOLINT
      input.rows(),                              // NOLINT
      input.cols(),                              // NOLINT
      color_masked_image_invalid_pixel_,         // NOLINT
      color_unmasked_image_invalid_pixel_,       // NOLINT
      unmasked_output->dataPtr(),                // NOLINT
      masked_output->dataPtr(),                  // NOLINT
      getOverlayDataPtr(masked_color_overlay));  // NOLINT
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

void ImageMasker::splitImageOnGPU(
    const DepthImage& depth_input, const MonoImage& mask,
    const Transform& T_CM_CD, const Camera& depth_camera,
    const Camera& mask_camera, DepthImage* unmasked_depth_output,
    DepthImage* masked_depth_output, ColorImage* masked_depth_overlay) {
  timing::Timer image_masking_timer("image_masker/split_depth_image\n");

  // First check if images and cameras have the same dimensions and are not
  // empty (depth_input is checked in allocateOutput)
  CHECK_GT(mask.rows(), 0);
  CHECK_GT(mask.cols(), 0);
  CHECK((depth_input.rows() == depth_camera.rows()) &&
        (depth_input.cols() == depth_camera.cols()));
  CHECK((mask.rows() == mask_camera.rows()) &&
        (mask.cols() == mask_camera.cols()));

  allocateOutput(depth_input, unmasked_depth_output, masked_depth_output,
                 masked_depth_overlay);
  // Allocate output images if required

  // Kernel call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks_depth(depth_input.cols() / kThreadsPerThreadBlock.x + 1,
                              depth_input.rows() / kThreadsPerThreadBlock.y + 1,
                              1);
  const dim3 num_blocks_mask(mask.cols() / kThreadsPerThreadBlock.x + 1,
                             mask.rows() / kThreadsPerThreadBlock.y + 1, 1);

  // Initialize the minimum depth image
  constexpr float max_value = std::numeric_limits<float>::max();
  DepthImage min_depth_image =
      DepthImage(mask.rows(), mask.cols(), mask.memory_type());
  initializeImageKernel<<<num_blocks_mask, kThreadsPerThreadBlock, 0,
                          *cuda_stream_>>>(
      max_value,                   // NOLINT
      min_depth_image.rows(),      // NOLINT
      min_depth_image.cols(),      // NOLINT
      min_depth_image.dataPtr());  // NOLINT

  // Find the minimal depth values seen from the mask camera
  constexpr uint8_t kPatchSize = 5;
  getMinimumDepthKernel<kPatchSize>
      <<<num_blocks_depth, kThreadsPerThreadBlock, 0, *cuda_stream_>>>(
          depth_input.dataConstPtr(),  // NOLINT
          T_CM_CD,                     // NOLINT
          depth_camera,                // NOLINT
          mask_camera,                 // NOLINT
          min_depth_image.dataPtr());  // NOLINT

  // Split the depth image according to the mask considering occlusion.
  splitDepthImageKernel<<<num_blocks_depth, kThreadsPerThreadBlock, 0,
                          *cuda_stream_>>>(
      depth_input.dataConstPtr(),                // NOLINT
      mask.dataConstPtr(),                       // NOLINT
      T_CM_CD,                                   // NOLINT
      depth_camera,                              // NOLINT
      mask_camera,                               // NOLINT
      occlusion_threshold_m_,                    // NOLINT
      depth_masked_image_invalid_pixel_,         // NOLINT
      depth_unmasked_image_invalid_pixel_,       // NOLINT
      min_depth_image.dataConstPtr(),            // NOLINT
      unmasked_depth_output->dataPtr(),          // NOLINT
      masked_depth_output->dataPtr(),            // NOLINT
      getOverlayDataPtr(masked_depth_overlay));  // NOLINT

  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
  image_masking_timer.Stop();
}

template <typename ImageType>
void ImageMasker::allocateOutput(const ImageType& input,
                                 ImageType* unmasked_output,
                                 ImageType* masked_output,
                                 ColorImage* overlay_output) {
  CHECK_NOTNULL(unmasked_output);
  CHECK_NOTNULL(masked_output);
  CHECK_GT(input.rows(), 0);
  CHECK_GT(input.cols(), 0);

  // Allocate output images if required
  if ((input.rows() != unmasked_output->rows()) ||
      (input.cols() != unmasked_output->cols())) {
    *unmasked_output =
        ImageType(input.rows(), input.cols(), input.memory_type());
  }
  if ((input.rows() != masked_output->rows()) ||
      (input.cols() != masked_output->cols())) {
    *masked_output = ImageType(input.rows(), input.cols(), input.memory_type());
  }
  if (overlay_output) {
    if ((input.rows() != overlay_output->rows()) ||
        (input.cols() != overlay_output->cols())) {
      *overlay_output =
          ColorImage(input.rows(), input.cols(), input.memory_type());
    }
    CHECK((input.rows() == overlay_output->rows()) &&
          (input.cols() == overlay_output->cols()));
  }

  CHECK((input.rows() == unmasked_output->rows()) &&
        (input.cols() == unmasked_output->cols()));
  CHECK((input.rows() == masked_output->rows()) &&
        (input.cols() == masked_output->cols()));
}

float ImageMasker::occlusion_threshold_m() const {
  return occlusion_threshold_m_;
}

void ImageMasker::occlusion_threshold_m(float occlusion_threshold_m) {
  occlusion_threshold_m_ = occlusion_threshold_m;
}

float ImageMasker::depth_masked_image_invalid_pixel() const {
  return depth_masked_image_invalid_pixel_;
}

void ImageMasker::depth_masked_image_invalid_pixel(float value) {
  depth_masked_image_invalid_pixel_ = value;
}

float ImageMasker::depth_unmasked_image_invalid_pixel() const {
  return depth_unmasked_image_invalid_pixel_;
}

void ImageMasker::depth_unmasked_image_invalid_pixel(float value) {
  depth_unmasked_image_invalid_pixel_ = value;
}

parameters::ParameterTreeNode ImageMasker::getParameterTree(
    const std::string& name_remap) const {
  // NOTE(alexmillane): I'm omitting the invalid pixel values
  const std::string name = (name_remap.empty()) ? "image_masker" : name_remap;
  return parameters::ParameterTreeNode(
      name, {
                parameters::ParameterTreeNode("occlusion_threshold_m:",
                                              occlusion_threshold_m_),
            });
}

}  // namespace nvblox
