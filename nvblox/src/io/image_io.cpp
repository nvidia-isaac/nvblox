#include "nvblox/io/image_io.h"

#include <memory>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "nvblox/io/internal/thirdparty/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "nvblox/io/internal/thirdparty/stb_image.h"

namespace nvblox {
namespace io {
namespace internal {

template <typename ImageType>
struct NumChannels;

template <>
struct NumChannels<DepthImage> {
  static const int value = 1;
};

template <>
struct NumChannels<MonoImage> {
  static const int value = 1;
};

template <>
struct NumChannels<ColorImage> {
  static const int value = 4;
};

}  // namespace internal

template <typename ImageType>
bool writeToPngTemplate(const std::string& filepath, const ImageType& frame) {
  // Transfer GPU -> Host (if required)
  const ImageType* frame_host_ptr;
  ImageType tmp(MemoryType::kHost);
  if (frame.memory_type() != MemoryType::kHost) {
    tmp = ImageType(MemoryType::kHost);
    tmp.copyFrom(frame);
    frame_host_ptr = &tmp;
  } else {
    frame_host_ptr = &frame;
  }
  // Channels
  const int channels = internal::NumChannels<ImageType>::value;
  // Write using stb
  const int ret = stbi_write_png(filepath.c_str(), frame_host_ptr->cols(),
                                 frame_host_ptr->rows(), channels,
                                 frame_host_ptr->dataConstPtr(),
                                 frame_host_ptr->cols() * channels);
  if (ret == 0) {
    LOG(WARNING) << "Writing image failed";
  }
  return static_cast<bool>(ret);
}

bool writeToPng(const std::string& filepath, const MonoImage& frame) {
  return writeToPngTemplate(filepath, frame);
}

bool writeToPng(const std::string& filepath, const ColorImage& frame) {
  return writeToPngTemplate(filepath, frame);
}

bool writeToPng(const std::string& filepath, const DepthImage& frame) {
  // Make a modifyable copy, and (possibly) transfer CPU -> GPU.
  auto depth_image_gpu_ptr = std::make_unique<DepthImage>(MemoryType::kDevice);
  depth_image_gpu_ptr->copyFrom(frame);

  // Scale the image 0-255 uint8_t
  float max_value = image::maxGPU(*depth_image_gpu_ptr);
  const float scale_factor = std::numeric_limits<uint8_t>::max() / max_value;
  image::elementWiseMultiplicationInPlaceGPU(scale_factor,
                                             depth_image_gpu_ptr.get());
  MonoImage image_out(MemoryType::kHost);
  image::castGPU(*depth_image_gpu_ptr, &image_out);

  // Write as mono image
  return writeToPng(filepath, image_out);
}

template <typename ImageType>
bool readFromUint8PngTemplate(const std::string& filepath,
                              ImageType* frame_ptr) {
  CHECK_NOTNULL(frame_ptr);
  int width;
  int height;
  int num_channels;
  uint8_t* image_data =
      stbi_load(filepath.c_str(), &width, &height, &num_channels, 0);
  if (image_data == nullptr) {
    LOG(WARNING) << "Failure to load image at: " << filepath
                 << "\nAre you sure the file exists?";
    return false;
  }
  if (num_channels != internal::NumChannels<ImageType>::value) {
    LOG(WARNING)
        << "The image you tried to load had the wrong number of channels.";
    // TODO(alex): We could try forcing the channels. This is an available input
    // to stbi, but I can't be bothered working through the implications of that
    // right now. For now we just fail.
    stbi_image_free(image_data);
    return false;
  }
  // Convert to a MonoImage
  frame_ptr->copyFrom(
      height, width,
      reinterpret_cast<typename ImageType::ElementType*>(image_data));
  // Free memory used by stbi
  stbi_image_free(image_data);
  return true;
}

bool readFromPng(const std::string& filepath, DepthImage* frame_ptr,
                 const float scale_factor) {
  CHECK_NOTNULL(frame_ptr);
  int width;
  int height;
  int num_channels;
  uint16_t* image_data =
      stbi_load_16(filepath.c_str(), &width, &height, &num_channels, 0);

  if (image_data == nullptr) {
    return false;
  }
  if (num_channels != internal::NumChannels<DepthImage>::value) {
    LOG(WARNING)
        << "The image you tried to load had the wrong number of channels.";
    // TODO(alex): We could try forcing the channels. This is an available input
    // to stbi, but I can't be bothered working through the implications of that
    // right now. For now we just fail.
    stbi_image_free(image_data);
    return false;
  }

  // The image is expressed in mm. We need to divide by 1000 and convert to
  // float.
  // TODO(alexmillane): It's likely better to do this on the GPU.
  //                    Follow up: I measured and this scaling + cast takes
  //                    ~1ms. So only do this when 1ms is relevant.
  std::vector<float> float_image_data(height * width);
  for (size_t lin_idx = 0; lin_idx < float_image_data.size(); lin_idx++) {
    float_image_data[lin_idx] =
        static_cast<float>(image_data[lin_idx]) * scale_factor;
  }

  frame_ptr->copyFrom(height, width, float_image_data.data());

  stbi_image_free(image_data);
  return true;
}

bool readFromPng(const std::string& filepath, ColorImage* frame_ptr) {
  return readFromUint8PngTemplate(filepath, frame_ptr);
}

bool readFromPng(const std::string& filepath, MonoImage* frame_ptr) {
  return readFromUint8PngTemplate(filepath, frame_ptr);
}

}  // namespace io

}  // namespace nvblox
