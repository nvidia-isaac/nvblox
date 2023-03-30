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
#include "nvblox/io/csv.h"

#include <fstream>
#include <memory>

namespace nvblox {
namespace io {

void writeToCsv(const std::string& filepath, const DepthImage& frame) {
  LOG(INFO) << "Writing DepthImage to: " << filepath;

  // Create a temporary image in unified memory if the image resides in device
  // memory.
  const DepthImage* write_frame_ptr;
  std::unique_ptr<DepthImage> tmp;
  if (frame.memory_type() == MemoryType::kDevice) {
  tmp = std::make_unique<DepthImage>(frame, MemoryType::kHost);
    write_frame_ptr = tmp.get();
  } else {
    write_frame_ptr = &frame;
  }

  std::ofstream file_stream(filepath, std::ofstream::out);
  for (int row_idx = 0; row_idx < write_frame_ptr->rows(); row_idx++) {
    for (int col_idx = 0; col_idx < write_frame_ptr->cols(); col_idx++) {
      file_stream << (*write_frame_ptr)(row_idx, col_idx) << " ";
    }
    file_stream << "\n";
  }
  file_stream.close();
}

void writeToCsv(const std::string& filepath, const ColorImage& frame) {
  LOG(INFO) << "Writing ColorImage to: " << filepath;

  // Create a temporary image in unified memory if the image resides in device
  // memory.
  const ColorImage* write_frame_ptr;
  std::unique_ptr<ColorImage> tmp;
  if (frame.memory_type() == MemoryType::kDevice) {
    tmp = std::make_unique<ColorImage>(frame, MemoryType::kHost);
    write_frame_ptr = tmp.get();
  } else {
    write_frame_ptr = &frame;
  }

  std::ofstream file_stream(filepath, std::ofstream::out);
  for (int row_idx = 0; row_idx < write_frame_ptr->rows(); row_idx++) {
    for (int col_idx = 0; col_idx < write_frame_ptr->cols(); col_idx++) {
      file_stream << std::to_string((*write_frame_ptr)(row_idx, col_idx).r)
                  << " "
                  << std::to_string((*write_frame_ptr)(row_idx, col_idx).g)
                  << " "
                  << std::to_string((*write_frame_ptr)(row_idx, col_idx).b)
                  << " "
                  << std::to_string((*write_frame_ptr)(row_idx, col_idx).a)
                  << " ";
    }
    file_stream << "\n";
  }
  file_stream.close();
}

void writeToCsv(const std::string& filepath, const MonoImage& frame) {
  LOG(INFO) << "Writing MonoImage to: " << filepath;

  // Create a temporary image in unified memory if the image resides in device
  // memory.
  const MonoImage* write_frame_ptr;
  std::unique_ptr<MonoImage> tmp;
  if (frame.memory_type() == MemoryType::kDevice) {
    tmp = std::make_unique<MonoImage>(frame, MemoryType::kHost);
    write_frame_ptr = tmp.get();
  } else {
    write_frame_ptr = &frame;
  }

  std::ofstream file_stream(filepath, std::ofstream::out);
  for (int row_idx = 0; row_idx < write_frame_ptr->rows(); row_idx++) {
    for (int col_idx = 0; col_idx < write_frame_ptr->cols(); col_idx++) {
      file_stream << static_cast<int>((*write_frame_ptr)(row_idx, col_idx))
                  << " ";
    }
    file_stream << "\n";
  }
  file_stream.close();
}

}  // namespace io
}  // namespace nvblox