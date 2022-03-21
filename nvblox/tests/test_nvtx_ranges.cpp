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

#include "nvblox/core/unified_vector.h"
#include "nvblox/utils/nvtx_ranges.h"
#include "nvblox/utils/timing.h"

#include "nvblox/tests/test_utils_cuda.h"

using namespace nvblox;

// NOTE(alexmillane): This file doesn't really test per-se NvtxRange. The intent
// is that this test generates nsight system trace for inspection.

TEST(NvtxTest, Ranges) {
  timing::NvtxRange whole_test_nvtx_range("whole_test", Color::Orange());

  constexpr int kSize = 1e6;
  constexpr int kInitialValue = 0;
  unified_vector<int> vec(kSize, kInitialValue);

  constexpr int kNumAdds = 10;
  for (int i = 0; i < kNumAdds; i++) {
    timing::NvtxRange nvtx_range("add one idx: " + std::to_string(i),
                                 Color::Red());
    test_utils::addOneToAllGPU(&vec);
  }

  timing::NvtxRange nvtx_range("check", Color::Green());
  EXPECT_TRUE(
      test_utils::checkVectorAllConstant(vec, kInitialValue + kNumAdds));
  nvtx_range.Stop();
}

TEST(NvtxTest, TimerNvtx) {
  constexpr int kSize = 1e6;
  constexpr int kInitialValue = 0;
  unified_vector<int> vec(kSize, kInitialValue);

  constexpr int kNumAdds = 10;
  for (int i = 0; i < kNumAdds; i++) {
    timing::mark("add:" + std::to_string(i), Color::Red());
    timing::TimerNvtx timer("add_one");
    test_utils::addOneToAllGPU(&vec);
  }

  timing::TimerNvtx timer("check", Color::Green());
  EXPECT_TRUE(
      test_utils::checkVectorAllConstant(vec, kInitialValue + kNumAdds));
  timer.Stop();

  std::cout << timing::Timing::Print() << std::endl;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
