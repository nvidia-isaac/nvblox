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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/time.h"

using namespace nvblox;

TEST(TimeTest, TestAll) {
  Time time_0{0};
  Time time_0_duplicate{0};
  Time time_1{1};
  Time time_2{2};

  // Check conversion operator
  EXPECT_EQ(static_cast<int64_t>(time_1), 1);

  // Check overloaded operators
  EXPECT_TRUE(time_0 == time_0_duplicate);  // ==
  EXPECT_TRUE(time_0 < time_2);             // <
  EXPECT_TRUE(time_2 > time_0);             // >
  EXPECT_TRUE(time_0 <= time_2);            // <=
  EXPECT_TRUE(time_0 <= time_0_duplicate);  // <=
  EXPECT_TRUE(time_2 >= time_0);            // >=
  EXPECT_TRUE(time_0 >= time_0_duplicate);  // >=
  time_2 += time_1;
  EXPECT_EQ(static_cast<int64_t>(time_2), 3);  // +=
  time_2 -= time_1;
  EXPECT_EQ(static_cast<int64_t>(time_2), 2);           // -=
  EXPECT_EQ(static_cast<int64_t>(time_1 + time_2), 3);  // +
  EXPECT_EQ(static_cast<int64_t>(time_2 - time_1), 1);  // -
  EXPECT_EQ(static_cast<int64_t>(2 * time_1), 2);       // *
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
