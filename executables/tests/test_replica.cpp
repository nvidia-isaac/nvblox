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

#include <iostream>

#include "nvblox/datasets/replica.h"
#include "nvblox/fuser/fuser.h"
#include "nvblox/tests/utils.h"
using namespace nvblox;

constexpr float kTolerance = 1e-4;

class DatasetReplicaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_path_ = test_utils::getTestDataPath("data/replica/office0");
  }

  std::string base_path_;
};

TEST_F(DatasetReplicaTest, RunReplicaFuser) {
  auto fuser = datasets::replica::createFuser(base_path_);
  EXPECT_NE(fuser, nullptr);
  const int result = fuser->run();
  EXPECT_EQ(result, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
