#[[
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
]]

include(GoogleTest)
# Add some tests.
set(TEST_OPTIONS
    DISCOVERY_TIMEOUT 30 DISCOVERY_MODE
    PRE_TEST # Avoid running test executables in the build step
)
function(add_nvblox_torch_test test_name)
  add_nvblox_executable(${test_name} SOURCE_FILES ${test_name}.cpp
                        LINK_LIBRARIES_PUBLIC gtest py_nvblox)
  gtest_discover_tests(${test_name} ${TEST_OPTIONS})
endfunction()

add_nvblox_torch_test(test_tensor_conversions)
add_nvblox_torch_test(test_color_tensor)
