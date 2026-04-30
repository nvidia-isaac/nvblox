"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# Generate config.h from config.h.in
# Use the cuda backend
genrule(
    name = "stdgpu_config",
    srcs = ["src/stdgpu/config.h.in"],
    outs = ["src/stdgpu/config.h"],
    cmd = """
cp $< $@
sed -i 's/@stdgpu_VERSION_MAJOR@/1/g' $@
sed -i 's/@stdgpu_VERSION_MINOR@/3/g' $@
sed -i 's/@stdgpu_VERSION_PATCH@/0/g' $@
sed -i 's/@stdgpu_VERSION@/1.3.0/g' $@
sed -i 's/@STDGPU_BACKEND@/STDGPU_BACKEND_CUDA/g' $@
sed -i 's/@STDGPU_BACKEND_DIRECTORY@/cuda/g' $@
sed -i 's/@STDGPU_BACKEND_NAMESPACE@/cuda/g' $@
sed -i 's/@STDGPU_BACKEND_MACRO_NAMESPACE@/CUDA/g' $@

sed -i 's/#cmakedefine01 STDGPU_ENABLE_CONTRACT_CHECKS/#define STDGPU_ENABLE_CONTRACT_CHECKS 0/g' $@
sed -i 's/#cmakedefine01 STDGPU_USE_32_BIT_INDEX/#define STDGPU_USE_32_BIT_INDEX 0/g' $@
sed -i 's/#cmakedefine01 STDGPU_USE_FAST_DESTROY/#define STDGPU_USE_FAST_DESTROY 0/g' $@
sed -i 's/#cmakedefine01 STDGPU_USE_FIBONACCI_HASHING/#define STDGPU_USE_FIBONACCI_HASHING 0/g' $@
sed -i 's/#cmakedefine01 STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING/#define STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING 0/g' $@
sed -i 's/#cmakedefine01 STDGPU_ENABLE_MANAGED_ARRAY_WARNING/#define STDGPU_ENABLE_MANAGED_ARRAY_WARNING 0/g' $@
""",
)

cuda_library(
    name = "stdgpu",
    srcs = glob([
        "src/stdgpu/**/*.cpp",
    ],
    exclude = [
        "src/stdgpu/hip/**/*.cpp",
    ]),
    hdrs = [":stdgpu_config"] + glob([
        "src/stdgpu/**/*.h",
        "src/stdgpu/**/*.cuh",
    ],
    exclude = [
        "src/stdgpu/hip/**/*.h",
        "src/stdgpu/hip/**/*.cuh",
    ]),
    copts = [
        # Allow sharing constexpr between host and device code
        "--expt-relaxed-constexpr",
        # Allow host, device annotations in lambda declarations
        "--extended-lambda",
        # Suppress all warning messages
        "--disable-warnings",
        "--diag-suppress=20012",
        "--compiler-options=-fPIC",
    ],
    host_defines = [
        # For some reason the redefinitions of limits.h
        # are not picking up macros from climits
        "CHAR_BIT=8",
        "CHAR_MAX=127",
        "CHAR_MIN=-128",
        "INT_MAX=2147483647",
        "INT_MIN=-2147483648",
        "LONG_MAX=2147483647",
        "LONG_MIN=-2147483648",
        "SCHAR_MAX=127",
        "SCHAR_MIN=-128",
        "SHRT_MAX=32767",
        "SHRT_MIN=-32768",
        "UCHAR_MAX=255",
        "ULONG_MAX=4294967295",
        "USHRT_MAX=65535",
        "UINT_MAX=4294967295",
        # And this one because its not picking this up from
        # LLVM/Clang
        "PATH_MAX=4096",
    ],
    includes = [
        "src",
        "src/stdgpu",
    ],
    deps = [
        "@cuda//:thrust",
    ],
    visibility = ["//visibility:public"],
)
