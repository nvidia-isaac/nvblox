"""
Copyright (c) 2019-2024x, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)
load("@rules_cc//cc:defs.bzl", "cc_common")

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = ctx.attr.cxx_compiler,
        ),
        tool_path(
            name = "ld",
            path = ctx.attr.cxx_ld,
        ),
        tool_path(
            name = "ar",
            path = ctx.attr.ar,
        ),
        tool_path(
            name = "cpp",
            path = ctx.attr.cxx_compiler,
        ),
        tool_path(
            name = "gcov",
            path = ctx.attr.gcov,
        ),
        tool_path(
            name = "nm",
            path = ctx.attr.nm,
        ),
        tool_path(
            name = "objdump",
            path = ctx.attr.objdump,
        ),
        tool_path(
            name = "strip",
            path = ctx.attr.strip,
        ),
    ]

    cpp17_feature = feature(
        name = "c++17",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++17"])],
            ),
        ],
    )

    disable_cxx11_abi_feature = feature(
        name = "c++11_dual_abi",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"])],
            ),
        ],
    )

    cxx_compile_opts_feature = feature(
        name = "c++_compile_opts",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.cxx_compile_opts,
                    ),
                ],
            ),
        ],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Og",
                            "-ggdb3",
                            "-g",
                            "--debug",
                            "--debug-info",
                            "--device-debug",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    fastbuild_feature = feature(
        name = "fastbuild",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-O2",
                            "-g2",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-O3",
                            "-ggdb3",
                            "-DNDEBUG",
                        ],
                    ),
                ],
            ),
        ],
        implies = [],
    )

    linker_opts_feature = feature(
        name = "linker_opts",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.cxx_link_opts,
                    ),
                ],
            ),
        ],
    )

    features = [
        cpp17_feature,
        cxx_compile_opts_feature,
        opt_feature,
        dbg_feature,
        fastbuild_feature,
        linker_opts_feature,
    ]

    if ctx.attr.disable_cxx11_abi_feature:
        features.append(disable_cxx11_abi_feature)

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "gcc-toolchain",
        compiler = "nvcc-10.0-gcc-7.3.0",
        abi_version = "gcc-7.3.0",
        abi_libc_version = "glibc-2.19",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
        features = features,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cxx_builtin_include_directories": attr.string_list(default = []),
        "cxx_compile_opts": attr.string_list(default = []),
        "cxx_link_opts": attr.string_list(default = []),
        "cxx_compiler": attr.string(),
        "cxx_ld": attr.string(default = "/usr/bin/ld"),
        "ar": attr.string(default = "/usr/bin/ar"),
        "gcov": attr.string(default = "/usr/bin/gcov"),
        "nm": attr.string(default = "/usr/bin/nm"),
        "objdump": attr.string(default = "/usr/bin/objdump"),
        "strip": attr.string(default = "/usr/bin/strip"),
        "toolchain_identifier": attr.string(),
        "disable_cxx11_abi_feature": attr.bool(default = False),
    },
    provides = [CcToolchainConfigInfo],
)
