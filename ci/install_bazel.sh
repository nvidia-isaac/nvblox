#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set -euo pipefail

echo "Installing Bazel 8.2.1..."

# Add Bazel repository key
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel-archive-keyring.gpg
mv bazel-archive-keyring.gpg /usr/share/keyrings

# Add Bazel repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

# Install Bazel
apt-get update
apt-get install -y bazel-8.2.1

# symlink
ln -s /usr/bin/bazel-8.2.1 /usr/bin/bazel

# verify installation
echo "Bazel 8.2.1 installed successfully"
bazel --version
