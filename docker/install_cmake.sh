#!/bin/bash
set -e

# Modern nvcc C++20 builds need CMake >= 3.25.2. Keep CMake 3.x while our
# third-party projects still rely on policies removed in CMake 4.
ubuntu_version=$(lsb_release -r | cut -f2)
echo "Detected ubuntu version: $ubuntu_version"

if [ "$ubuntu_version" == "24.04" ]
then
    # Ubuntu 24 supplies CMake 3.28, which supports nvcc's C++20 mode.
    echo "Installing cmake for Ubuntu$ubuntu_version"
    apt-get update && apt-get install -y cmake

elif [ "$ubuntu_version" == "22.04" ]
then
    # Ubuntu 22's apt package is 3.22 and cannot configure CUDA C++20 targets.
    # Isolate CMake from the Torch environment; python3-venv is installed by
    # Dockerfile.deps. PyPI provides CMake binaries for both x86_64 and aarch64.
    echo "Installing CMake 3.x with CUDA C++20 support for Ubuntu$ubuntu_version"
    python3 -m venv /opt/cmake
    /opt/cmake/bin/python3 -m pip install --no-cache-dir 'cmake>=3.25.2,<4'
    for tool in cmake ctest cpack; do
        ln -sf "/opt/cmake/bin/$tool" "/usr/local/bin/$tool"
    done

elif [ "$ubuntu_version" == "20.04" ]
then
    # On Ubuntu 20 we need a custom install
    echo "Installing cmake for Ubuntu20.04"
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update \
    && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
    && apt-get install -y kitware-archive-keyring \
    && apt-get remove -y cmake-data && apt-get purge -y cmake \
    && apt-get install -y cmake=3.22.1-0kitware1ubuntu20.04.1 cmake-data=3.22.1-0kitware1ubuntu20.04.1 \
    && cmake --version \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean
    else
        echo "ERROR. Unsupported Ubuntu version: $ubuntu_version"
    exit 1
fi
