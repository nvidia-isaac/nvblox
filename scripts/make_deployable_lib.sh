#!/bin/bash

# Build nvblox and deps using configuation suitable for deployment.
#
# This script should normally be called from make_deployable_artifact.sh but
# also works stand-alne


set -exo pipefail # Make the script more robust

[[ $# != 2 ]] && echo "Usage: $0 nvblox_commit_sha output_dir" && exit 1

NVBLOX_COMMIT_SHA=$1

OUTPUT_DIR=$2
INSTALL_DIR=$OUTPUT_DIR/$(uname -m)
mkdir -p $INSTALL_DIR
INSTALL_DIR=$(realpath $INSTALL_DIR) # Get absolute path

N_CORES=$(nproc)

# If DIR does not exist, URL will be downloaded and extracted
# DIR will be new working directory
function maybe_download_and_extract()
{
    DIR=$1
    URL=$2
    TAR=$(basename $URL)

    if [ ! -d $DIR ]
       then
           wget $URL
           tar -xvf $TAR
    fi
    cd $DIR
}

####################
# SQLITE
####################
(
    cd $INSTALL_DIR
    maybe_download_and_extract sqlite-autoconf-3400100 \
                               https://sqlite.org/2022/sqlite-autoconf-3400100.tar.gz
    CFLAGS=-fPIC ./configure --prefix=$INSTALL_DIR
    make -j $N_CORES install
)

####################
# GLOG
####################
(
    cd $INSTALL_DIR
    maybe_download_and_extract glog-0.4.0 \
                               https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz
    mkdir -p build
    cd build
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DWITH_GFLAGS=OFF -DBUILD_SHARED_LIBS=OFF
    make -j $N_CORES install
)

####################
# GFLAGS
####################
(
    cd $INSTALL_DIR
    maybe_download_and_extract gflags-2.2.2/ \
                               https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz
    mkdir -p build
    cd build
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DGFLAGS_BUILD_STATIC_LIBS=ON
    make -j $N_CORES install
)

####################
# NVBLOX
####################
(
    cd $INSTALL_DIR
    if [ ! -d nvblox ]
       then
           git clone ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox.git
    fi
    cp nvblox/CHANGELOG.md $OUTPUT_DIR
    cd nvblox/nvblox
    git fetch origin
    git checkout $NVBLOX_COMMIT_SHA
    mkdir -p build
    cd build
    cmake .. \
          -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
          -DBUILD_FOR_ALL_ARCHS=TRUE \
          -DBUILD_REDISTRIBUTABLE=TRUE \
          -DSQLITE3_BASE_PATH=$INSTALL_DIR \
          -DGLOG_BASE_PATH=$INSTALL_DIR \
          -DGFLAGS_BASE_PATH=$INSTALL_DIR
    make -j $N_CORES install

)

