#!/bin/bash

# Build nvblox artifact zip for delivery
#
# - Calls make_deployable_lib.sh for building libs on x86 and optionally for jetson
# - Packages the libs in an archive suitable for deployment
#
# the jetson host must be accissible through ssh without providing any username, e.g. by using
#   ssh hostname
#
# To achive this, it might be necessary to add entries like the following in ~/.ssh/config:
#
#   Host ros-orin-02
#     HostName 10.111.83.64
#     User nvidia
set -exo pipefail

echo $#
if [[ $# < 1 ||  $# > 3 ]]
then
    echo "Usage: $0 nvblox_commit_sha install_dir [jetson_host]"
    exit 1
fi

NVBLOX_COMMIT_SHA=$1
OUTPUT_DIR=$2
JETSON_HOST=$3
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

####################
# Build for x86
####################
$SCRIPT_DIR/make_deployable_lib.sh $NVBLOX_COMMIT_SHA $OUTPUT_DIR


##############################
# Optionally Build for Jetson
##############################
if [ ! -z $JETSON_HOST ]
then
    # Create remote working dir
    REMOTE_TMP_DIR=/tmp/nvblox_build
    ssh $JETSON_HOST "mkdir -p $REMOTE_TMP_DIR"

    # Copy the build script and launch it
    scp $SCRIPT_DIR/make_deployable_lib.sh $JETSON_HOST:$REMOTE_TMP_DIR
    ssh $JETSON_HOST "export PATH=/usr/local/cuda-11.4/bin:$PATH && $REMOTE_TMP_DIR/make_deployable_lib.sh $NVBLOX_COMMIT_SHA $REMOTE_TMP_DIR"

    # Copy results back
    rsync -va $JETSON_HOST:$REMOTE_TMP_DIR/aarch64 $OUTPUT_DIR
 fi

##############################
# Create the artifact archive
##############################
ARTIFACT_DIR=$OUTPUT_DIR/artifact
mkdir -p $ARTIFACT_DIR

# Rename directories and copy them to artifacto output folder
rsync -a $OUTPUT_DIR/x86_64/lib/* $ARTIFACT_DIR/lib_x86_64
rsync -a $OUTPUT_DIR/x86_64/include/* $ARTIFACT_DIR/include
cp $OUTPUT_DIR/x86_64/nvblox/CHANGELOG.md $ARTIFACT_DIR
[[ -d $OUTPUT_DIR/aarch64/lib ]] && cp -r $OUTPUT_DIR/aarch64/lib $ARTIFACT_DIR/lib_aarch64

(
    cd $ARTIFACT_DIR

    # TODO(dtingdah) automatic versioning
    NVBLOX_COMMIT_SHA_SHORT=$(echo $NVBLOX_COMMIT_SHA | head -c8)
    ARTIFACT_FNAME=nvblox_YY.MM.X_${NVBLOX_COMMIT_SHA_SHORT}.zip
    zip -r $ARTIFACT_FNAME *

    # Sanity check
    unzip -l $ARTIFACT_FNAME | grep libnvblox_lib.so
    echo Created artifact: $(realpath $ARTIFACT_FNAME)
)

