#!/bin/bash
set -ex


while getopts ":ha:i:d:" OPTION; do
    case $OPTION in

        a)
            ADDITIONAL_DOCKER_ARGS=$OPTARG
            ;;
        i)
            IMAGE_NAME=$OPTARG
            ;;
        d)
            DATASETS_FOLDER_HOST=$OPTARG
            ;;
        h | *)
            echo "Run the nvblox development docker"
            echo "Usage:"
            echo "run_docker.sh -a "additional_docker_args""
            echo "run_docker.sh -h"
            echo ""
            echo "  -a Additional arguments passed to docker run."
            echo "  -h help (this output)"
            echo "  -i image name to launch. Defaults to dev image."
            exit 0
            ;;
    esac
done

echo "IMAGE_NAME: $IMAGE_NAME"
echo "ADDITIONAL_DOCKER_ARGS: $ADDITIONAL_DOCKER_ARGS"

# The optionally mounted datasets folder.
# If not provided, the default is $HOME/datasets
if [ -z "$DATASETS_FOLDER_HOST" ]; then
    DATASETS_FOLDER_HOST=$HOME/datasets
fi
DATASETS_FOLDER_DOCKER=/datasets

# This portion of the script will only be executed *inside* the docker when
# this script is used as entrypoint further down. It will setup an user account for
# the host user inside the docker s.t. created files will have correct ownership.
if [ -f /.dockerenv ]
then
    set -euxo pipefail

    # Make sure that all shared libs are found. This should normally not be needed, but resolves a
    # problem with the opencv installation. For unknown reaosns, the command doesn't bite if placed
    # at the end of the dockerfile
    ldconfig

    # Add the group of the user. User/group ID of the host user are set through env variables when calling docker run further down.
    groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

    # Re-add the user
    userdel "$DOCKER_RUN_USER_NAME" || true
    if id $DOCKER_RUN_USER_ID; then
        echo "User $DOCKER_RUN_USER_NAME with $DOCKER_RUN_USER_ID already exists."
        EXISTING_USER_NAME=$(id -nu $DOCKER_RUN_USER_ID)
        echo "Existing user name: $EXISTING_USER_NAME. Deleting"
        userdel "$EXISTING_USER_NAME"
    fi
    useradd --no-log-init \
            --create-home \
            --uid "$DOCKER_RUN_USER_ID" \
            --gid "$DOCKER_RUN_GROUP_NAME" \
            --groups sudo \
            --shell /bin/bash \
            $DOCKER_RUN_USER_NAME
    chown $DOCKER_RUN_USER_NAME /home/$DOCKER_RUN_USER_NAME

    # Change the root user password (so we can su root)
    echo 'root:root' | chpasswd
    echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

    # Allow sudo without password
    echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

    # Create the datasets folder if it doesn't exist.
    if [ ! -d "$DATASETS_FOLDER_DOCKER" ]; then
        mkdir -p $DATASETS_FOLDER_DOCKER
        chown $DOCKER_RUN_USER_NAME $DATASETS_FOLDER_DOCKER
    fi

    set +x

    GREEN='\033[0;32m'
    IGREEN='\033[0;92m'
    NO_COLOR='\033[0m'

    echo -e "${GREEN}********************************************************"
    echo -e "* ${IGREEN}NVBLOX DEV DOCKER"
    echo -e "${GREEN}********************************************************"
    echo -e ${NO_COLOR}
    # Change into the host user and start interactive session
    su $DOCKER_RUN_USER_NAME
    exit
fi

# Detect L4T major version on Jetson.
# Sets L4T_MAJOR (e.g. "36", "38") on success, returns non-zero otherwise.
# Try 1: parse /etc/nv_tegra_release (authoritative on any L4T image).
# Try 2: fall back to the JETSON_L4T env variable (legacy behavior).
detect_l4t_major() {
    L4T_MAJOR=""

    if [ -r /etc/nv_tegra_release ]; then
        # First line looks like: "# R36 (release), REVISION: 3.0, ..."
        L4T_MAJOR=$(sed -n 's/^#[[:space:]]*R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release | head -n1)
        if [ -n "$L4T_MAJOR" ]; then
            echo "Detected L4T R${L4T_MAJOR} from /etc/nv_tegra_release" >&2
            return 0
        fi
        echo "WARNING: /etc/nv_tegra_release exists but could not be parsed" >&2
    fi

    if [ -n "${JETSON_L4T:-}" ]; then
        L4T_MAJOR="${JETSON_L4T%%.*}"
        echo "Detected L4T R${L4T_MAJOR} from \$JETSON_L4T" >&2
        return 0
    fi

    return 1
}

DOCKER_BUILD_ARGS=()
if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=nvblox_deps

    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        DOCKERFILE="Dockerfile.deps"
    elif [ "$ARCH" = "aarch64" ]; then
        DOCKERFILE="Dockerfile.jetson_deps"
        # On Jetson we must pick a base image matching the host's L4T release,
        if ! detect_l4t_major; then
            echo "ERROR: could not detect L4T version." >&2
            echo "  Neither /etc/nv_tegra_release nor \$JETSON_L4T yielded a result." >&2
            echo "  Set JETSON_L4T=<major>.<minor> (e.g. 36.3 or 38.0) and retry." >&2
            exit 1
        fi
        case "$L4T_MAJOR" in
            # Jetpack 6 (L4T 36.x): Use the L4T JetPack 36 base image
            36) DOCKER_BUILD_ARGS=(--build-arg "BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.3.0") ;;
            # Thor / Jetpack 7 (L4T 38.x): use the PyTorch JP7 image
            38) DOCKER_BUILD_ARGS=(--build-arg "BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3") ;;
            *)
                echo "ERROR: unsupported L4T major version: R${L4T_MAJOR}" >&2
                echo "  Supported: R36 (JetPack 6), R38 (JetPack 7 / Thor)." >&2
                exit 1
                ;;
        esac
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
# Build the container.
docker build --network=host -t "$IMAGE_NAME" -f "docker/$DOCKERFILE" "${DOCKER_BUILD_ARGS[@]}" .
fi

# Create the ccache directory if it doesn't exist. Otherwise, docker run will create it with root ownership.
mkdir -p $HOME/.ccache

# Remove any exited containers.
if [ "$(docker ps -a --quiet --filter status=exited --filter name=$IMAGE_NAME)" ]; then
    docker rm $IMAGE_NAME > /dev/null
fi

# If container is running, attach to it, otherwise start
if [ "$( docker container inspect -f '{{.State.Running}}' $IMAGE_NAME)" = "true" ]; then
  echo "Container already running. Attaching."
  docker exec -it $IMAGE_NAME su $(id -un)

else
    DOCKER_RUN_ARGS+=("--name" "$IMAGE_NAME"
                      "--privileged"
                      "--net=host"
                      "--runtime=nvidia"
                      "--gpus" 'all,"capabilities=compute,utility,graphics"'
                      "-v" ".:/workspaces/nvblox"
                      "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                      "-v" "$HOME/.Xauthority"
                      "-v" "$HOME/.ccache:$HOME/.ccache:rw"
                      "--env" "DISPLAY"
                      "--env" "DOCKER_RUN_USER_ID=$(id -u)"
                      "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
                      "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
                      "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
                      "--entrypoint" "/workspaces/nvblox/docker/run_docker.sh"
                      "--workdir" "/workspaces/nvblox"
                 )
    if [ -d "$DATASETS_FOLDER_HOST" ]; then
        DOCKER_RUN_ARGS+=("-v" "$DATASETS_FOLDER_HOST:$DATASETS_FOLDER_DOCKER")
    else
        echo "DATASETS_FOLDER: $DATASETS_FOLDER_HOST does not exist. It will not be mounted."
    fi
    if [ -n "${ADDITIONAL_DOCKER_ARGS}" ]; then
        DOCKER_RUN_ARGS+=($ADDITIONAL_DOCKER_ARGS)
    fi

    docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty "$IMAGE_NAME"
fi
