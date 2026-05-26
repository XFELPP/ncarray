#!/usr/bin/env bash
set -e

## This build script will create an isolated build environment, and cache it for reuse
## By caching, we can use meson (fast) to actually build C++ code.
## pip gets used only at the very end to install the entry-points.

PROJECT_NAME="ncarray"

usage()
{
    cat << EOF
$(basename "$0"):
    Build an installation of ${PROJECT_NAME}.

    This build script will create an isolated build environment. It is cached in
    your home directory under ~/.cache/${PROJECT_NAME}_build_env_XXXX where the final portion
    is created from a hash of base installation directory.

    Subsequent runs of the build will not need to re-create the build environment,
    speeding up the process significantly. You can of course delete the build environment
    from the specified folder at any time, and it will be recreated next time the script
    is run. You can pass a parameter to this script to do cleanup as well.

    Options:
        -c|--clean
          Clean up the build environment
        -d|--debug
          Create a debug release.
        -h|--help
          Display this message.
        -t|--tests
          Run test suite.

    Options that apply on subsequent runs of the build script:
        -e|--entry_points
          Re-run the pip install command. This is only needed if pyproject.toml is
          modified.
        -r|--reconfigure
          Re-run the meson setup. This is only required if meson.build files have been
          modified, or meson options/the install prefix have changed since the last
          time it was run.
EOF
}

POSITIONAL=()

while [[ $# -gt 0 ]]
do
    flag="$1"

    case $flag in
    -c|--clean)
        NEED_CLEANUP=1
        shift
        ;;
    -d|--debug)
        DEBUG_RELEASE=1
        shift
        ;;
    -e|--entry_points)
        NEED_ENTRYPOINTS=1
        shift
        ;;
    -r|--reconfigure)
        NEED_RECONFIG=1
        shift
        ;;
    -t|--tests)
        RUN_TESTS=1
        shift

        ;;
    -h|--help)
        usage
        exit
        ;;
    *)
        POS+=("$1")
        shift
        ;;
    esac
done
set -- "${POS[@]}"

# Bunch of functions to pretty print...
center_text() {
    local TEXT="$1"
    local INNER_WIDTH="$2"
    local LEN=${#TEXT}
    if (( LEN >= INNER_WIDTH )); then
        echo "===== ${TEXT} ====="
    else
        local LEFT_PAD=$(( (INNER_WIDTH - LEN) / 2))
        local RIGHT_PAD=$(( INNER_WIDTH - LEN - LEFT_PAD ))
        # Calculate spaces on each side
        printf "===== %*s%s%*s =====\n" \
               "${LEFT_PAD}" "" \
               "${TEXT}" \
               "${RIGHT_PAD}" ""
        #NEW_LINE=$(printf "%*s%s%*s\n" "${PAD_SIZE}" "" "${TEXT}" "$((WIDTH - LEN - PAD_SIZE))" " ")
        #echo "===== ${NEW_LINE} ====="
    fi
}

## Create a banner that looks like:
## ===============================
## =====         Line 1      =====
## =====         Line 2      =====
## =====         Line 3      =====
## ===============================
## For the LINES that are provided
print_banner() {
    local LINES=("$@")
    local MAXLEN=0
    local NEWLINES=()
    for LINE in "${LINES[@]}"; do
        NEWLINES+=("$LINE")
        (( ${#LINE} > MAXLEN )) && MAXLEN=${#LINE}
    done

    local BORDERLEN=$((MAXLEN+12))
    if (( BORDERLEN < 80)); then
        BORDERLEN=80
        #MAXLEN=$((BORDERLEN - 12))
    fi
    local INNER_WIDTH=$((BORDERLEN - 12))

    # Print top border
    printf '%*s\n' "${BORDERLEN}" '' | tr ' ' '='

    # Print centered lines
    for LINE in "${NEWLINES[@]}"; do
        center_text "${LINE}" "${INNER_WIDTH}"
    done

    # Print the bottom border
    printf '%*s\n' "${BORDERLEN}" '' | tr ' ' '='
}

LINES=("Building an installation of ${PROJECT_NAME}...")
print_banner "${LINES[@]}"

# Determine build directories, install directory, and where to put the build env
BASE_DIR="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
BUILD_DIR="${BASE_DIR}/_build"
INSTALL_DIR="${BASE_DIR}/install"

# Virtual environment needs to be outside the source tree
# Otherwise you get some path errors with meson
# At least I couldn't figure out any other way to do it...
BUILD_ENV="${HOME}/.cache/${PROJECT_NAME}_build_env_$(echo ${BASE_DIR} | md5sum | cut -d' ' -f1)"

mkdir -p "${INSTALL_DIR}"
mkdir -p "${HOME}/.cache"

# On S3DF get a standard Python3 - otherwise you're on your own
if [[ $HOSTNAME =~ "sdf" ]]; then
    LINES=("Sourcing the Psana1 environment (for Python3)")
    print_banner "${LINES[@]}"
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
fi

# Save host/conda env Python for later
HOST_PYTHON=$(which python3)

# Create a build environment if it doesn't yet exist
if [ ! -d "${BUILD_ENV}" ]; then
    LINES=("Creating isolated build environment in ${BUILD_ENV}...")
    print_banner "${LINES[@]}"
    python3 -m venv "${BUILD_ENV}"
    # Source before installing dependencies
    source "${BUILD_ENV}/bin/activate"
    pip install --upgrade pip
    # Install the relevant build dependencies and nothing else
    pip install "meson>=1.10.1" "meson-python" "ninja" "numpy" "pybind11" "setuptools"
    # Mark that this is a first-build. This is used to make decisions later
    FIRST_BUILD=1
else
    # If it already exists (you've run build.sh before) just activate it
    LINES=("Activating build environment at ${BUILD_ENV}")
    print_banner "${LINES[@]}"
    source "${BUILD_ENV}/bin/activate"
fi

LINES=(
    "Will build and installation of ${PROJECT_NAME} at ${INSTALL_DIR}"
    "(Build cache available at: ${BUILD_DIR})"
    "(Build environment available at: ${BUILD_ENV})"
)

print_banner "${LINES[@]}"

# Setup ccache
export CC="ccache gcc"
export CXX="ccache g++"
export CUDAHOSTCXX="ccache g++"

# Run meson configure/setup if it hasn't be done yet or it has been requested
# It generally only needs to rerun if install prefix has changed, or meson.build
# files have been modified.
if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"

    if [[ ${DEBUG_RELEASE} ]]; then
        LINES=("Running meson setup for build configuration [DEBUG RELEASE]")
        NCA_RELTYPE=debug
    else
        LINES=("Running meson setup for build configuration")
        NCA_RELTYPE=release
    fi
    print_banner "${LINES[@]}"
    meson setup "${BUILD_DIR}" -Dnca_cuda_archs="80" -Dbuild_examples=true --prefix="${INSTALL_DIR}" -Dbuildtype=${NCA_RELTYPE}
elif [[ ${FIRST_BUILD} || ${NEED_RECONFIG} ]]; then
    LINES=("Running meson setup reconfiguration")
    print_banner "${LINES[@]}"
    # Reconfigure in case prefix or options changed, but keep cache
    meson setup "${BUILD_DIR}" -Dnca_cuda_archs="80" -Dbuild_examples=true --reconfigure --prefix="${INSTALL_DIR}"
else
    LINES=(
        "!!!!! Skipping meson setup reconfiguration !!!!!"
        "This is normally fine; however, if you have modified any meson.build"
        "files, have changed meson options, or the install prefix, you should"
        "re-run this script with the -r option to reconfigure meson."
    )
    print_banner "${LINES[@]}"
fi

LINES=("Compiling...")
print_banner "${LINES[@]}"
meson compile -j 4 -C "${BUILD_DIR}" -l 4

# Test...
if [[ ${RUN_TESTS} ]]; then
    LINES=("Running test cases...")
    print_banner "${LINES[@]}"
    meson test -C "${BUILD_DIR}"
fi

# Installation - this always needs to run. This does do the installation of
# the Python source code as well
LINES=("Installing files in ${INSTALL_DIR}...")
print_banner "${LINES[@]}"
meson install -C "${BUILD_DIR}"

# Run pip at the end - this gets the entrypoints defined in pyproject.toml
# It can be pointed at the build directory to prevent pip from trying to rebuild
# the rest of the stuff from scratch.

# We will also use the underlying Python (e.g. from psconda.sh)
# This way, the build env can be kept small, and it can be deleted as well.
# Otherwise, the Python scripts would end up pointing to the Python from that env.
if [[ ${FIRST_BUILD} || ${NEED_ENTRYPOINTS} ]]; then
    LINES=("Creating Python entry points")
    print_banner "${LINES[@]}"
    BUILD_VENV_SITE_PACKAGES=(${BUILD_ENV}/lib/python*/site-packages)
    PYTHONPATH="${BUILD_VENV_SITE_PACKAGES}:${PYTHONPATH}" \
    PATH="${BUILD_ENV}/bin:${PATH}" \
    ${HOST_PYTHON} -m pip install . \
        --prefix="${INSTALL_DIR}" \
        --no-dependencies \
        --no-build-isolation \
        --config-settings=build-dir="${BUILD_DIR}"
else
    LINES=(
        "!!!!! Skipping entry-point recreation !!!!!"
        "This is normally fine; however, if you have modified pyproject.toml"
        "you may need to rerun this script with the -e argument to recreate"
        "the entry points."
    )
    print_banner "${LINES[@]}"
fi

if [[ ${NEED_CLEANUP} ]]; then
    LINES=(
        "!!!!! Cleaning up the build environment !!!!!"
        "It will be re-created if you re-run this script."
    )
    print_banner "${LINES[@]}"
    rm -rf "${BUILD_ENV}"
fi
