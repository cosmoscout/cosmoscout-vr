#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
# usage: ./make_debug.sh [build_directory] [install_directory] [externals_install_directory]     #
# ------------------------------------------------------------------------------------------------ #

# Exit on error.
set -e

# create some required variables -------------------------------------------------------------------

# This directory should contain the top-level CMakeLists.txt - it is assumed to reside in the same
# directory as this script.
CMAKE_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Get the current directory - this is the default location for the build and install directory.
CURRENT_DIR="$(pwd)"

# The build directory can be passed as first parameter.
BUILD_DIR="${1:-$CURRENT_DIR/build/linux-debug}"

# The install directory can be passed as second parameter.
INSTALL_DIR="${2:-$CURRENT_DIR/install/linux-debug}"

# This directory should be the one used as install directory for make_externals.sh.
EXTERNALS_INSTALL_DIR="${3:-$CURRENT_DIR/install/linux-externals}"

# create build directory if neccessary -------------------------------------------------------------

if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
fi

# configure, compile & install ---------------------------------------------------------------------

cd "$BUILD_DIR"
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=Debug -DCOSMOSCOUT_EXTERNALS_DIR="$EXTERNALS_INSTALL_DIR" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=On "$CMAKE_DIR"

cmake --build . --target install --parallel 8

# Delete empty files install by cmake
find "$INSTALL_DIR" -type d -empty -delete

echo "Finished successfully."
