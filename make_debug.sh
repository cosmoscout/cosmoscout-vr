#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Exit on error.
set -e

# create some required variables -------------------------------------------------------------------

# This directory should contain the top-level CMakeLists.txt - it is assumed to reside in the same
# directory as this script.
CMAKE_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Get the current directory - this is the default location for the build and install directory.
CURRENT_DIR="$(pwd)"

# The build directory.
BUILD_DIR="$CURRENT_DIR/build/linux-debug"

# The install directory.
INSTALL_DIR="$CURRENT_DIR/install/linux-debug"

# This directory should be used as the install directory for make_externals.sh.
EXTERNALS_INSTALL_DIR="$CURRENT_DIR/install/linux-externals"

# The optional parameter --with-ccache enables the ccache support of CMake.
# ccache must be installed on your system.
if [[ $* == *--with-ccache* ]]
then
    CCACHE_FLAGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"
else
    CCACHE_FLAGS=""
fi

# create build directory if neccessary -------------------------------------------------------------

if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
fi

# configure, compile & install ---------------------------------------------------------------------

cd "$BUILD_DIR"
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" $CCACHE_FLAGS \
      -DCMAKE_BUILD_TYPE=Debug -DCOSMOSCOUT_EXTERNALS_DIR="$EXTERNALS_INSTALL_DIR" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=On "$CMAKE_DIR"

cmake --build . --target install --parallel 8

# Delete empty files install by cmake
find "$INSTALL_DIR" -type d -empty -delete

echo "Finished successfully."
