#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Change working directory to the location of this script.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# Set paths so that all libraries are found.
export LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH

# Most of the time, this tests will use MESA's software rasterizer. Currently, this only supports
# OpenGL 3.3 - so we are limited to this in our tests.
export MESA_GLSL_VERSION_OVERRIDE=330
export MESA_GL_VERSION_OVERRIDE=3.3

# Run all tests which are marked to require a display. An X-Server with a virtual framebuffer is
# used to run the tests. This way, this script can be run on a machine without a dedicated GPU.
xvfb-run --server-args="-screen 0 800x600x24" ./cosmoscout --run-tests --test-case="*[graphical]*"