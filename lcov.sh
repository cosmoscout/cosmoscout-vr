#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Exit on error.
set -e

# ------------------------------------------------------------------------------------------------ #
# This script uses lcov to capture the source coverage of the test executed with run_all_tests.sh. #
# The environment variable COSMOSCOUT_DEBUG_BUILD is checked in order to use the data in           #
# build/linux-$BUILD_TYPE and install/linux-$BUILD_TYPE.                                           #
# If you pass any argument to the script (say ./lcov.sh foo) then it will create also an html      #
# report and open this report in your web browser.                                                 #
# ------------------------------------------------------------------------------------------------ #

# create some required variables -------------------------------------------------------------------

# Get the location of this script.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $SCRIPT_DIR

# Check if ComoScout VR debug build is enabled with "export COSMOSCOUT_DEBUG_BUILD=true".
BUILD_TYPE=Release
case "$COSMOSCOUT_DEBUG_BUILD" in
  (true) echo "CosmoScout VR debug build is enabled!"; BUILD_TYPE=Debug;
esac

# Get the current directory - this should contain the build and install directory.
CURRENT_DIR="$(pwd)"

# The build directory.
BUILD_DIR="$CURRENT_DIR/build/linux-$BUILD_TYPE"

# The install directory.
INSTALL_DIR="$CURRENT_DIR/install/linux-$BUILD_TYPE"

# create zero-coverage info ------------------------------------------------------------------------

lcov -q --zerocounters --directory .
lcov -q --capture --no-external --initial --directory . --output-file $BUILD_DIR/zero_coverage.info

# run the tests ------------------------------------------------------------------------------------

$INSTALL_DIR/bin/run_tests.sh
$INSTALL_DIR/bin/run_graphical_tests.sh

# capture the coverage of the test -----------------------------------------------------------------

lcov -q --capture --no-external --directory . --output-file $BUILD_DIR/test_coverage.info
lcov -q -a $BUILD_DIR/zero_coverage.info -a $BUILD_DIR/test_coverage.info --o $BUILD_DIR/coverage.info

# Remove any coverage from externals, examples and test directories.
lcov -q --remove $BUILD_DIR/coverage.info \*externals\* --output-file $BUILD_DIR/coverage.info
lcov -q --remove $BUILD_DIR/coverage.info \*test\* --output-file $BUILD_DIR/coverage.info

# Generate html report and open it in a web browser when an argument was passed to the script.
if [ $# != 0 ]
  then
    genhtml $BUILD_DIR/coverage.info --output-directory $BUILD_DIR/coverage
    xdg-open $BUILD_DIR/coverage/index.html
fi
