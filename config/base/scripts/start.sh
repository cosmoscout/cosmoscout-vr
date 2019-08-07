#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Change working directory to the location of this script.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# Scene config file can be passed as first parameter.
SETTINGS="${1:-../share/config/simple_desktop.json}"

# Vista ini can be passed as second parameter.
VISTA_INI="${2:-vista.ini}"

# Set paths so that all libraries are found.
export LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH
export VISTACORELIBS_DRIVER_PLUGIN_DIRS=../lib/DriverPlugins

# gdb --args ./cosmoscout --settings=$SETTINGS -vistaini $VISTA_INI
./cosmoscout --settings=$SETTINGS -vistaini $VISTA_INI
