#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# create the data directory if necessary
DATA_DIR="$( cd "$( dirname "$0" )" && pwd )/data"

# download the hipparcos and the tycho2 catalogue
mkdir -p "$DATA_DIR/stars"
cd "$DATA_DIR/stars"

wget -nc ftp://ftp.imcce.fr/pub/catalogs/HIPP/cats/hip_main.dat
wget -nc ftp://ftp.imcce.fr/pub/catalogs/TYCHO-2/catalog.dat -O tyc2_main.dat


# download some basic spice kernels
mkdir -p "$DATA_DIR/spice"
cd "$DATA_DIR/spice"

wget -nc https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/pck/pck00010.tpc
wget -nc https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/lsk/naif0011.tls
wget -nc https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/spk/cg_1950_2050_v01.bsp