<!--
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Eclipse Shadow Map Generator

This tool can be used to generate the eclipse shadow maps used by CosmoScout VR.

## Building

**Per default, the eclipse shadow map generator is not built.
To build it, you need to pass `-DCS_ECLIPSE_SHADOW_GENERATOR=On` in the make script.**

Cuda support in CMake is sometimes a bit wonky, so if you run into trouble, you can also try to build the eclipse shadow map generator manually.
This small script may serve as an example on how to do this:

```bash
#!/bin/bash

SRC_DIR="$( cd "$( dirname "$0" )" && pwd )"

nvcc -ccbin g++-12 -allow-unsupported-compiler -arch=sm_75 -rdc=true \
     -Xcompiler --std=c++17 -Xcompiler \"-Wl,-rpath-link,"$SRC_DIR/../../install/linux-Release/lib"\" \
     -Xcompiler \"-Wl,--disable-new-dtags,-rpath,"$SRC_DIR/../../install/linux-Release/lib"\" "$SRC_DIR"/*.cu \
     -I"$SRC_DIR/../../build/linux-Release/src/cs-utils" \
     -I"$SRC_DIR/../../install/linux-externals-Release/include" \
     -L"$SRC_DIR/../../install/linux-Release/lib" \
     -lcs-utils \
     -o eclipse-shadow-generator
```

## Usage

Once compiled, you'll need to set the library search path to contain the `install/<os>-<build_type>/lib` directory.
This depends on where the `eclipse-shadow-generator` is installed to, but this may be something like this:

```powershell
# For powershell
$env:Path += ";..\lib"

# For bash
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
```

To learn about the usage of `eclipse-shadow-generator`, you can now issue this command:

```bash
./eclipse-shadow-generator --help
```

Here are some simple examples to get you started:

```bash
# This simple command creates the default eclipse shadow map of CosmoScout VR
./eclipse-shadow-generator limb-darkening --with-umbra --output "fallbackShadow.tif"

./eclipse-shadow-generator bruneton --with-umbra --input ../share/resources/atmosphere-data/earth/ --radius-occ 6370900 --radius-atmo 6451000 --sun-occ-dist 149600000000 --output "earthShadow.tif" --size 512
./eclipse-shadow-generator bruneton --with-umbra --input ../share/resources/atmosphere-data/mars/ --radius-occ 3389500 --radius-atmo 3469500 --sun-occ-dist 227900000000 --output "marsShadow.tif" --size 512
./eclipse-shadow-generator limb-luminance --with-umbra --input ../share/resources/atmosphere-data/earth/ --radius-occ 6370900 --radius-atmo 6451000 --sun-occ-dist 149600000000 --output "earthLimbLuminance.tif" --size 64
./eclipse-shadow-generator limb-luminance --with-umbra --input ../share/resources/atmosphere-data/mars/ --radius-occ 3389500 --radius-atmo 3469500 --sun-occ-dist 227900000000 --output "marsLimbLuminance.tif" --size 64

# These are used for debugging purposes and can be used to visualize the results of the atmosphere rendering.
./eclipse-shadow-generator planet-view --input ../share/resources/atmosphere-data/earth/ --exposure 0.00005 --x 0.5 --y 0.5 --fov 1 --size 1024
./eclipse-shadow-generator atmo-view --input ../share/resources/atmosphere-data/earth/ --with-umbra --exposure 0.00005 --x 0.2 --y 0.3 --size 1024

# Here are some other examples related to the paper "Real-Time Rendering of Eclipses without Incorporation of Atmospheric Effects"
./eclipse-shadow-generator circles --output "circles.tif"
./eclipse-shadow-generator smoothstep --output "smoothstep.tif"
./eclipse-shadow-generator linear --with-umbra --mapping-exponent 5 --output "linear_with_umbra.tif"
```

For visualization purposes, you can use the following to create an animation of 250 frames where the Sun gradually sets behind the Earth:

```bash
mkdir output

for i in {0..150}; do
  y=$(echo "scale=4; (150 - $i) / 150" | bc)
  echo "Generating frame $i with delta $delta"
  ./eclipse-shadow-generator atmo-view --input ../share/resources/atmosphere-data/earth/ --output "output/shadow_$i.tif" --exposure 0.00005 --x 0.3 --y $y --with-umbra --size 1024
done

```
