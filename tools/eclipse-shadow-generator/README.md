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

To learn about the usage, you can issue this command:


```bash
./eclipse-shadow-generator --help
```

Here are some other examples to get you started:

```bash
# This simple command creates the default eclipse shadow map of CosmoScout VR
./eclipse-shadow-generator

# Here are some other examples
./eclipse-shadow-generator --mode circles --output "circles.hdr"
./eclipse-shadow-generator --mode smoothstep --output "smoothstep.hdr"
./eclipse-shadow-generator --mode linear --with-umbra --mapping-exponent 5 --output "linear_with_umbra.hdr"
```