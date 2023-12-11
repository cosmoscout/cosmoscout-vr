<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->
 
 # Atmosphere Lookup-Texture Generator

This tool can be used to generate the precomputed atmosphere data required for the advanced atmosphere model used by CosmoScout VR.

## Building

**Per default, the atmosphere generator is not built.
To build it, you need to pass `-DCS_ATMOSPHERE_GENERATOR=On` in the make script.**

## Usage

Once compiled, you'll need to set the library search path to contain the `install/<os>-<build_type>/lib` directory.
This depends on where the `atmosphere-generator` is installed to, but this may be something like this:

```powershell
# For powershell
cd install\windows-Release\bin
$env:Path += ";..\lib"

# For bash
cd install/linux-Release/bin
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
```

To learn about the different operation modes, you can now issue this command:


```bash
./atmosphere-generator --help
```

### `mie` Mode

Use `./atmosphere-generator mie --help` to learn about all the options.

> [!IMPORTANT]
> Length units must always be given in µm. This is true for instance for wavelengths and for particle radii.


Here are some other examples to get you started:

```bash
# This uses the particle settings from settings/marsBimodal.json and precomputes the phase
# functions and scattering coefficients for the three given wavelengths. The output will be
# written to the default 'particles.csv' files. The phase function will be sampled in one-
# degree steps.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/particle-settings/marsBimodal.json \
                           --lambdas 0.44,0.55,0.68 --theta-samples 91 --radius-samples 10000
```

```bash
# This computes phase functions (in half-degree steps) and scattering coefficients for
# 15 default wavelengths for rain-drop like particles and writes the output to 'rain.csv'.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/particle-settings/rain.json \
                           -o rain.csv --theta-samples 181 --radius-samples 100
```