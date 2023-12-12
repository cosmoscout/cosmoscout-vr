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

This mode allows computation of phase functions as well as scattering- and absorption cross-sections for a given particle mixture.
The particles follow a specified multi-modal size distribution and can have a complex, wavelength-dependent refractive index.
The results are stored in a CSV file.
Use `./atmosphere-generator mie --help` to learn about all the options.

> [!IMPORTANT]
> In this mode, length units must always be given in µm. For instance, this is true for wavelengths and for particle radii.

Here are some other examples to get you started:

```bash
# This uses the particle settings from settings/marsBimodal.json and precomputes the phase
# functions and scattering cross-sections for the three given wavelengths. The output will be
# written to the default 'particles.csv' files. The phase function will be sampled in one-
# degree steps.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/mie-settings/marsBimodal.json \
                           --lambdas 0.44,0.55,0.68 --theta-samples 91 --radius-samples 10000
```

```bash
# This computes phase functions (in half-degree steps) and scattering cross-sections for
# 15 default wavelengths for rain-drop like particles and writes the output to 'rain.csv'.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/mie-settings/rain.json \
                           -o rain.csv --theta-samples 181 --radius-samples 1000
```

### `density` Mode

This mode samples a given multi-modal density function at evenly spaced altitudes and writes the resulting data to a CSV file.
Use `./atmosphere-generator density --help` to learn about all the options.

> [!IMPORTANT]
> In this mode, length units must always be given in m. For instance, this is true for altitudes and scale heights.

Here are some other examples to get you started:

```bash
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/rain.json
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/mars.json --max-altitude 60000 -o marsDensity.csv
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/ozone.json -o ozoneDensity.csv
```