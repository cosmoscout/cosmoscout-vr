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

> [!IMPORTANT]
> Unless stated otherwise, length units must always be given in m. For instance, this is true for altitudes, wavelengths, and for particle radii.

### `mie` Mode

This mode computes phase functions as well as scattering- and absorption coefficients for a given particle mixture.
The particle mixture follows a specified multi-modal size distribution and can have a complex, wavelength-dependent refractive index.
The results are stored in a CSV files.
Use `./atmosphere-generator mie --help` to learn about all the options.

Here are some other examples to get you started:

```bash
# This uses the particle settings from settings/marsBimodal.json and precomputes the phase
# functions and scattering coefficients for the three given wavelengths. The output will be
# written to the default 'mie_phase.csv', 'mie_scattering.csv', and 'mie_absorption.csv' files.
# The phase function will be sampled in one-degree steps.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/mie-settings/mars_bimodal.json \
                           --lambdas 440e-9,550e-9,680e-9 --theta-samples 91 \
                           --number-density 7e6 --radius-samples 10000
```

```bash
# This computes phase functions (in half-degree steps) and scattering coefficients for
# 15 default wavelengths for rain-drop like particles and writes the output to
# 'rain_phase.csv', 'rain_scattering.csv', and 'rain_absorption.csv'.
./atmosphere-generator mie -i ../../../tools/atmosphere-generator/mie-settings/earth_rain.json \
                           -o rain --theta-samples 181 --radius-samples 1000
```

### `rayleigh` Mode

This mode writes the phase function and scattering coefficients of Rayleigh molecules in m² for the specified wavelengths to a CSV file.
Use `./atmosphere-generator rayleigh --help` to learn about all the options.
Here is an example:

```bash
# This will write scattering data for 15 default wavelengths for small
# molecules to 'rayleigh_phase.csv', 'rayleigh_scattering.csv', and 
# 'rayleigh_absorption.csv'.
./atmosphere-generator rayleigh
```

### `manual` Mode

This mode writes some user-specified values for the scattering coefficients or absorption coefficients for the specified wavelengths to a CSV file.
Use `./atmosphere-generator manual --help` to learn about all the options.
Here are some examples:

```bash
# Write three different scattering coefficients for the given wavelengths.
/atmosphere-generator manual --lambdas 440e-9,510e-9,680e-6 --quantity beta_sca --values 0.1,0.2,0.3 -o scattering

# Write 0 absorption for all default wavelengths.
./atmosphere-generator manual --quantity beta_abs --values 0 -o absorption
```

### `cornette`, `henyey`, and `dhenyey` Modes

These modes write either the Cornette-Shanks, the Henyey-Greenstein, or the Double-Henyey-Greenstein parametric phase function for the specified wavelengths to a CSV file.
Use `./atmosphere-generator <mode> --help` to learn about all the options.

### `ozone` Mode

This mode writes the absorption coefficients of ozone molecules in m² for the specified wavelengths to a CSV file.
Use `./atmosphere-generator ozone --help` to learn about all the options.
Here is an example:

```bash
# This will write ozone absorption coefficients for
# 15 default wavelengths to ozone_absorption.csv.
./atmosphere-generator ozone
```

### `density` Mode

This mode samples a given multi-modal density function at evenly spaced altitudes and writes the resulting data to a CSV file.
Use `./atmosphere-generator density --help` to learn about all the options.

Here are some other examples to get you started:

```bash
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_rain.json -o rain
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/mars_bimodal.json --max-altitude 60000 -o mars
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_bruneton_ozone.json -o ozone
```

## Creating Atmospheres According to Different Papers

### Collienne (Mars)

```bash
# Molecules are modelled using a manual parametrization of Rayleigh scattering.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/mars_collienne_molecules.json -o mars_collienne_molecules
./atmosphere-generator rayleigh --lambdas 440e-9,510e-9,680e-6 -o mars_collienne_molecules
./atmosphere-generator manual --lambdas 440e-9,510e-9,680e-6 --quantity beta_sca --values 5.75e-6,13.57e-6,19.918e-6 -o mars_collienne_molecules_scattering

# Aerosols use a wavelength-independent Cornette-Shanks phase function and some arbitrary density values.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/mars_collienne_aerosols.json -o mars_collienne_aerosols
./atmosphere-generator cornette --lambdas 440e-9,510e-9,680e-6 --g 0.76 -o mars_collienne_aerosols
./atmosphere-generator manual --lambdas 440e-9,510e-9,680e-6 --quantity beta_sca --values 3e-6 -o mars_collienne_aerosols_scattering
./atmosphere-generator manual --lambdas 440e-9,510e-9,680e-6 --quantity beta_abs --values 0 -o mars_collienne_aerosols_absorption
```

### Bruneton 2008 (Earth)

```bash
# Molecules are modelled using standard Rayleigh scattering. However, neither the molecular number
# density nor the index of refraction is given. Hence, we use the explicitly given numbers.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_bruneton_molecules.json -o earth_bruneton2008_molecules
./atmosphere-generator rayleigh --lambdas 440e-9,550e-9,680e-9 -o earth_bruneton2008_molecules
./atmosphere-generator manual --lambdas 440e-9,550e-9,680e-9 --quantity beta_sca --values 33.1e-6,15.5e-6,5.8e-6 -o earth_bruneton2008_molecules_scattering

# Aerosols use a wavelength-independent Cornette-Shanks phase function and some arbitrary density values.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_bruneton_aerosols.json -o earth_bruneton2008_aerosols
./atmosphere-generator cornette --lambdas 440e-9,550e-9,680e-9 --g 0.76 -o earth_bruneton2008_aerosols
./atmosphere-generator manual --lambdas 440e-9,550e-9,680e-9 --quantity beta_sca --values 2.1e-3 -o earth_bruneton2008_aerosols_scattering
./atmosphere-generator manual --lambdas 440e-9,550e-9,680e-9 --quantity beta_abs --values 2.1e-4 -o earth_bruneton2008_aerosols_absorption
```

### Bruneton 2016 (Earth)

```bash
# Molecules are modelled using standard Rayleigh phase function and extinction values from Penndorf.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_bruneton_molecules.json -o earth_bruneton2016_molecules
./atmosphere-generator rayleigh --lambda-samples 40 --penndorf-extinction -o earth_bruneton2016_molecules

# Aerosols use a wavelength-independent Cornette-Shanks phase function and some arbitrary density values.
./atmosphere-generator density -i ../../../tools/atmosphere-generator/density-settings/earth_bruneton_aerosols.json -o earth_bruneton2016_aerosols
./atmosphere-generator cornette --lambda-samples 40 --g 0.7 -o earth_bruneton2016_aerosols
./atmosphere-generator angstrom --lambda-samples 40 --alpha 0.8 --beta 0.04 --single-scattering-albedo 0.8 --scale-height 1200 -o earth_bruneton2016_aerosols

# In his 2016 paper, Bruneton included Ozone.
./atmosphere-generator ozone --lambda-samples 40 -o earth_bruneton2016_ozone
```
