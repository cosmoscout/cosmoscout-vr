<!--
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Eclipse Shadow Map Generator

This tool can be used to generate the eclipse shadow maps used by CosmoScout VR as well as the precomputed limb luminance maps used for rendering the atmosphere around planets when the Sun is in opposition.

There are two basic types of eclipse shadows: Those which do not consider an atmosphere around the occluder and those which do.
The former are generated according to the paper ["Real-Time Rendering of Eclipses without Incorporation of Atmospheric Effects"](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14676).
The latter use an extended version of the Bruneton atmosphere model described in ["Physically Based Real-Time Rendering of Atmospheres using Mie Theory"](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15010).

The limb luminance maps are 3D textures which encode the luminance around the limb of a planet for every possible viewing position in the shadow of the planet.

## Building

Per default, the eclipse shadow map generator is not built.
To build it, you need to add `"CS_ECLIPSE_SHADOW_GENERATOR": "On",` to the `"cacheVariables"` in [CMakePresets.json](../../CMakePresets.json).
Then it will be built together with the rest of CosmoScout VR.

## Usage

Once compiled, you'll need to set the library search path to contain the `install/<os>-<build_type>/lib` directory.
This depends on where the `eclipse-shadow-generator` is installed to, but this may be something like this:

```powershell
# For Windows (powershell)
$env:Path += ";install\windows-Release\lib"

# For Linux (bash)
export LD_LIBRARY_PATH=install/linux-Release/lib:$LD_LIBRARY_PATH
```

To learn about the usage of `eclipse-shadow-generator`, you can now issue this command:

```bash
install/linux-Release/bin/eclipse-shadow-generator --help
```

### Creating the Eclipse Shadow Maps used by CosmoScout VR

The following commands were used to generate the eclipse shadow maps used by CosmoScout VR.
The `fallbackShadow.tif` is used for all celestial bodies which do not have a specific shadow map.
It includes the effect of limb darkening but no atmosphere.
There are two specific shadow maps for Earth and Mars, which include the effect of limb darkening and the atmosphere of the respective planet.

The shadow maps for Earth and Mars require the output of the [Bruneton atmosphere model preprocessor](../../plugins/csp-atmospheres/bruneton-preprocessor/README.md) which in turn requires the output of the [scattering-table-generator](../../plugins/csp-atmospheres/scattering-table-generator/README.md).

You find more information on how to use these tools in the respective READMEs.
For convenience, we provide the commands used to generate all necessary data below.

<details>
<summary>Scattering-Table Computation</summary>

```bash
# Earth Molecules
install/linux-Release/bin/scattering-table-generator density -i plugins/csp-atmospheres/scattering-table-generator/density-settings/earth_bruneton_molecules.json -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_molecules
install/linux-Release/bin/scattering-table-generator rayleigh --scattering-depolarization 0.0279 --phase-depolarization 0.0279 --penndorf-ior --theta-samples 91 -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_molecules

# Earth Aerosols
install/linux-Release/bin/scattering-table-generator density -i plugins/csp-atmospheres/scattering-table-generator/density-settings/earth_bruneton_aerosols.json -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_aerosols
install/linux-Release/bin/scattering-table-generator mie -i plugins/csp-atmospheres/scattering-table-generator/mie-settings/earth_haze.json --theta-samples 91 --number-density 5e8 --radius-samples 10000 -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_aerosols

# Earth Ozone
install/linux-Release/bin/scattering-table-generator density -i plugins/csp-atmospheres/scattering-table-generator/density-settings/earth_bruneton_ozone.json -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_ozone
install/linux-Release/bin/scattering-table-generator ozone -o plugins/csp-atmospheres/scattering-table-generator/output/earth_cosmoscout_ozone

# Mars Molecules
install/linux-Release/bin/scattering-table-generator density -i plugins/csp-atmospheres/scattering-table-generator/density-settings/mars_cosmoscout_molecules.json -o plugins/csp-atmospheres/scattering-table-generator/output/mars_cosmoscout_molecules_cinematic
install/linux-Release/bin/scattering-table-generator rayleigh --lambdas 440e-9,550e-9,680e-9 --ior 1.00000337 --scattering-depolarization 0.09 --phase-depolarization 0.09 --number-density 2.05e23 --theta-samples 91 -o plugins/csp-atmospheres/scattering-table-generator/output/mars_cosmoscout_molecules_cinematic

# Mars Aerosols
install/linux-Release/bin/scattering-table-generator density -i plugins/csp-atmospheres/scattering-table-generator/density-settings/mars_cosmoscout_aerosols_cinematic.json -o plugins/csp-atmospheres/scattering-table-generator/output/mars_cosmoscout_aerosols_cinematic
install/linux-Release/bin/scattering-table-generator mie --lambdas 440e-9,550e-9,680e-9 -i plugins/csp-atmospheres/scattering-table-generator/mie-settings/mars_cinematic.json --phase-flattening 0.8 --theta-samples 91 --number-density 5e9 --radius-samples 10000 -o plugins/csp-atmospheres/scattering-table-generator/output/mars_cosmoscout_aerosols_cinematic
```

</details>

<details>
<summary>Atmospheric Scattering Computation</summary>

```bash
# Earth
install/linux-Release/bin/bruneton-preprocessor plugins/csp-atmospheres/bruneton-preprocessor/settings/earth.json plugins/csp-atmospheres/bruneton-preprocessor/output/earth

# Mars
install/linux-Release/bin/bruneton-preprocessor plugins/csp-atmospheres/bruneton-preprocessor/settings/mars.json plugins/csp-atmospheres/bruneton-preprocessor/output/mars
```

</details>

<details>
<summary>Shadow-Map Computation</summary>

```bash
# Create the fallback shadow map.
install/linux-Release/bin/eclipse-shadow-generator simple-limb-darkening --with-umbra --output "resources/textures/fallbackShadow.tif" --size 256

# Create the shadow maps for Earth and Mars.
install/linux-Release/bin/eclipse-shadow-generator advanced-shadow --with-umbra --input plugins/csp-atmospheres/bruneton-preprocessor/output/earth/ --radius-occ 6371000 --radius-atmo 6451000 --sun-occ-dist 149600000000 --output "resources/textures/earthShadow.tif" --size 256
install/linux-Release/bin/eclipse-shadow-generator advanced-shadow --with-umbra --input plugins/csp-atmospheres/bruneton-preprocessor/output/mars/ --radius-occ 3389500 --radius-atmo 3469500 --sun-occ-dist 227900000000 --output "resources/textures/marsShadow.tif" --size 256

# Create the limb luminance maps for Earth and Mars.
install/linux-Release/bin/eclipse-shadow-generator limb-luminance --with-umbra --input plugins/csp-atmospheres/bruneton-preprocessor/output/earth/ --radius-occ 6371000 --radius-atmo 6451000 --sun-occ-dist 149600000000 --output "resources/textures/earthLimbLuminance.tif" --size 64
install/linux-Release/bin/eclipse-shadow-generator limb-luminance --with-umbra --input plugins/csp-atmospheres/bruneton-preprocessor/output/mars/ --radius-occ 3389500 --radius-atmo 3469500 --sun-occ-dist 227900000000 --output "resources/textures/marsLimbLuminance.tif" --size 64
```

</details>

### Recreating Paper Figures and other Examples

These are used for debugging purposes and can be used to visualize the results of the atmosphere rendering.

```bash
install/linux-Release/bin/eclipse-shadow-generator advanced-planet-view --input plugins/csp-atmospheres/bruneton-preprocessor/output/earth --with-umbra --exposure 0.0001 --x 0.099 --y 0.9 --size 1024 --fov 6 --output "planet-view.tif"
install/linux-Release/bin/eclipse-shadow-generator advanced-atmo-view --input plugins/csp-atmospheres/bruneton-preprocessor/output/earth --with-umbra --exposure 0.0001 --x 0.099 --y 0.9 --size 1024 --output "atmo-view.tif"
```

Here are some examples related to the paper "Real-Time Rendering of Eclipses without Incorporation of Atmospheric Effects".

```bash
install/linux-Release/bin/eclipse-shadow-generator simple-circles --output "circles.tif"
install/linux-Release/bin/eclipse-shadow-generator simple-smoothstep --output "smoothstep.tif"
install/linux-Release/bin/eclipse-shadow-generator simple-linear --with-umbra --mapping-exponent 5 --output "linear_with_umbra.tif"
```

For visualization purposes, you can use the following to create an animation of 250 frames where the Sun gradually sets behind the Earth:

```bash
mkdir output

for i in {0..150}; do
  y=$(echo "scale=4; (150 - $i) / 150" | bc)
  echo "Generating frame $i with delta $delta"
  install/linux-Release/bin/eclipse-shadow-generator advanced-atmo-view --input ../share/resources/atmosphere-data/earth/ --output "output/shadow_$i.tif" --exposure 0.00005 --x 0.3 --y $y --with-umbra --size 1024
done

```
