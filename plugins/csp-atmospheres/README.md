<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

<p align="center"> 
  <img src ="../../docs/img/banner-sunset.jpg" />
</p>

# Configurable Atmospheres for CosmoScout VR

This is a CosmoScout VR plugin which draws atmospheres around celestial bodies.
It is designed so that multiple atmospheric models can be implemented and configured through the scene settings.
For now, it supports a simple `"CosmoScoutVR"` model which performs raytracing in the fragment shader without any preprocessing as well a more advanced `"Bruneton"` model based on the [excellent work](https://github.com/ebruneton/precomputed_atmospheric_scattering) by Eric Bruneton.


This plugin can be enabled with a configuration like the following in your `settings.json`.
You will have to configure the individual `atmospheres` according to the instructions further below.

```javascript
{
  ...
  "plugins": {
    ...
    "csp-atmospheres": {
      "atmospheres": {
        "Earth": {
          ...
        },
        "Mars": {
          ...
        },
        ...
      }
    },
    ...
  }
}
```

## Model-Agnostic Parameters

For each atmosphere, there are some parameters which are available regardless of the model in use.
These parameters are available for all models:

Property | Default Value | Description
-------- | ------------- | -----------
`topAltitude` | _mandatory_ | The altitude in [m] of the upper atmosphere boundary relative to the planet's surface.
`bottomAltitude` | `0.0` | The altitude in [m] of the lower atmosphere boundary relative to the planet's surface. Can be negative.
`enableClouds` | `true` | If set to `true`, an pseudo-volumetric cloud texture will be drawn. Requires that `cloudTexture` is set as well.
`cloudTexture` | `""` | The file path to an equirectangular global cloud texture.
`cloudAltitude` | `3000.0` | The altitude in [m] of the cloud layer relative to the planet's surface.
`enableWater` | `false` | If set to `true`, an ocean will be drawn.
`enableWaves` | `true` | If set to `true`, the ocean will have some animated waves.
`waterLevel` | `0.0` | The altitude in [m] of the ocean surface relative to the planet's surface.
`model` | `"CosmoScoutVR"` | The model to use for this atmosphere. This can either be `"CosmoScoutVR"` or `"Bruneton"`.
`modelSettings` | _model-dependent_ | The parameters for the model. See below.

## The `CosmoScoutVR` model

This is a very simple model which computes single-scattering in real time.
For each pixel, a ray is cast through the atmosphere.
Single scattering contributions are integrated along the ray using Rayleigh scattering for molecules and the Cornette-Shanks phase function for aerosols.
No preprocessing is required.

Below you find example configurations for Earth and Mars.
The values used for Mars are based on the [paper by Peter Collienne](https://www.semanticscholar.org/paper/Physically-Based-Rendering-of-the-Martian-Collienne-Wolff/e71c3683a70f75aedfce3f6bad401e6819d0d713).
They are not physically based but provide some plausible results.

<details>
<summary>Example Configuration for Earth</summary>

```javascript
"Earth": {
  "topAltitude": 80000,
  "bottomAltitude": -100,
  "cloudTexture": "../share/resources/textures/earth-clouds.jpg",
  "model": "CosmoScoutVR",
  "modelSettings": {
    "mieAnisotropy": 0.76,
    "mieHeight": 1200,
    "mieScattering": [
      4.0e-5,
      4.0e-5,
      4.0e-5
    ],
    "rayleighAnisotropy": 0,
    "rayleighHeight": 8000,
    "rayleighScattering": [
      5.1768e-6,
      12.2588e-6,
      30.5964e-6
    ]
  }
}
```
</details>

<details>
<summary>Example Configuration for Mars</summary>

```javascript
"Mars": {
  "topAltitude": 100000,
  "bottomAltitude": -4500,
  "model": "CosmoScoutVR",
  "modelSettings": {
    "mieAnisotropy": 0.76,
    "mieHeight": 1200,
    "mieScattering": [
      3.0e-6,
      3.0e-6,
      3.0e-6
    ],
    "rayleighAnisotropy": 0,
    "rayleighHeight": 11000,
    "rayleighScattering": [
      19.981e-6,
      13.57e-6,
      5.75e-6
    ]
  }
}
```
</details>

Property | Default Value | Description
-------- | ------------- | -----------
`mieHeight` | `1200.0` | The scale height in [m] for aerosol particles relative to the lower atmosphere boundary.
`mieScattering` | `[4.0e-5, 4.0e-5, 4.0e-5]` | The RGB scattering coefficients for aerosols in [m^-1].
`mieAnisotropy` | `0.76` | The `g` parameter of the Cornette-Shanks phase function.
`rayleighHeight` | `8000.0` | The scale height in [m] for moleules relative to the lower atmosphere boundary.
`rayleighScattering` | `[5.1768e-6, 12.2588e-6, 30.5964e-6]` | The RGB scattering coefficients for molecules in [m^-1].
`rayleighAnisotropy` | `0.0` | The `g` parameter of the Cornette-Shanks phase function. Use `0.0` for Rayleigh scattering.
`primaryRaySteps` | `7` | The number of samples to take along each primary ray.
`secondaryRaySteps` | `3` | The number of samples to take along each secondary ray.

## The `Bruneton` model

The Bruneton model is significantly more advanced.
It pre-computes multiple scattering and is based on [this open-source implementation](https://github.com/ebruneton/precomputed_atmospheric_scattering) (see also the corresponding [Paper](https://inria.hal.science/inria-00288758/en)).

Similar to the `CosmoScoutVR` model, the original implementation of by Eric Bruneton uses Rayleigh scattering for molecules and the Cornette-Shanks phase function for aerosols.
We generalized this implementation to load phase functions, extinction coefficients, and particle density distributions from CSV files.
This allows us to simulate arbitrary particle types.
In particular, we can now use Mie Theory to pre-compute the scattering behaviour of a wide variety of particle types, including for instance Martian dust.

To perform this pre-processing, the `csp-atmospheres` plugin comes with a small command-line utility: [`atmosphere-preprocessor`](preprocessor/README.md).
You can use this to generate the CSV files used in the examples below.
Also, the [README.md](preprocessor/README.md) of the command-line utility provides more information on the required CSV file format.

<details>
<summary>Example Configuration for Earth</summary>

```javascript
"Earth": {
  "cloudTexture": "../share/resources/textures/earth-clouds.jpg",
  "topAltitude": 80000,
  "bottomAltitude": -100,
  "model": "Bruneton",
  "modelSettings": {
    "sunAngularRadius": 0.004675,
    "molecules": {
      "phase": "../share/resources/data/csp-atmospheres/earth_cosmoscout_molecules_phase.csv",
      "betaSca": "../share/resources/data/csp-atmospheres/earth_cosmoscout_molecules_scattering.csv",
      "betaAbs": "../share/resources/data/csp-atmospheres/earth_cosmoscout_molecules_absorption.csv",
      "density": "../share/resources/data/csp-atmospheres/earth_cosmoscout_molecules_density.csv"
    },
    "aerosols": {
      "phase": "../share/resources/data/csp-atmospheres/earth_cosmoscout_aerosols_phase.csv",
      "betaSca": "../share/resources/data/csp-atmospheres/earth_cosmoscout_aerosols_scattering.csv",
      "betaAbs": "../share/resources/data/csp-atmospheres/earth_cosmoscout_aerosols_absorption.csv",
      "density": "../share/resources/data/csp-atmospheres/earth_cosmoscout_aerosols_density.csv"
    },
    "ozone": {
      "betaAbs": "../share/resources/data/csp-atmospheres/earth_cosmoscout_ozone_absorption.csv",
      "density": "../share/resources/data/csp-atmospheres/earth_cosmoscout_ozone_density.csv"
    }
  }
}
```
</details>

<details>
<summary>Example Configuration for Mars</summary>

```javascript
"Mars": {
  "topAltitude": 80000,
  "bottomAltitude": -4500,
  "model": "Bruneton",
  "modelSettings": {
    "sunAngularRadius": 0.003054,
    "molecules": {
      "phase": "../share/resources/data/csp-atmospheres/mars_cosmoscout_molecules_phase.csv",
      "betaSca": "../share/resources/data/csp-atmospheres/mars_cosmoscout_molecules_scattering.csv",
      "betaAbs": "../share/resources/data/csp-atmospheres/mars_cosmoscout_molecules_absorption.csv",
      "density": "../share/resources/data/csp-atmospheres/mars_cosmoscout_molecules_density.csv"
    },
    "aerosols": {
      "phase": "../share/resources/data/csp-atmospheres/mars_cosmoscout_aerosols_phase.csv",
      "betaSca": "../share/resources/data/csp-atmospheres/mars_cosmoscout_aerosols_scattering.csv",
      "betaAbs": "../share/resources/data/csp-atmospheres/mars_cosmoscout_aerosols_absorption.csv",
      "density": "../share/resources/data/csp-atmospheres/mars_cosmoscout_aerosols_density.csv"
    }
  }
}
```
</details>

Property | Default Value | Description
-------- | ------------- | -----------
`molecules` | _mandatory_ | This object should contain `"phase"`, `"betaSca"`, `"betaAbs"`, and `"density"` CSV file paths. See above for an example.
`aerosols` | _mandatory_ | This object should contain `"phase"`, `"betaSca"`, `"betaAbs"`, and `"density"` CSV file paths. See above for an example.
`ozone` | _optional_ | If specified, this object should contain `"betaAbs"` and `"density"` CSV file paths. See above for an example.
`sunAngularRadius` | `0.004675` | The angular radius of the Sun needs to be specified. As SPICE is not fully available when the plugin is loaded, we cannot compute it.
`groundAlbedo` | `0.1` | The average reflectance of the ground used during multiple scattering.
`multiScatteringOrder` | `4` | The number of multiple scattering events to precompute. Use zero for single-scattering only.
`sampleCountOpticalDepth` | `500` | The number of samples to evaluate when pre-computing the optical depth.
`sampleCountSingleScattering` | `50` | The number of samples to evaluate when pre-computing the single scattering. Larger values improve the sampling of thin atmospheric layers.
`sampleCountMultiScattering` | `50` | The number of samples to evaluate when pre-computing the multiple scattering. Larger values tend to darken the horizon for thick atmospheres.
`sampleCountScatteringDensity` | `16` | The number of samples to evaluate when pre-computing the scattering density. Larger values spread out colors in the sky.
`sampleCountIndirectIrradiance` | `32` | The number of samples to evaluate when pre-computing the indirect irradiance.
`transmittanceTextureWidth` | `256` | The horizontal resolution of the transmittance texture. Larger values can improve the sampling of thin atmospheric layers close to the horizon.
`transmittanceTextureHeight` | `64` | The vertical resolution of the transmittance texture. Larger values can improve the sampling of thin atmospheric layers close to the horizon.
`scatteringTextureRSize` | `32` | Larger values improve sampling of thick low-altitude layers.
`scatteringTextureMuSize` | `128` | Larger values reduce circular banding artifacts around zenith for thick atmospheres.
`scatteringTextureMuSSize` | `32` | Larger values reduce banding in the day-night transition when seen from space.
`scatteringTextureNuSize` | `8` | Larger values reduce circular banding artifacts around sun for thick atmospheres.
`irradianceTextureWidth` | `64` | The horizontal resolution of the irradiance texture.
`irradianceTextureHeight` | `16` | The vertical resolution of the irradiance texture.

## Creating new Atmospheric Models

For learning how to create new models, please refer to the comments in [`ModelBase.hpp`](src/ModelBase.hpp).