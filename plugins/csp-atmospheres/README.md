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
For now, it supports a simple "CosmoScoutVR" model which performs raytracing in the fragment shader without any preprocessing as well a more advanced "Bruneton" model based on the [excellent work](https://github.com/ebruneton/precomputed_atmospheric_scattering) by Eric Bruneton.


This plugin can be enabled with the following configuration in your `settings.json`.
Then, you will have to add configuration objects for each body to the `atmospheres` object.

```javascript
{
  ...
  "plugins": {
    ...
    "csp-atmospheres": {
      "atmospheres": {}
    },
    ...
  }
}
```

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

### Available Parameters

## The `Bruneton` model

The Bruneton model is significantly more advanced.
It pre-computes multiple scattering and is based on [this open-source implementation](https://github.com/ebruneton/precomputed_atmospheric_scattering) (see also the corresponding [Paper](https://inria.hal.science/inria-00288758/en)).

Similar to the `CosmoScoutVR` mode, the original implementation of by Eric Bruneton uses Rayleigh scattering for molecules and the Cornette-Shanks phase function for aerosols.
We generalized this implementation to load phase functions, extinction coefficients, and particle density distributions from CSV files.
This allows us to simulate arbitrary particle types.
In particular, we can now use Mie Theory to pre-compute the scattering behaviour of a wide variety of particle types, including for instance Martian dust.

To perform this pre-processing, the `csp-atmospheres` plugin comes with a small [command-line utility](preprocessor/README.md).

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

### Available Parameters

## Creating new Atmospheric Models

For learning how to create new models, please refer to the comments in [`ModelBase.hpp`](src/ModelBase.hpp).