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

> [!IMPORTANT]
> This plugin uses code published by Eric Bruneton under the BSD-3-Clause license. Consequently, not all source files are available under the MIT license. See the individual SPDX-tags for details.

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

| Property         | Default Value     | Description                                                                                                         |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `topAltitude`    | _mandatory_       | The altitude in [m] of the upper atmosphere boundary relative to the planet's surface.                              |
| `bottomAltitude` | `0.0`             | The altitude in [m] of the lower atmosphere boundary relative to the planet's surface. Can be negative.             |
| `enableClouds`   | `true`            | If set to `true`, an pseudo-volumetric cloud texture will be drawn. Requires that `cloudTexture` is set as well.    |
| `cloudTexture`   | `""`              | The file path to an equirectangular global cloud texture.                                                           |
| `cloudAltitude`  | `3000.0`          | The altitude in [m] of the cloud layer relative to the planet's surface.                                            |
| `enableWater`    | `false`           | If set to `true`, an ocean will be drawn.                                                                           |
| `enableWaves`    | `true`            | If set to `true`, the ocean will have some animated waves.                                                          |
| `waterLevel`     | `0.0`             | The altitude in [m] of the ocean surface relative to the planet's surface.                                          |
| `renderSkydome`  | `false`           | If this is set to `true`, the plugin will save a fish-eye view of the sky to a file once the preprocessing is done. |
| `model`          | `"CosmoScoutVR"`  | The model to use for this atmosphere. This can either be `"CosmoScoutVR"` or `"Bruneton"`.                          |
| `modelSettings`  | _model-dependent_ | The parameters for the model. See below.                                                                            |

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
  "bottomAltitude": 0,
  "cloudTexture": "../share/resources/textures/earth-clouds.jpg",
  "cloudTypeTexture": "../share/resources/textures/cloudTop.jpg",
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

| Property             | Default Value                         | Description                                                                                 |
| -------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------- |
| `mieHeight`          | `1200.0`                              | The scale height in [m] for aerosol particles relative to the lower atmosphere boundary.    |
| `mieScattering`      | `[4.0e-5, 4.0e-5, 4.0e-5]`            | The RGB scattering coefficients for aerosols in [1/m].                                      |
| `mieAnisotropy`      | `0.76`                                | The `g` parameter of the Cornette-Shanks phase function.                                    |
| `rayleighHeight`     | `8000.0`                              | The scale height in [m] for moleules relative to the lower atmosphere boundary.             |
| `rayleighScattering` | `[5.1768e-6, 12.2588e-6, 30.5964e-6]` | The RGB scattering coefficients for molecules in [1/m].                                     |
| `rayleighAnisotropy` | `0.0`                                 | The `g` parameter of the Cornette-Shanks phase function. Use `0.0` for Rayleigh scattering. |
| `primaryRaySteps`    | `7`                                   | The number of samples to take along each primary ray.                                       |
| `secondaryRaySteps`  | `3`                                   | The number of samples to take along each secondary ray.                                     |

## The `Bruneton` model

The Bruneton model is significantly more advanced.
It precomputes multiple scattering and is based on [this open-source implementation](https://github.com/ebruneton/precomputed_atmospheric_scattering) (see also the corresponding [Paper](https://inria.hal.science/inria-00288758/en)).

We have significantly extended the original implementation to allow for more flexibility.
See our paper [Physically Based Real-Time Rendering of Atmospheres using Mie Theory](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15010) for more details.

As a first change, we now load phase functions, extinction coefficients, and particle density distributions from CSV files.
This allows us to simulate arbitrary particle types.
In particular, we can now use Mie Theory to precompute the scattering behaviour of a wide variety of particle types, including for instance Martian dust.

Next, our implementation can compute refraction.
This allows for displaced horizons and the simulation of astronomical refraction.
This is also used for computing light entering the eclipse shadows of celestial bodies.

Another change to the original implementation is that we put the precomputation of the atmospheric scattering into a separate executable.
This allows us to perform the preprocessing offline with a much higher fidelity than what would be possible during application startup.

There are **two preprocessing steps are required to use this model**.

#### Preprocessing Step 1: Precompute the Particle-Scattering CSV Tables

In this first preprocessing step, the scattering properties of the individual particles are precomputed.

To perform this preprocessing, the `csp-atmospheres` plugin comes with a small command-line utility: [`scattering-table-generator`](scattering-table-generator/README.md).
You can use this to generate the CSV files used in the next preprocessing step.
The [README.md](scattering-table-generator/README.md) of the command-line utility provides more information on the resulting CSV file format and some examples.
For convenience, we provide some precomputed tables for Earth and Mars in the [`scattering-table-generator/output`](scattering-table-generator/output) directory.

#### Preprocessing Step 2: Precompute the Atmospheric-Scattering Textures

In the second preprocessing step, multiple scattering is precomputed for an entire atmosphere and the results are stored in lookup textures.

This step is performed by the command-line utility [`bruneton-preprocessor`](bruneton-preprocessor/README.md).
This consumes the CSV files generated in the first step and precomputes the atmospheric scattering textures.
The textures are accompanied by a JSON file containing some metadata on the precomputed values.
During runtime, the plugin will load these textures and use them to render the atmosphere.
For convenience, we provide precomputed textures for Earth and Mars in the [`bruneton-preprocessor/output`](bruneton-preprocessor/output) directory.
These are installed to `share/resources/atmosphere-data` and can be used like shown below.

### Example Configurations

Once the multiple scattering textures are precomputed, the configuration for the `Bruneton` model is very simple:

<details>
<summary>Example Configuration for Earth</summary>

```javascript
"Earth": {
  "cloudTexture": "../share/resources/textures/earth-clouds.jpg",
  "topAltitude": 80000,
  "bottomAltitude": 0,
  "model": "Bruneton",
  "modelSettings": {
    "dataDirectory": "../share/resources/atmosphere-data/earth"
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
    "dataDirectory": "../share/resources/atmosphere-data/mars"
  }
}
```

</details>

## Creating new Atmospheric Models

For learning how to create new models, please refer to the comments in [`ModelBase.hpp`](src/ModelBase.hpp).
