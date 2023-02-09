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
For now, it supports a simple "CosmoScoutVR" model which performs raytracing in the fragment shader without any preprocessing as well as the "Bruneton" model which is based on the [excellent work](https://github.com/ebruneton/precomputed_atmospheric_scattering) by Eric Bruneton.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.

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

Then, you will have to add configuration objects to the `atmospheres` object.
Here is an example for Earth's atmosphere using the "CosmoScoutVR" model.

```javascript
"Earth": {
  "height": 80000,
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

The same model can also be used for Mars.
This is not physically based but provides pretty plausible results.

```javascript
"Mars": {
  "height": 100000,
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
    "rayleighHeight": 11000,
    "rayleighScattering": [
      19.981e-6,
      13.57e-6,
      5.75e-6
    ]
  }
}
```

The "Bruneton" model is designed for Earth only.
Here is an exemplary parametrization which is based on the [demo code by Eric Bruneton](https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc#L188).

```javascript
"Earth": {
  "height": 80000,
  "cloudTexture": "../share/resources/textures/earth-clouds.jpg",
  "model": "Bruneton",
  "modelSettings": {
    "sunAngularRadius": 0.004675,
    "rayleigh": 1.24062e-6,
    "rayleighScaleHeight": 8000.0,
    "mieScaleHeight": 1200.0,
    "mieAngstromAlpha": 0.0,
    "mieAngstromBeta": 5.328e-3,
    "mieSingleScatteringAlbedo": 0.9,
    "miePhaseFunctionG": 0.8,
    "groundAlbedo": 0.1,
    "useOzone": true
  }
}
```

## Creating new Atmospheric Models

For learning how to create new models, please refer to the comments in [`ModelBase.hpp`](src/ModelBase.hpp).