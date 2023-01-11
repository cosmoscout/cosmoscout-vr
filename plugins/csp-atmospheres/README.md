<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Custom Web UI for CosmoScout VR

A CosmoScout VR plugin which allows adding custom HTML-based UI elements as sidebar-tabs, as floating windows or into free space.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values are just some examples, feel free to add your own items:

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

### CosmoScout VR Model

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

### Bruneton Model

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
