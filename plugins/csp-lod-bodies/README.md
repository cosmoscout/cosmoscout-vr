<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Level-of-Detail Bodies for CosmoScout VR

A CosmoScout VR plugin which draws level-of-detail planets and moons.
This plugin supports the visualization of entire planets in a 1:1 scale.
The data is streamed via Web-Map-Services (WMS) over the internet.
A dedicated MapServer is required to use this plugin.

:information_source: _We provide a containerized MapServer which can be used to load some example data as well as for serving your own custom datasets. [A detailed guide is available here](https://github.com/cosmoscout/docker-mapserver/)._

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-lod-bodies": {
      "maxGPUTilesColor": <int>,     // The maximum allowed colored tiles.
      "maxGPUTilesDEM": <int>,       // The maximum allowed elevation tiles.
      "tileResolutionDEM": <int>,    // The vertex grid resolution of the tiles.
      "tileResolutionIMG": <int>,    // The pixel resolution which is used for the image data.
      "mapCache": <string>,          // The path to map cache folder>.
      "bodies": {
        <anchor name>: {
          "activeImgDataset": <string>,   // The name on the currently active image data set.
          "activeDemDataset": <string>,   // The name on the currently active elevation data set.
          "imgDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
                                     // Use "offline" to only use cached data for this dataset.
              "layers": <string>,    // A comma,seperated list of WMS layers.
              "maxLevel": <int>      // The maximum quadtree depth to load.
            },
            ... <more image datasets> ...
          },
          "demDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
                                     // Use "offline" to only use cached data for this dataset.
              "layers": <string>,    // A comma,seperated list of WMS layers.
              "maxLevel": <int>      // The maximum quadtree depth to load.
            },
            ... <more elevation datasets> ...
          }
        },
        ... <more bodies> ...
      }
    }
  }
}
```

## Customize Shading

CosmoScout VR supports physically based rendering for each body separately.
Simple Lambertian shading is applied per default. You can choose custom BRDFs and parameterize them.
Here is an example configuration to set up a custom BRDF:

```json
"csp-lod-bodies": {
  ...
  "bodies": {
    ...
    "Earth": {
      ...
      "brdfHdr": {
        "source": "../share/resources/shaders/brdfs/oren-nayar.glsl",
        "properties": {
          "$rho": 0.2,
          "$sigma": 20.0
        }
      },
      "brdfNonHdr": {
        "source": "../share/resources/shaders/brdfs/oren-nayar_scaled.glsl",
        "properties": {
          "$rho": 1.0,
          "$sigma": 20.0
        }
      },
      "avgLinearImgIntensity": 0.0388402
    }
  }
}
```

A BRDF is defined by GLSL-like source code and represents a material with specific properties.
The properties are represented by key-variables and values.
The settings `brdfHdr` and `brdfNonHdr` set up the BRDFs to be used in HDR rendering and when lighting is enabled.
When HDR rendering and lighting is enabled, then the BRDF as defined by `brdfHdr` is used.
The last setting `avgLinearImgIntensity` adjusts the shading by dividing the fragments by the given value in HDR rendering.
The division by the average linear (!) intensity of the color maps leads to a more accurate representation of luminance in the scene.
To calculate the right value, you need to first gamma decode your image to linear space.
Then you need to calculate the average brightness and weight the pixels depending on their position.
Your image is likely an equirectangular projection so e.g. the pixels in the first row describe all the same point.
To make things easy: You can also just calculate the average brightness of an image,
normalize and raise the result to the power of gamma, e.g. 2.2 with a casual sRGB image.
This is quick and simple but also less accurate.
The visual appearance of the scene is not affected by this setting,
so feel free to skip it if you don't care about accurate luminance values.

### Adding a custom BRDF

There are some BRDFs already present that work well for most cases.
If you want to add a new BRDF, just add another file to the current repertoire and use it.
Let's look at the definition of the Lambertian BRDF:

```
// Lambertian reflectance to represent ideal diffuse surfaces.

// rho: Reflectivity of the surface in range [0, 1].

float $BRDF(vec3 N, vec3 L, vec3 V)
{
  return $rho / 3.14159265358979323846;
}
```

The signature of the BRDF has to be `float $BRDF(vec3 N, vec3 L, vec3 V)`, where `N` is the surface normal,
`L` is the direction of incident illumination and `V` is the direction of observation.
The given vectors are normalized. Properties are injected via the dollar sign syntax.
Besides the mentioned restrictions, the code shall be GLSL code.
Please include a description for each parameter and a reference to where the BRDF is defined if possible.