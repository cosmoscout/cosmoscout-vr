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