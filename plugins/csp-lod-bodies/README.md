# Level-of-detail bodies for CosmoScout VR

A CosmoScout VR plugin wich draws level-of-detail planets and moons. This plugin supports the visualization of entire planets in a 1:1 scale. The data is streamed via Web-Map-Services (WMS) over the internet. A dedicated MapServer is required to use this plugin.


## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-lod-bodies": {
      "maxGPUTilesColor": <int>,     // The maximum allowed colored tiles.
      "maxGPUTilesGray": <int>,      // The maximum allowed gray tiles.
      "maxGPUTilesDEM": <int>,       // The maximum allowed elevation tiles.
      "mapCache": <string>,          // The path to map cache folder>.
      "bodies": {
        <anchor name>: {
          "activeImgDataset": <string>,   // The name on the currently active image data set.
          "activeDemDataset": <string>,   // The name on the currently active elevation data set.
          "imgDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "format": <string>,    // "Float32", "UInt8" or "U8Vec3".
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
              "layers": <string>,    // A comma,seperated list of WMS layers.
              "maxLevel": <int>      // The maximum quadtree depth to load.
            },
            ... <more image datasets> ...
          },
          "demDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "format": <string>,    // "Float32", "UInt8" or "U8Vec3".
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
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

**More in-depth information and some tutorials will be provided soon.**
