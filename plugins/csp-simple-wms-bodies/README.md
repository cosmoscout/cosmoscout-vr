# Simple WMS bodies for CosmoScout VR

A CosmoSout VR plugin which renders simple spherical celestial bodies using a background surface texture and overlays it with time dependent WMS based textures. The bodies are drawn as an ellipsoid with an equirectangular texture.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-simple-wms-bodies": {
	  "mapCache": <string>,           // The path to map cache folder.
      "bodies": {
        <anchor name>: {
          "gridResolutionX": <int>,   // The x resolution of the body grid.
          "gridResolutionY": <int>,   // The y resolution of the body grid.
          "texture": <string>,        // The path to background surface texture. The texture from the WMS image will be overlaid.
          "activeWms": <string>,      // The name of the currectly active WMS data set.
          "wms": {
            <dataset name> : {
              "copyright": <string>,  // The copyright holder of the data set (also shown in the UI).
              "url": <string>,        // The URL of the map server including the "SERVICE=wms" parameter.
              "width": <int>,         // The width of the WMS image.
              "height": <int>,        // The height of the WMS image.
              "time": <string>,       // Time intervals of WMS images, optional.
              "layers": <string>,     // A comma,separated list of WMS layers.
              "preFetch": <int>       // The amount of textures that gets pre-fetched in every time direction, optional.
            },
            ... <more WMS datasets> ...
          }
        },
        ... <more bodies> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**

## MIT License
