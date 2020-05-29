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

Copyright (c) 2019 German Aerospace Center (DLR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
