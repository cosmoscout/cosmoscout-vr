# WMS overlay for CosmoScout VR

A CosmoSout VR plugin which overlays existing planets with time dependent WMS based textures.
The planets have to be rendered by another plugin such as `csp-simple-bodies` or `csp-lod-bodies`.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-wms-overlays": {
      "mapCache": <string>,        // The path of a directory in which map textures should be cached.
      "capabilitycache": <string>, // The path of a directory in which WMS capabilities documents should be cached.
      "prefetch": <int>,           // The amount of images to prefetch in both directions of time.
      "maxTextureSize": <int>      // The length of the longer side of requested images in pixels.
      "bodies": {
      <anchor name>: {
        "activeServer": <string>,  // The name of the currectly active WMS server.
        "activeLayer": <string>,   // The name of the currectly active WMS layer.
        "activeStyle": <string>,   // The name of the currectly active layer style.
        "activeBounds": [          // The currect geographical bounds.
          <double>,                // Minimum longitude
          <double>,                // Maximum longitude
          <double>,                // Minimum latitude
          <double>                 // Maximum latitude
        "wms": [
          <string>,                // URL of a WMS server without a query string.
          ... <more WMS URLs> ...
        ]
      },
      ... <more bodies> ...
    },
    ...
  },
  ...
}
```

**More in-depth information and some tutorials will be provided soon.**

## MIT License
