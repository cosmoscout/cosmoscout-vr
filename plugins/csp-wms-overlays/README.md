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
      "mapCache": <string>,          // The path of a directory in which map textures should be cached.
      "capabilityCache": <string>,   // The path of a directory in which WMS capability documents should be cached.
      "useCapabilityCache": <string> // The cache mode for capability documents. For more details see section 'Capability cache'.
      "prefetch": <int>,             // The amount of images to prefetch in both directions of time.
      "maxTextureSize": <int>        // The length of the longer side of requested images in pixels.
      "bodies": {
      <anchor name>: {
        "activeServer": <string>,    // The name of the currectly active WMS server.
        "activeLayer": <string>,     // The name of the currectly active WMS layer.
        "activeStyle": <string>,     // The name of the currectly active layer style.
        "activeBounds": [            // The currect geographical bounds.
          <double>,                  // Minimum longitude
          <double>,                  // Maximum longitude
          <double>,                  // Minimum latitude
          <double>                   // Maximum latitude
        "wms": [
          <string>,                  // URL of a WMS server without a query string.
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

### Capability cache

Capability documents for WMS servers can be cached to speed up the initialization time of this plugin.
However, as most servers that supply current data frequently update the timespans for which data is available, this feature is disabled by default.
The caching can be controlled using the `useCapabilityCache` entry in the plugin configuration.
Available values are:

| Value | Description |
| :- | :- |
| `"never"` | Disables caching and requests new capability documents each time the plugin is started. This is the default. |
| `"updateSequence"` | Tries to check if the cached file is up to date using an update sequence number given in the capabilities. Requests a new capability document from the server if a newer document is available or no update sequence was given. This should only be used if all servers correctly update their update sequence on each change to the capabilities. |
| `"always"` | Always uses a cached document if one is available. This should only be used if you are sure the capabilities of the given servers haven't changed since the cache file was created. |

**More in-depth information and some tutorials will be provided soon.**
