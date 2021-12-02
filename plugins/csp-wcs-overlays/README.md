# WCS overlay for CosmoScout VR

A CosmoSout VR plugin which overlays existing planets with time dependent WCS based textures.
The planets have to be rendered by another plugin such as `csp-simple-bodies` or `csp-lod-bodies`.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-wcs-overlays": {
      "coverageCache": <string>,     // The path of a directory in which coverage textures should be cached.
      "capabilityCache": <string>,   // The path of a directory in which WCS capability documents should be cached.
      "useCapabilityCache": <string> // The cache mode for capability documents. For more details see section 'Capability cache'.
      "wcsRequestFormat": <string>,  // Mime Format used in WCS requests, e.g. image/tiff, application/x-netcdf, etc. 
      "prefetch": <int>,             // The amount of images to prefetch in both directions of time.
      "maxTextureSize": <int>        // The length of the longer side of requested images in pixels.
      "bodies": {
      <anchor name>: {
        "activeServer": <string>,    // The name of the currectly active WCS server.
        "activeLayer": <string>,     // The name of the currectly active WCS coverage.
        "activeBounds": [            // The currect geographical bounds.
          <double>,                  // Minimum longitude
          <double>,                  // Maximum longitude
          <double>,                  // Minimum latitude
          <double>                   // Maximum latitude
        "wcs": [
          <string>,                  // URL of a WCS server without a query string.
          ... <more WCS URLs> ...
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

Capability documents for WCS servers can be cached to speed up the initialization time of this plugin.
However, as most servers that supply current data frequently update the timespans for which data is available, this feature is disabled by default.
The caching can be controlled using the `useCapabilityCache` entry in the plugin configuration.
Available values are:

| Value | Description |
| :- | :- |
| `"never"` | Disables caching and requests new capability documents each time the plugin is started. This is the default. |
| `"updateSequence"` | Tries to check if the cached file is up to date using an update sequence number given in the capabilities. Requests a new capability document from the server if a newer document is available or no update sequence was given. This should only be used if all servers correctly update their update sequence on each change to the capabilities. |
| `"always"` | Always uses a cached document if one is available. This should only be used if you are sure the capabilities of the given servers haven't changed since the cache file was created. |
