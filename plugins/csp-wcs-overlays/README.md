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

### Technical Notes
This tries to outline the basic logic / execution flow of the WCS plugin.

`Plugin.cpp` reads all available WCS Servers from `settings.json` and instantiates `WebCoverageServices` through `Plugin::onLoad`. If a defined anchor name is deemed active, all corresponding WCS Servers are added to the sidebar dropdown. The connection is set in `Plugin::init`.
The previously empty dropdown then triggers the `wcsOverlays.setServer` callback, defined in `Plugin::registerSidebarCallbacks` which in turn sets the currently active WCS Server by calling `Plugin::setWCSServer`.

`Plugin::setWCSServer` tries to find a `WebCoverageService` that matches a given title. If a server is found, all available coverages are added to the coverage select.

### `WebCoverageService`
On instantiation the services tries to request its corresponding capabilities (either from cache of from its url `WebCoverageService::getGetCapabilitiesUrl`).  
After the XML document was loaded, all available coverages are parsed in `WebCoverageService::parseCoverages`, and made available through `WebCoverageService::getCoverages`.

Each coverage is described by a `WebCoverage` object.

### `WebCoverage`
A `WebCoverage` is a "layer" containing actual coverage data. Further information about a coverage is requested by calling `WebCoverage::update`.  
The call to updated loads additional coverage details like the supported time periods. The call is automatically handled through `Plugin::setWCSCoverage`.

A dedicated update call ensures that very short time periods (e.g. 5 minutes and under) don't come out of scope.  
Example: A coverage with 5 minute time intervals has a moving lower and upper time interval bound that throws an exception if an interval is requested that lies not inside them.  

### `TextureOverlayRenderer`
Displays the selected coverage of a selected service (`TextureOverlayRenderer::setActiveWCS`).  
To get the actual data a call to `WebCoveraageTextureLoader` is made in `TextureOverlayRenderer::setLayer`.

The request is configured by calling `TextureOverlayRenderer::getRequest`.  
This defines the format, maximum texture size and possible sub-sets of the coverage.

After the was loaded (see next heading) it is displayed through `TextureOverlayRenderer::Do`.

### `WebCoverageTextureLoader`
Does the actual loading of the coverage data.  
`WebCoverageTextureLoader::loadTexture` is used to load a given texture and is initially called when selecting a coverage.  
If the coverage has time intervals defined, `WebCoverageTextureLoader::loadTextureAsync` is automatically called 2 * pre-load count (in each direction e.g. the previous and next interval).

Textures are loaded using Gdal (`common/GDALReader.hpp`). If a complete (i.e. not subsetted) coverage is requested, the file is saved to disk.  
In all other cases an in-memory file system is used see `GDALReader::ReadGrayScaleTexture` (`vsimem` part).  
The GDAL reader also handles the different types of data returned by the coverage.
