# Stars for CosmoScout VR

A CosmoSout VR plugin which draws 3D-stars loaded from catalogues. For now, it supports the Tycho, the Tycho2 and the Hipparcos catalogue. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-stars": {
    "backgroundColor1": [<r>, <g>, <b>, <a>],
    "backgroundColor2": [<r>, <g>, <b>, <a>],
    "backgroundTexture1": <path to skybox file>,
    "backgroundTexture2": <path to skybox file>,
    "maxMagnitude": <float>                       // Example value:  15.0,
    "maxOpacity": <float>                         // Example value:  1.0,
    "maxSize": <float>                            // Example value:  3.0,
    "minMagnitude": <float>                       // Example value: -15.0,
    "minOpacity": <float>                         // Example value:  0.5,
    "minSize": <float>                            // Example value:  0.1,
    "scalingExponent": <float>                    // Example value:  3.0,
    "starTexture": <path to billboard file>,
    "hipparcosCatalog": <path to hip_main.dat>,
    "tycho2Catalog": <path to tyc2_main.dat>
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
