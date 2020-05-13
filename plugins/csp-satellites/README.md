# Satellites for CosmoScout VR

A CosmoScout VR plugin which draws GTLF models at positions based on SPICE data. It uses physically based rendering for surface shading. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-satellites": {
      "satellites": {
        <anchor name>: {
          "modelFile": <path to model file>,     // .glb or .gltf
          "environmentMap": <path to env map>,   // .dds, .ktx or .kmg
          "size": <float>,                       // The radius in meters.
          "transformation": {                    // optional
              "translation": [<x>, <y>, <z>],
              "rotation": [<x>, <y>, <z>, <w>],  // Quaternion
              "scale": <float>
          }
        },
        ... <more satellites> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
