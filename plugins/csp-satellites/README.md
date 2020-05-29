# Satellites for CosmoScout VR

A CosmoScout VR plugin which draws GTLF models at positions based on SPICE data. It uses physically based rendering for surface shading.

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
