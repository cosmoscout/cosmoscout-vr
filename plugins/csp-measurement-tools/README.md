# Measurement tools for CosmoScout VR

A CosmoScout VR plugin with several tools for terrain measurements. For example, it supports the measurement of distances, height profiles, volumes or areas. The plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

* **Location Flag:** Displays the geographic coordinates and the address of the selected point as accurately as possible.
* **Landing Ellipse:** Puts an ellipse on the body, which is controllable by the center and two points. The center also has a Location Flag.
* **Path Measurement:** Enables the user to place a piecewise linear path on the surface of a body. Information about distance and elevation along the path will be shown.
* **Dip & Strike:** Used to measure the orientation of a surface feature. The dip shows the angle/steepness and the strike the orientation of the surface feature.
* **Polygon:** Measures the area and volume of an arbitrary polygon on surface with a Delaunay-mesh.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present some good starting values for your customization, however all are optional:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-measurement-tools": {
      "polygonHeightDiff": 1.002, // Maximum allowed relative height difference along an edge
      "polygonMaxAttempt": 5,     // Maximum mesh refinement operations
      "polygonMaxPoints": 1000,   // Maximum number of vertices in the generated mesh
      "polygonSleekness": 15      // Minimum allowed triangle corner angle
      "ellipseSamples": 360       // Number of elevation samples taken along the ellipse
      "pathSamples": 256          // Number of elevation samples taken between path control points
      "dipStrikes": []            // An array of currently active dip & strike tools.
      "ellipses": []              // An array of currently active ellipse tools.
      "flags": []                 // An array of currently active flag tools.
      "paths": []                 // An array of currently active path tools.
      "polygons": []              // An array of currently active polygon tools.
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
