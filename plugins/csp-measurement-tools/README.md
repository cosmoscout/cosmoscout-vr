# Measurement tools for CosmoScout VR

A CosmoScout VR plugin with several tools for terrain measurements. For example, it supports the measurement of distances, height profiles, volumes or areas.

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
