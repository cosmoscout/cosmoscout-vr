# Floor Grid for CosmoScout VR

A CosmoScout VR plugin which can draw a simple grid around the observer to reduce cybersickness symptoms.
The grid can be enabled and disabled, as well as configured as stated below in [Configuration](#configuration).

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-floor-grid": {
      "enabled": <bool>,    // Toggle whether the Grid should be visible.
      "size": <float>,      // Modifier to scale the texture (grid mesh size).
      "offset": <float>,    // The Vertical offset of the grid in meters.
      "falloff": <float>,   // The size of the plane, the grid is drawn on.
      "texture": <string>,  // The path to the texture used for the grid ("../share/resources/textures/gridCrossSmall.png", ".../gridCrossSmall.png", ".../gridCentered.png").
      "alpha": <float>,     // The transparency of the grid.
      "color": <sting>      // The color of the grid (as a hex-code string).
    }
  }
}
```

Most of the configuration (all options, except for `falloff` and `texture`) is also available at runtime in the "Floor Grid" tab in the settings menu.
