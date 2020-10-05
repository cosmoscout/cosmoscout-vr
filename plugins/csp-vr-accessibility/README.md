# VR Accessibility for CosmoScout VR

A CosmoScout VR plugin collection of settings to reduce cybersickness symptoms.
The collection contains a floor grid to give the user a solid plane of reference inside the simulation.
The grid can be enabled and disabled, as well as configured as stated below in [Configuration](#configuration).

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-vr-accessibility": {
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

Most of the configuration (all options, except for `falloff` and `texture`) is also available at runtime in the "VR Accessibility" tab in the settings menu.
