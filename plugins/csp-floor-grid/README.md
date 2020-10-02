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
      "enabled": true,                                              // Toggle whether the Grid should be visible
      "size": 1.0,                                                  // Modifier to scale the texture (grid mesh size)
      "offset": -1.80,                                              // The Vertical offset of the grid in meters
      "falloff": 32.0,                                              // The size of the plane, the grid is drawn on
      "texture": "../share/resources/textures/gridCrossSmall.png",  // The path to the texture used for the grid (also available: `gridCrossSmall.png`, `gridCentered`)
      "alpha": 1.0,                                                 // The transparency of the grid
      "color": "#ffffff"                                            // The color of the grid (as a hex-code string)
    }
  }
}
```

Most of the configuration (all options, except for `falloff` and `texture`) is also available at runtime in the "Floor Grid" tab in the settings menu.
