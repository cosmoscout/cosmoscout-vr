# Floor Grid for CosmoScout VR

A CosmoScout VR plugin which can draw a simple grid around the observer to reduce cybersickness symptoms.
The grid can be enabled and disabled, as well as configured as stated below in [Configuration](#configuration).

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```json
{
  ...
  "plugins": {
    ...
    "csp-floor-grid": {
      "enabled": true,
      "size": 1.0,
      "offset": -1.80,
      "falloff": 32.0,
      "texture": "../share/resources/textures/gridCrossSmall.png",
      "alpha": 1.0,
      "color": "#ffffff"
    }
  }
}
```

Most of the configuration is also available at runtime in the "Floor Grid" tab in the settings menu.

| Setting   | Available at Runtime | Default Value                                      | Description                                    | Comment                                                                                                      |
|-----------|----------------------|----------------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `enabled` | :heavy_check_mark:   | `true`                                             | Toggle whether the Grid should be visible      | Recommended `true` for VR configurations, `false` for desktop configurations.                                |
| `size`    | :heavy_check_mark:   | `1.0`                                              | Modifier to scale the texture (grid mesh size) | At runtime, the mesh size can only be adjusted in multiples of 2 (double or half size, etc.).                |
| `offset`  | :heavy_check_mark:   | `-1.80`                                            | Vertical offset of the grid in meters          | Negative values are moving the grid downward. At runtime, the offset can be adjusted within 0 and -3 meters. |
| `falloff` | :x:                  | `32.0`                                             | Size of the plane, the grid is drawn on        | To work smoothly together with the size setting, powers of 2 (2<sup>n</sup>) are recommended.                |
| `texture` | :x:                  | `"../share/resources/textures/gridCrossSmall.png"` | Path to the texture used for the grid          | additional available textures: `gridCrossSmall.png`, `gridCentered`.                                         |
| `alpha`   | :heavy_check_mark:   | `1.0`                                              | Modifier to set the transparency of the grid   |                                                                                                              |
| `color`   | :heavy_check_mark:   | `#ffffff`                                          | Value to color the grid                        | Any color hex-code possible, modifiable through a color picker at runtime.                                   |
