<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# VR Accessibility for CosmoScout VR

A CosmoScout VR plugin collection of settings to reduce cybersickness symptoms.
The collection contains a floor grid to give the user a solid plane of reference inside the simulation, and a vignette to focus the users view on the center of the Headset lenses.
Both can be enabled and disabled, as well as configured as stated below in [Configuration](#configuration).

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-vr-accessibility": {
      "grid": {
        "enabled": bool,   // Toggle whether the Grid should be visible.
        "size": float,     // Modifier to scale the texture (grid mesh size).
        "offset": float,   // The Vertical offset of the grid in meters.
        "extent": float,   // The size of the plane, the grid is drawn on.
        "texture": string, // The path to the texture used for the grid ("../share/resources/textures/gridCrossSmall.png", ".../gridCrossSmall.png", ".../gridCentered.png").
        "alpha": float,    // The transparency of the grid.
        "color": string    // The color of the grid (as a hex-code string, i.e. #ffffff).
      },
      "vignette": {
        "enabled": bool,                 // Toggle whether the FoV Vignette should be visible.
        "debug": bool,                   // Toggle whether the Vignette is always drawn with its minimum radius.
        "radii": [float, float],         // The inner and outer radii of the vignette.
        "color": string,                 // The color of the vignette (as a hex-code string).
        "fadeDuration": float,           // The time it takes for the animation to fade in the vignette in seconds.
        "fadeDeadzone": float,           // The time of movement above the velocity threshold that is needed before the animation is played in seconds.
        "velocityThresholds": [float, float], // The lower and upper thresholds between which the vignette will fade-in or fade-out. 
        "useDynamicRadius": bool,        // Toggle whether to use the dynamic vignette radius, or the fade animation with fixed radii.
        "useVerticalOnly": bool          // Toggle whether to only draw the vignette horizontally and keep the sides unobstructed.
      }
    }
  }
}
```

Most of the configuration (all options, except for Grid's `offset` and `texture`) is also available at runtime in the "VR Accessibility" tab in the settings menu.

### Example Configuration

Here is an example configuration with both features enabled and some default values:

```javascript
"csp-vr-accessibility": {
      "grid": {
        "enabled": true,
        "size": 0.5,
        "offset": -1.80,
        "extent": 10.0,
        "texture": "../share/resources/textures/gridCrossLarge.png",
        "alpha": 1.0,
        "color": "#ffffff"
      },
      "vignette": {
        "enabled": true,
        "debug": false,
        "radii": [0.5, 1.0],
        "color": "#000000",
        "fadeDuration": 1.0,
        "fadeDeadzone": 0.5,
        "velocityThresholds": [0.2, 10.0],
        "useDynamicRadius": true,
        "useVerticalOnly": false
      }
    },
```
