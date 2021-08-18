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
      "gridEnabled": <bool>,    // Toggle whether the Grid should be visible.
      "gridSize": <float>,      // Modifier to scale the texture (grid mesh size).
      "gridOffset": <float>,    // The Vertical offset of the grid in meters.
      "gridFalloff": <float>,   // The size of the plane, the grid is drawn on.
      "gridTexture": <string>,  // The path to the texture used for the grid ("../share/resources/textures/gridCrossSmall.png", ".../gridCrossSmall.png", ".../gridCentered.png").
      "gridAlpha": <float>,     // The transparency of the grid.
      "gridColor": <sting>      // The color of the grid (as a hex-code string).
      "vignetteEnabled": <bool>, // Toggle whether the FoV Vignette should be visible.
      "vignetteDebug": <bool>,    // Toggle whether the Vignette is always drawn with its minimum radius.
      "vignetteInnerRadius": <float>, // The inner radius of the vignette where the inside is 100% transparent.
      "vignetteOuterRadius": <float>, // The outer radius of the vignette after which the vignette is 100% opaque.
      "vignetteColor": <string>,  // The color of the vignette (as a hex-code string).
      "vignetteFadeDuration": <float>, // The time it takes for the animation to fade in the vignette in seconds.
      "vignetteFadeDeadzone": <float>, // The time of movement above the velocity threshold that is needed before the animation is played in seconds.
      "vignetteLowerVelocityThreshold": <float>, // The lower threshold below which the vignette will not display (relative to Spice frame from 0..~10). 
      "vignetteUpperVelocityThreshold": <float>, // The upper threshold above which the vignette is always at ist minimum radius (relative to Spice frame from 0..~10).
      "vignetteUseDynamicRadius": <bool>, // Toggle whether to use the dynamic vignette radius, or the fade animation with fixed radii.
      "vignetteUseVerticalOnly": <bool> // Toggle whether to only draw the vignette horizontally and keep the sides unobstructed.
    }
  }
}
```

Most of the configuration (all options, except for `gridFalloff` and `gridTexture`) is also available at runtime in the "VR Accessibility" tab in the settings menu.
