# Atmospheres for CosmoScout VR

A CosmoScout VR plugin for drawing atmospheres around celestial bodies. It calculates single Mie- and Rayleigh scattering via raycasting in real-time. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values represent values for Earth:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-atmospheres": {
      "atmospheres": {
        <anchor name>: {
          "atmosphereHeight": 0.015,      // Relative atmosphere height compared to planet radius
          "mieAnisotropy": 0.76,
          "mieHeight": 0.000188679,
          "mieScatteringB": 133.56,
          "mieScatteringG": 133.56,
          "mieScatteringR": 133.56,
          "rayleighAnisotropy": 0,
          "rayleighHeight": 0.001257862,
          "rayleighScatteringB": 210.516,
          "rayleighScatteringG": 85.86,
          "rayleighScatteringR": 36.89,
          "sunIntensity": 15,
          "cloudHeight": 0.001,          // Relative cloud layer altitude
          "cloudTexture": "../share/resources/textures/earth-clouds.jpg" // Optional
        },
        ... <more atmospheres> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
