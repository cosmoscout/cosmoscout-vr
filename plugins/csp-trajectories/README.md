# Trajectories for CosmoScout VR

A CosmoSout VR plugin which draws trajectories of celestial bodies and spacecrafts. The color, length, number of samples and reference frame can be configured. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-trajectories": {
      "trajectories": {
        <anchor name>: {
          "color": [<red>, <green>, <blue>], // floating point values between 0 and 1
          "drawFlare": <boolean>,            // optional
          "drawDot": <boolean>,              // optional
          "trail": {                         // optional
            "length": <float>,               // in days
            "samples": <int>,
            "parentCenter": <spice parent center name>,
            "parentFrame": <spice parent frame name>
          }
        },
        ... <more trajectories> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
