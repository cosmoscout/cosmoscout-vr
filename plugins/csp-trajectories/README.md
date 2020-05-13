# Trajectories for CosmoScout VR

A CosmoSout VR plugin which draws trajectories of celestial bodies and spacecrafts. The color, length, number of samples and reference frame can be configured.

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
