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

## MIT License

Copyright (c) 2019 German Aerospace Center (DLR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
