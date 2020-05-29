# Stars for CosmoScout VR

A CosmoSout VR plugin which draws 3D-stars loaded from catalogues. For now, it supports the Tycho, the Tycho2 and the Hipparcos catalogue. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-stars": {
    "backgroundColor1": [<r>, <g>, <b>, <a>],
    "backgroundColor2": [<r>, <g>, <b>, <a>],
    "backgroundTexture1": <path to skybox file>,
    "backgroundTexture2": <path to skybox file>,
    "maxMagnitude": <float>                       // Example value:  15.0,
    "maxOpacity": <float>                         // Example value:  1.0,
    "maxSize": <float>                            // Example value:  3.0,
    "minMagnitude": <float>                       // Example value: -15.0,
    "minOpacity": <float>                         // Example value:  0.5,
    "minSize": <float>                            // Example value:  0.1,
    "scalingExponent": <float>                    // Example value:  3.0,
    "starTexture": <path to billboard file>,
    "hipparcosCatalog": <path to hip_main.dat>,
    "tycho2Catalog": <path to tyc2_main.dat>
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
