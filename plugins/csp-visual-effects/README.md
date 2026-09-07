<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-FileCopyrightText: Xor <https://www.shadertoy.com/view/4sycRW>
SPDX-License-Identifier: CC-BY-NC-SA-3.0
 -->

# Visual Effects for CosmoScout VR

This plugin allows for visual effects put onto an object.
The visual effect is displayed on a quad placed on the object.
Currently it allows for visualising solar flares for example on the sun.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present some good starting values for your customization:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-visual-effects": {
      "solarFlares": {
        "Sun": {}
      }
    }
  }
}
```
