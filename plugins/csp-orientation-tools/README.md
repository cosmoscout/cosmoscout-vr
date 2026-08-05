<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Orientation Tools for CosmoScout VR

This plugin allows for adding tools to objects for visualization.
It supports the visualisation of x, y and z of a cartesian coordinate system as arrows (similar to Blender).
It also allows to visualize axes like the rotation axis of the Earth.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present some good starting values for your customization:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-orientation-tools": {
      "arrows": {
        "Moon": {
          "size": 0.1
        }
      },
      "axes": {
        "Earth": {
          "size": 15000,
          "color": [
            1.0,
            1.0,
            0.5
          ],
          "disableX": true,
          "disableZ": true
        }
      }
    }
  }
}