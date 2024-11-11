<!--
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Trajectories for CosmoScout VR

This plugin is providing HUD elements that display trajectories and markers for orbiting objects.
It also draws some flares around objects in HDR and non-HDR mode so that bodies are visible even if they are smaller than a pixel.

The non-HDR (LDR) flares are usually only drawn for the Sun.
They are drawn as a simple quad around the object with an exponential circle gradient so that the Sun looks like it is glowing.
HDR flares can be added for all objects.
They will be scaled so that they exactly cover the object.
They are faded in when the object gets very small on the screen and will stay above a certain size when the object gets even smaller.
Their luminance is also scaled with the distance to the camera so that they are faded out realistically.
The camera glare will make the dots glow in HDR mode.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```json
{
  ...
  "plugins": {
    ...
    "csp-trajectories": {
      "enableTrajectories": <boolean>,          // optional, default: true
      "enableLDRFlares": <boolean>,             // optional, default: true
      "enableHDRFlares": <boolean>,             // optional, default: true
      "enablePlanetMarks": <boolean>,           // optional, default: true
      "trajectories": {
        <object name>: {
          "color": [<red>, <green>, <blue>],      // between 0 and 1
          "drawDot": <boolean>,                   // optional, default: true
          "drawLDRFlare": <boolean>,              // optional, default: false
          "drawHDRFlare": <boolean>,              // optional, default: true
          "flareColor": [<red>, <green>, <blue>], // optional, default: 1, 1, 1
          "trail": {                                // optional
            "length": <float>,                      // in days
            "samples": <int>,
            "parent": <spice parent object name>
          }
        },
        ... <more trajectories> ...
      }
    }
  }
}
```
