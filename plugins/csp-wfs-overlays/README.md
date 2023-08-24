<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# WFS Overlays for CosmoScout VR

A CosmoScout VR plugin which loads data from a server and draws it. The specific options within a certain server are shown in a list so the user can select what to render. Some other features like the size and color can be adjusted.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present some good starting values for your customization:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-wfs-overlays": {
      "enabled": true,
      "wfs": [
        "https://maps.dwd.de/geoserver/dwd/wfs",
        "https://geoservice.dlr.de/eoc/basemap/wfs"
      ],
      "interpolation": false
    }
  }
}
```
## Usage
Once the plugin is loaded, you can select the server and its specific data to render. For now, only the DWD and DLR geoservers are included, but any other Web Feature Services should work by just adding its URL at the settings.
* The main properties related to the size of the points and lines (maximum and minimum values, step value) can easily be modified through the .html file.
* We can also select from the settings sidetab the way we want to render the features:
  - For points, it is possible to draw them squared or rounded. 
  - Regarding the lines, there is an option to use an interpolation Bézier-curve based method to get an smoother rendering.
