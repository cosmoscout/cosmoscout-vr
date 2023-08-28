<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# WFS Overlays for CosmoScout VR

A CosmoScout VR plugin which loads data from a server and draws it. The specific options within a certain server are shown in a list so the user can select what to render. Some other features like the size and color of the features can be adjusted.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.

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
* The main properties related to the size of the points and lines (maximum and minimum values, step value) can be easily modified through the .html file.
* For points, it is possible to load any texture (image). We use a circled-shape. 
* Regarding the lines, there is an option at the settings sidetab that allows us to choose if we want to use an interpolation Bézier-curve based method to get a smoother rendering.
