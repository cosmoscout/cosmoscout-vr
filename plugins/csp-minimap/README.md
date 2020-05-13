# Minimap for CosmoScout VR

A CosmoScout VR plugin which shows a 2D-Minimap in the user interface. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values are just some examples, feel free to add your own maps:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-minimap": {
      "defaultMap": {
        "projection": "equirectangular",
        "type": "wms",
        "url": "https://geoservice.dlr.de/eoc/basemap/wms?",
        "config": {
          "attribution": "&copy; DLR",
          "layers": "ne:ne_graticules",
          "BGCOLOR": "0x000000"
        }
      },
      "maps": {
        "Earth": {
          "projection": "mercator",
          "type": "wmts",
          "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
          "config": {
            "attribution": "&copy; OpenStreetMap contributors"
          }
        },
        "Moon": {
          "projection": "mercator",
          "type": "wmts",
          "url": "https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/opm-moon-basemap-v0-1/all/{z}/{x}/{y}.png",
          "config": {
            "attribution": "&copy; OpenPlanetary"
          }
        },
        "Mars": {
          "projection": "mercator",
          "type": "wmts",
          "url": "https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/opm-mars-basemap-v0-2/all/{z}/{x}/{y}.png",
          "config": {
            "attribution": "&copy; OpenPlanetary"
          }
        }
      }
    }
    ...
  }
}
```

### Some other examples:

#### DLR EOC Maps
```javascript
"projection": "equirectangular",
"type": "wms",
"url": "https://geoservice.dlr.de/eoc/basemap/wms?",
"config": {
  "attribution": "&copy; DLR",
  "layers": "basemap,tm:tm_worldborders"
}
```

#### Sentinel 2 cloudless by EOX
```javascript
"projection": "mercator",
"type": "wmts",
"url": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg",
"config": {
  "attribution": "Sentinel-2 cloudless - <a href='https://s2maps.eu/' target='_blank'>s2maps.eu</a> by <a href='https://eox.at/' target='_blank'>EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2019)"
}
```

#### NASA MarsTrek Viking
```javascript
"projection": "equirectangular",
"type": "wmts",
"url": "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
"config": {
  "attribution": "Map tiles from <a href='https://trek.nasa.gov' target='_blank'>trek.nasa.gov</a>"
}
```

## MIT License

Copyright (c) 2020 German Aerospace Center (DLR)

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
