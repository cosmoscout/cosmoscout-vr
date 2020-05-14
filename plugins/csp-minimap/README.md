# Minimap for CosmoScout VR

A CosmoScout VR plugin which shows a 2D-Minimap in the user interface.

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
