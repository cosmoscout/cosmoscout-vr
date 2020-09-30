# Level-of-Detail Bodies for CosmoScout VR

A CosmoScout VR plugin which draws level-of-detail planets and moons. This plugin supports the visualization of entire planets in a 1:1 scale. The data is streamed via Web-Map-Services (WMS) over the internet. A dedicated MapServer is required to use this plugin.

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-lod-bodies": {
      "maxGPUTilesColor": <int>,     // The maximum allowed colored tiles.
      "maxGPUTilesGray": <int>,      // The maximum allowed gray tiles.
      "maxGPUTilesDEM": <int>,       // The maximum allowed elevation tiles.
      "mapCache": <string>,          // The path to map cache folder>.
      "bodies": {
        <anchor name>: {
          "activeImgDataset": <string>,   // The name on the currently active image data set.
          "activeDemDataset": <string>,   // The name on the currently active elevation data set.
          "imgDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "format": <string>,    // "Float32", "UInt8" or "U8Vec3".
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
                                     // Use "offline" to only use cached data for this dataset.
              "layers": <string>,    // A comma,seperated list of WMS layers.
              "maxLevel": <int>      // The maximum quadtree depth to load.
            },
            ... <more image datasets> ...
          },
          "demDatasets": {
            <dataset name>: {        // The name of the data set as shown in the UI.
              "copyright": <string>, // The copyright holder of the data set (also shown in the UI).
              "format": <string>,    // "Float32", "UInt8" or "U8Vec3".
              "url": <string>,       // The URL of the mapserver including the "SERVICE=wms" parameter.
                                     // Use "offline" to only use cached data for this dataset.
              "layers": <string>,    // A comma,seperated list of WMS layers.
              "maxLevel": <int>      // The maximum quadtree depth to load.
            },
            ... <more elevation datasets> ...
          }
        },
        ... <more bodies> ...
      }
    }
  }
}
```

## Setting up the MapServer on Ubuntu 20.04 

This guide will most likely work for newer versions of Ubuntu as well. A key requirement is at least version 6.3.0 of [PROJ](https://proj.org/). For Ubuntu 20.04 this is in the official repositories, for older versions, you may try to add the [UbuntuGIS](https://launchpad.net/~ubuntugis) repository.

If you are using Windows you can use the Windows Subsystem for Linux (WSL) to setup the server on your Windows machine. A detailed guide can be found [here](https://docs.microsoft.com/en-gb/windows/wsl/install-win10). WSL1 is sufficient and you don't need to upgrade to WSL2. We recommend installing the Ubuntu 20.04 image.

### Installing the MapServer

This tutorial will install the `mapserv` CGI script and Apache2 as a webserver.
Of course you could use any other server - also the Apache configuration is very basic, but it will serve as a starting point.

First install the required packages:

```
sudo apt-get install apache2 apache2-bin apache2-utils cgi-mapserver mapserver-bin \
                     mapserver-doc libmapscript-perl libapache2-mod-fcgid
```

Then enable `cgi` and `fastcgi` of Apache:

```
sudo a2enmod cgi fcgid
```

Then add `/usr/lib/cgi-bin` directory to Apache. To do this, add the following lines to the end of the Apache2 configuration file (e.g. `/etc/apache2/sites-available/000-default.conf`):

```
ScriptAlias /cgi-bin/ /usr/lib/cgi-bin/
<Directory "/usr/lib/cgi-bin/">
        AllowOverride All
        Options +ExecCGI -MultiViews +FollowSymLinks
        AddHandler fcgid-script .fcgi
        Require all granted
</Directory>
```

Finally, restart `apache2` Daemon.

```
sudo service apache2 restart
```

To check the MapServer installation, you can run this command:

```
mapserv -v 
```

Also, navigating with a web browser to http://localhost/cgi-bin/mapserv?service=wms&version=1.1.1&REQUEST=GetCapabilities should give the following message:
```
msCGILoadMap(): Web application error. CGI variable "map" is not set.
```

### Configuring the MapServer

First, create a directory where you want to store your datasets.
For this tutorial, we will assume that all data will be stored in `/storage/mapserver-datasets`.
In this directory, we have to create several configuration files.

#### The `epsg` file

CosmoScout VR requests map data in the [HEALPix projection](https://proj.org/operations/projections/healpix.html).
To make this non-standard projection available, we have to define custom epsg codes.
To do this, you have to download an official `epsg` file, put it into our datasets directory (e.g. `/storage/mapserver-datasets`) and add a few lines to it.
You can download the `epsg` file which is distributed with PROJ 5.2.0 (just[ download the zip](https://github.com/OSGeo/PROJ/releases/download/5.2.0/proj-5.2.0.zip) and extract the `epsg` file from the `/proj-5.2.0/nad/` directory). 

Then, add the following lines to the end of this file.

```
# custom rotated and scaled HEALPix, magic number is sqrt(2) * 2/pi
<900914> +proj=healpix +lon_0=0 +x_0=2.5 +y_0=2.5 +a=0.900316316157106 +rot_xy=45 +no_defs <>
# standard HEALPix on unit sphere
<900915> +proj=healpix +a=1 +b=1 <>
```

If anybody knows a better way to define custom epsg codes with current versions of PROJ, please [open an issue](https://github.com/cosmoscout/cosmoscout-vr/issues).


#### The `meta.map` file

Add a file called `meta.map` in your datasets directory (e.g. `/storage/mapserver-datasets`).
The name of this file is somewhat arbitrary, but it will be referenced further down in this tutorial, so use the same name everywhere.
This configuration file is very basic, for a complete list of options, please refer to the [official documentation](https://www.mapserver.org/mapfile/).

```bash
MAP
  NAME "CosmoScout VR Maps"
  STATUS ON
  EXTENT -180 -90 180 90
  SIZE 800 400

  # This tells the MapSever to look for PROJ init files next to this map file.
  # This way we can use our custom epsg codes.
  CONFIG "PROJ_LIB" "."

  PROJECTION
    "init=epsg:4326"
  END

  # This format will be requested by CosmoScout VR for elevation data.
  OUTPUTFORMAT
    NAME "tiffGray"
    DRIVER "GDAL/GTiff"
    IMAGEMODE FLOAT32
    EXTENSION "tiff"
    FORMATOPTION "COMPRESS=LZW"
  END

  # This format will be requested by CosmoScout VR for monochrome imagery data.
  OUTPUTFORMAT
    NAME "pngGray"
    DRIVER "GDAL/PNG"
    IMAGEMODE BYTE
    EXTENSION "png"
  END

  # This format will be requested by CosmoScout VR for color imagery data.
  OUTPUTFORMAT
    NAME "pngRGB"
    DRIVER "GD/PNG"
    IMAGEMODE RGB
    EXTENSION "png"
  END

  WEB
    METADATA 
      WMS_TITLE           "CosmoScout-VR-WMS-Server"
      WMS_ONLINERESOURCE  "localhost/cgi-bin/mapserv?"
      WMS_ENABLE_REQUEST  "*" 
      WMS_SRS             "EPSG:4326 EPSG:900914 EPSG:900915"
    END
  END

  INCLUDE "earth/bluemarble/bluemarble.map"
  INCLUDE "earth/naturalearth/naturalearth.map"
  INCLUDE "earth/etopo1/etopo1.map"
END
```

The three files included at the bottom of this file will be created in the next step of this tutorial.

#### Adding some Datasets

Now that the configuration files are in place, we need to download some datasets. Here are some examples which we will use for this tutorial:

* **NASA's Blue Marble:** There are some great images of Earth available at NASA's [visible earth catalog](https://visibleearth.nasa.gov/collection/1484/blue-marble).
For this tutorial we will download the one-big-jpeg variant of the [august dataset](https://visibleearth.nasa.gov/images/73776/august-blue-marble-next-generation-w-topography-and-bathymetry/73783l).
Here is the direct download link: https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.topo.bathy.200408.3x21600x10800.jpg. Save this file as `/storage/mapserver-datasets/earth/bluemarble/bluemarble.jpg`.
* **Natural Earth:** We will use a map from [naturalearthdata.com](http://naturalearthdata.com/).
Go and grap a map from [here (with Shaded Relief, Water and Drainages)](http://www.naturalearthdata.com/downloads/10m-raster-data/10m-natural-earth-1/) for example.
These maps already include surface shading which is not particularly useful, but they will serve our purpose here.
Extract the containing GeoTiff file to `/storage/mapserver-datasets/earth/naturalearth/NE1_HR_LC_SR_W_DR.tif`.
* **ETOPO1:** As an elevation dataset, we will use the [ETOPO1 Global Relief Model](https://www.ngdc.noaa.gov/mgg/global/global.html). Get the zip from [here](https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/cell_registered/georeferenced_tiff/ETOPO1_Ice_c_geotiff.zip) and extract the contained GeoTiff to `/storage/mapserver-datasets/earth/etopo1/ETOPO1_Ice_c_geotiff.tif`.

Now we need to create the map files which we already referenced in `meta.map`. Create the following three files:

##### `/storage/mapserver-datasets/earth/bluemarble/bluemarble.map`
```bash
LAYER
  NAME "earth.bluemarble.rgb"
  STATUS ON
  TYPE RASTER
  DATA "earth/bluemarble/bluemarble.jpg"

  # Decreasing the oversampling factor will increase performance but reduce quality.
  PROCESSING "OVERSAMPLE_RATIO=10"
  PROCESSING "RESAMPLE=BILINEAR"

  # The JPEG file obviously does not contain any projection information.
  # Therefore we have to give the extent and projection here.
  EXTENT -180 -90 180 90

  PROJECTION
    "init=epsg:4326"
  END

  METADATA
    WMS_TITLE "earth.bluemarble.rgb"
  END
END
```

##### `/storage/mapserver-datasets/earth/naturalearth/naturalearth.map`
```bash
LAYER
  NAME "earth.naturalearth.rgb"
  STATUS ON
  TYPE RASTER
  DATA "earth/naturalearth/NE1_HR_LC_SR_W_DR.tif"

  # Decreasing the oversampling factor will increase performance but reduce quality.
  PROCESSING "OVERSAMPLE_RATIO=10"
  PROCESSING "RESAMPLE=BILINEAR"

  # The GeoTiff is fully geo-referenced, so we can just use AUTO projection here.
  PROJECTION
    AUTO
  END

  METADATA
    WMS_TITLE "earth.naturalearth.rgb"
  END
END
```

##### `/storage/mapserver-datasets/earth/etopo1/etopo1.map`
```bash
LAYER
  NAME "earth.etopo1.dem"
  STATUS ON
  TYPE RASTER
  DATA "earth/etopo1/ETOPO1_Ice_c_geotiff.tif"

  # Decreasing the oversampling factor will increase performance but reduce quality.
  PROCESSING "OVERSAMPLE_RATIO=10"
  PROCESSING "RESAMPLE=BILINEAR"

  # The ETOPO1 GeoTiff contains extent information but no projection...
  PROJECTION
    "init=epsg:4326"
  END

  METADATA
    WMS_TITLE "earth.etopo1.dem"
  END
END
```

#### Testing the Datasets

With these files in place, we can now check if the MapServer can serve the maps as supposed.
Here are some URLs you can open in your browser.
In each of them you will have to adjust the location of your `meta.map` file.
You can also adjust the `layer=` parameter in each URL to be either `earth.bluemarble.rgb` or `earth.etopo1.dem`.

``` bash
# EPSG:4326
http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms&version=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=-90,-180,90,180&width=1600&height=800&crs=epsg:4326&format=pngRGB

# HEALPix
http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms&version=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=-3.142,-1.571,3.142,1.571&width=1600&height=800&crs=epsg:900915&format=pngRGB

# Rotated HEALPix
http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms&version=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=0,0,5,5&width=800&height=800&crs=epsg:900914&format=pngRGB

# One base patch of rotated HEALPix
http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms&version=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=3,2,4,3&width=800&height=800&crs=epsg:900914&format=pngRGB
```

#### Optional: Optimizing the Dataset

The datasets we downloaded for this tutorial are pretty small and need no optimization.
However, when you start using huge datasets, there are plenty of oportunities to improve the reprojection speed of the MapServer.
One way is to optimize the memory layout for faster access (make it TILED).
Other ways include compression and adding overviews.

```bash
# Make the dataset tiled
gdal_translate -co tiled=yes -co compress=deflate source.tif optimized.tif

# Add overviews
gdaladdo -r cubic optimized.tif 2 4 8 16
```

### Configuring CosmoScout VR

Now that the datasets are working, we only need to include them into CosmoScout VR.
To do this, add the following section to the `"plugins"` array in your `"share/config/simple_desktop.json"`.
You will have to adjust the mapserver links according to the location of your `meta.map` file.

```json
...
"csp-lod-bodies": {
  "maxGPUTilesColor": 1024,
  "maxGPUTilesGray": 1024,
  "maxGPUTilesDEM": 1024,
  "mapCache": "/tmp/map-cache/",
  "bodies": {
    "Earth": {
      "activeImgDataset": "Blue Marble",
      "activeDemDataset": "ETOPO1",
      "imgDatasets": {
        "Blue Marble": {
          "copyright": "NASA",
          "url": "http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms",
          "format": "U8Vec3",
          "layers": "earth.bluemarble.rgb",
          "maxLevel": 6
        },
        "Natural Earth": {
          "copyright": "NASA",
          "url": "http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms",
          "format": "U8Vec3",
          "layers": "earth.naturalearth.rgb",
          "maxLevel": 6
        }
      },
      "demDatasets": {
        "ETOPO1": {
          "copyright": "NOAA",
          "url": "http://localhost/cgi-bin/mapserv?map=/storage/mapserver-datasets/meta.map&service=wms",
          "format": "Float32",
          "layers": "earth.etopo1.dem",
          "maxLevel": 6
        }
      }
    }
  }
},
...
```

You should also **remove** the `"Earth"` section from the `"csp-simple-bodies"` plugin configuration in the same file, else you will have two Earths drawn on top of each other!

**Now you can start CosmoScout VR!**
There will be new configuration options in the sidebar where you can adjust the Visualization of Earth.
