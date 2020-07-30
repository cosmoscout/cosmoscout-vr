# Level-of-Detail Bodies for CosmoScout VR

A CosmoScout VR plugin which draws level-of-detail planets and moons. This plugin supports the visualization of entire planets in a 1:1 scale. The data is streamed via Web-Map-Services (WMS) over the internet. A dedicated MapServer is required to use this plugin.

## Setting up the MapServer on Ubuntu 20.04 

This guide will most likely work for newer versions of Ubuntu as well. For older versions, you may try to add the [UbuntuGIS](https://launchpad.net/~ubuntugis) repository.

### Installing the MapServer

#### 1. Install the packages
```
sudo apt-get install apache2 apache2-bin apache2-utils cgi-mapserver \
                     mapserver-bin mapserver-doc libmapscript-perl   \
                     libapache2-mod-fcgid
```

#### 2. Enable cgi and fastcgi

```
sudo a2enmod cgi fcgid
```

#### 3. Add `/usr/lib/cgi-bin` directory to Apache

Add the following lines on Apache2 configuration file (e.g. `/etc/apache2/sites-available/000-default.conf`):

```
         ScriptAlias /cgi-bin/ /usr/lib/cgi-bin/
         <Directory "/usr/lib/cgi-bin/">
                 AllowOverride All
                 Options +ExecCGI -MultiViews +FollowSymLinks
                 AddHandler fcgid-script .fcgi
                 Require all granted
         </Directory>
```

#### 4. Restart apache2 Daemon

```
sudo service apache2 restart
```

#### 5. Check MapServer Installation 

```
mapserv -v 
```

Navigating with a web browser to [http://localhost/cgi-bin/mapserv?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities](http://localhost/cgi-bin/mapserv?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities) should give the following message:
```
msCGILoadMap(): Web application error. CGI variable "map" is not set.
```

### Configuring the MapServer

#### The `meta.map` file

```
MAP
    NAME "CosmoScout VR Maps"
    STATUS ON
    EXTENT -180 -90 180 90
    SIZE 800 400

    CONFIG "PROJ_LIB" "."

    PROJECTION
        "init=epsg:4326"
    END

    OUTPUTFORMAT
      NAME "tiffGray"
      DRIVER "GDAL/GTiff"
      IMAGEMODE FLOAT32
      EXTENSION "tiff"
      FORMATOPTION "COMPRESS=LZW"
    END

    OUTPUTFORMAT
      NAME "pngGray"
      DRIVER "GDAL/PNG"
      IMAGEMODE BYTE
      EXTENSION "png"
    END

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
    INCLUDE "earth/cleantopo/cleantopo.map"
END
```

#### The `epsg` file

https://github.com/OSGeo/PROJ/releases/tag/5.2.0

```
# custom healpix, magic number is sqrt(2) * 2/pi
<900915> +proj=healpix +a=1 +b=1 <>
<900916> +proj=healpix +lon_0=0 +x_0=2.5 +y_0=2.5 +a=0.900316316 +rot_xy=45 +no_defs <>
```

#### Adding a new Dataset

https://visibleearth.nasa.gov/images/73776/august-blue-marble-next-generation-w-topography-and-bathymetry/73783l

We will use a map from [naturalearthdata.com](http://naturalearthdata.com/). Go and grap a map from [here (with Shaded Relief, Water and Drainages)](http://www.naturalearthdata.com/downloads/10m-raster-data/10m-natural-earth-1/) for example. These maps already include surface shading which is not particularly useful, but they will serve our purpose here.

Extract the containing GeoTiff file to `share\resources\terrain\earth\naturalearth\NE1_HR_LC_SR_W_DR.tif`.

Create a map file for the new dataset (e.g. `earth/naturalearth/naturalearth.map`) and paste the following lines into the file:

http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.1.1&request=GetMap&layers=earth.naturalearth.rgb&bbox=-180,-90,180,90&width=1600&height=800&srs=EPSG:4326&format=pngRGB

http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.1.1&request=GetMap&layers=earth.naturalearth.rgb&bbox=-3,-2,3,2&width=1600&height=800&srs=EPSG:900915&format=pngRGB

```
LAYER
    NAME "earth.naturalearth.rgb"
    STATUS ON
    TYPE RASTER
    DATA "earth/naturalearth/NE1_HR_LC_SR_W_DR.tif"

    PROCESSING "RESAMPLE=BILINEAR"
    PROCESSING "OVERSAMPLE_RATIO=5"

    PROJECTION
        AUTO
    END

    METADATA
        WMS_TITLE "earth.naturalearth.rgb"
    END
END
```

```
LAYER
    NAME "earth.bluemarble.rgb"
    STATUS ON
    TYPE RASTER
    DATA "earth/bluemarble/bluemarble.jpg"

    PROCESSING "RESAMPLE=BILINEAR"
    PROCESSING "OVERSAMPLE_RATIO=10"

    EXTENT -180 -90 180 90

    PROJECTION
        "init=epsg:4326"
    END

    METADATA
        WMS_TITLE "earth.bluemarble.rgb"
    END
END
```

```
LAYER
    NAME "earth.cleantopo.rgb"
    STATUS ON
    TYPE RASTER
    DATA "earth/cleantopo/CleanTOPO2.tif"

    PROCESSING "RESAMPLE=BILINEAR"
    PROCESSING "OVERSAMPLE_RATIO=5"

    PROJECTION
        AUTO
    END

    METADATA
        WMS_TITLE "earth.cleantopo.rgb"
    END
END
```

Then you only need to include the new mapfile in the main map file (`share\resources\terrain\meta.map`). At the bottom of this file, add one line similar to the one which is already there:

```
INCLUDE "earth/naturalearth/naturalearth.map"
```

When this is done and you typed everything as written here, the following links should show the new dataset in your browser. Make sure that apache is running!

[Naturalearth in EPSG:4326](http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=-90,-180,90,180&width=1600&height=800&crs=epsg:4326&format=pngRGB)

http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=-3.142,-1.571,3.142,1.571&width=1600&height=800&crs=epsg:900915&format=pngRGB

http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=0,0,5,5&width=800&height=800&crs=epsg:900916&format=pngRGB

[Naturalearth as basepatch three of HEALPix projection](http://localhost/cgi-bin/mapserv?map=/home/simon/Projects/cosmoscout/mapserver-datasets/meta.map&SERVICE=WMS&VERSION=1.3.0&request=GetMap&layers=earth.naturalearth.rgb&bbox=3,2,4,3&width=800&height=800&crs=epsg:900916&format=pngRGB)

If you did any mistake, you'll be prompted to download mapserv.exe. We're working on displaying useful error messages here...

#### Configure CosmoScout VR

Now that the dataset is working, we only need to include it into Virtual Planet. Therefore we need to add one file and edit another. First, create a `share\config\wms\earth_naturalearth_img.xml`. The contents should be the following:



The "URL" setting is just the base url of your local MapServer; "Layers" references the layer name we chose in the map file above; "Format" is either "U8Vec3" (for RGB-colored maps), "UInt8" (for grayscale maps) or "Float32" (for DEMs) and the last setting, "MaxLevel", defines the maximum depth of the Level-of-Detail structure. This has to be increased for datasets with higher resolution.

Now you only need to add one dataset section to the `imgDatasets` section of Earth `share\config\solarsystem.json`:

{ "name": "Natural Earth", "files": \[ "..\\\\share\\\\config\\\\wms\\\\earth\_naturalearth\_img.xml" \], "wms": true }

Here, "name" will be displayed in the user interface.

Now you can start Virtual Planet and select `Natural Earth` from the image channel selection menu. While this should be working, we can optimize the performance by optimizing the dataset.

#### Optional: Optimize the dataset

Optimizing the dataset can be done in several ways. One way is to optimize the memory layout for faster access (make it TILED). Other ways include compression and adding overviews. To do this, open the geo-tools command prompt (`mapserver\bin\SDKShell.bat`) and type:

cd ..\\share\\resources\\terrain\\earth\\naturalearth
gdal\_translate -co tiled=yes -co compress=deflate NE1\_HR\_LC\_SR\_W\_DR.tif optimized.tif
gdaladdo -r cubic optimized.tif 2 4 8

Now we can use this optimized GeoTiff in our layer. To do this, edit the DATA line in `earth/naturalearth/naturalearth.map`:

DATA "earth/naturalearth/optimized.tif"

Now the map should load faster!

#### Links to freely available datasets

##### WMS Servers

[Geospatial Web Services by DLR](https://geoservice.dlr.de/web/services)

[Sentinel2 Cloudless by EOX](https://s2maps.eu/)

##### Digital Elevation Models and Satellite Imagery

[Grayscale Gale Crater mosaic with 25 cm resolution](https://astrogeology.usgs.gov/search/map/Mars/MarsScienceLaboratory/Mosaics/MSL_Gale_Orthophoto_Mosaic_10m_v3)

[DEM for Gale Crater Mosaic with 1 m resolution](https://astrogeology.usgs.gov/search/map/Mars/MarsScienceLaboratory/Mosaics/MSL_Gale_DEM_Mosaic_10m)

[Blue Marble by NASA](https://visibleearth.nasa.gov/view_cat.php?categoryID=1484)

## Configuration

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

**More in-depth information and some tutorials will be provided soon.**
