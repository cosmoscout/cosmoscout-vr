# Location files for CosmoScout VR

The files in this directory have been retrieved from https://planetarynames.wr.usgs.gov/AdvancedSearch. They are used to perform forward and reverse geocoding in CosmoScout VR. You may add your own files here, the format is quite simple:

```csv
Feature_ID,Feature_Name,Diameter,Center_Latitude,Center_Longitude,
4,"Abalos Colles",235.83,76.83,-71.65,
5,"Abalos Undae",442.74,78.52,-87.5,
6,"Aban",4.28,15.91,111.1,
25,"Abus Vallis",60.99,-5.49,-147.2,
...
```

The name of the file should be the SPICE center name of the corresponding body. The column `Feature_ID` is not used, but required when requesting data from [planetarynames.wr.usgs.gov](https://planetarynames.wr.usgs.gov/AdvancedSearch). `Diameter` is in kilometers, `Center_Latitude` and `Center_Longitude` in planetographic coordinates.

The leading and trailing white-space from the files has been removed. For `mars.csv`, the really huge diameter of `4454,"Olympus Rupes",1914.77,18.4,-133.56,` had been reduced to `100`. Else no modifactions where made.

Some files are basically empty, because [planetarynames.wr.usgs.gov](https://planetarynames.wr.usgs.gov/AdvancedSearch) does not list any locations for those bodies. The empty files are there to prevent file loading error messages.