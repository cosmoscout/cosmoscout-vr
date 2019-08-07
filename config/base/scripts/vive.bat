@echo off

set VISTACORELIBS_DRIVER_PLUGIN_DIRS=..\lib\DriverPlugins
set PATH=..\lib;%PATH%

cosmoscout.exe --settings=simple_vive.json -vistaini vista_vive.ini

@echo on
