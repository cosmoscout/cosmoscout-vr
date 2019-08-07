@echo off

set VISTACORELIBS_DRIVER_PLUGIN_DIRS=..\lib\DriverPlugins
set PATH=..\lib;%PATH%

cosmoscout.exe --settings=../share/config/simple_desktop.json -vistaini vista.ini

@echo on
