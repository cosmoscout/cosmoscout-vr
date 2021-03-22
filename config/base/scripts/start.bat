@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem      and may be used under the terms of the MIT license. See the LICENSE file for details.     #
rem                         Copyright: (c) 2019 German Aerospace Center (DLR)                      #
rem ---------------------------------------------------------------------------------------------- #

rem Change working directory to the location of this script.
set SCRIPT_DIR=%~dp0
set CURRENT_DIR=%cd%
cd "%SCRIPT_DIR%"

rem Scene config file can be passed as first parameter.
set SETTINGS=../share/config/vestec.json
IF NOT "%1"=="" (
  SET SETTINGS=%1
  SHIFT
)

rem Vista ini can be passed as second parameter.
set VISTA_INI=vista.ini
IF NOT "%1"=="" (
  SET VISTA_INI=%1
  SHIFT
)

rem Set paths so that all libraries are found.
set PROJ_LIB=%SCRIPT_DIR%\proj6\share
 
set VISTACORELIBS_DRIVER_PLUGIN_DIRS=%SCRIPT_DIR%\..\lib\DriverPlugins
set PATH=%SCRIPT_DIR%\..\lib;%SCRIPT_DIR%\..\lib\ttk;%PATH%

cosmoscout.exe --settings=%SETTINGS% -vistaini %VISTA_INI%

rem Go back to where we came from
cd "%CURRENT_DIR%"

@echo on
