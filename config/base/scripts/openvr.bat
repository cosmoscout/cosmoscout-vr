@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem ---------------------------------------------------------------------------------------------- #

rem SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
rem SPDX-License-Identifier: CC0-1.0

rem Change working directory to the location of this script.
set SCRIPT_DIR=%~dp0
set CURRENT_DIR=%cd%
cd "%SCRIPT_DIR%"

rem Scene config file can be passed as first parameter.
set SETTINGS=../share/config/simple_openvr.json
IF NOT "%1"=="" (
  SET SETTINGS=%1
  SHIFT
)

rem Vista ini can be passed as second parameter.
set VISTA_INI=vista_openvr.ini
IF NOT "%1"=="" (
  SET VISTA_INI=%1
  SHIFT
)

rem Set paths so that all libraries are found.
set VISTACORELIBS_DRIVER_PLUGIN_DIRS=%SCRIPT_DIR%\..\lib\DriverPlugins
set PATH=%SCRIPT_DIR%\..\lib;%PATH%

cosmoscout.exe --settings=%SETTINGS% -vistaini %VISTA_INI%

rem Go back to where we came from
cd "%CURRENT_DIR%"

@echo on
