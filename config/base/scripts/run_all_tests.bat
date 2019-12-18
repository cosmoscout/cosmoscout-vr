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

rem Set paths so that all libraries are found.
set PATH=%SCRIPT_DIR%\..\lib;%PATH%

cosmoscout.exe --run-tests

rem Go back to where we came from
cd "%CURRENT_DIR%"

@echo on
