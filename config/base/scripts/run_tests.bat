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

rem Set paths so that all libraries are found.
set PATH=%SCRIPT_DIR%\..\lib;%PATH%

cosmoscout.exe --run-tests --test-case-exclude="*[graphical]*"
set RESULT=%ERRORLEVEL%

rem Go back to where we came from
cd "%CURRENT_DIR%"

@echo on

@rem Return the result of our tests.
@exit /b %RESULT%