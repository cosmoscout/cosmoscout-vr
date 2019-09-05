@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem      and may be used under the terms of the MIT license. See the LICENSE file for details.     #
rem                         Copyright: (c) 2019 German Aerospace Center (DLR)                      #
rem ---------------------------------------------------------------------------------------------- #

rem create some required variables -----------------------------------------------------------------

rem This directory should contain the top-level CMakeLists.txt - it is assumed to reside in the same
rem directory as this script.
set CMAKE_DIR=%~dp0

rem Get the current directory - this is the default location for the build and install directory.
set CURRENT_DIR=%cd%

rem The build directory.
set BUILD_DIR=%CURRENT_DIR%\build\windows-release

rem The install directory.
set INSTALL_DIR=%CURRENT_DIR%\install\windows-release

rem This directory should be used as the install directory for make_externals.bat.
set EXTERNALS_INSTALL_DIR=%CURRENT_DIR%\install\windows-externals

rem create build directory if necessary -----------------------------------------------------------

if exist "%BUILD_DIR%" goto BUILD_DIR_CREATED
    mkdir "%BUILD_DIR%"
:BUILD_DIR_CREATED

rem configure, compile & install -------------------------------------------------------------------

cd "%BUILD_DIR%"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCOSMOSCOUT_EXTERNALS_DIR="%EXTERNALS_INSTALL_DIR%"^
      -DCMAKE_EXPORT_COMPILE_COMMANDS=On "%CMAKE_DIR%"

cmake --build . --config Release --target install --parallel 8 || exit /b

rem Delete empty files installed by cmake
robocopy "%INSTALL_DIR%" "%INSTALL_DIR%" /s /move

cd "%CURRENT_DIR%"
echo Finished successfully.

@echo on
