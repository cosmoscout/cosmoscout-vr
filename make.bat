@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem      and may be used under the terms of the MIT license. See the LICENSE file for details.     #
rem                         Copyright: (c) 2019 German Aerospace Center (DLR)                      #
rem ---------------------------------------------------------------------------------------------- #

rem ---------------------------------------------------------------------------------------------- #
rem Default build mode is release, if "export COSMOSCOUT_DEBUG_BUILD=true" is executed before, the #
rem application will be built in debug mode.                                                       #
rem Usage:                                                                                         #
rem    make.bat [additional CMake flags, defaults to -G "Visual Studio 15 Win64"]                  #
rem Examples:                                                                                      #
rem    make.bat                                                                                    #
rem    make.bat -G "Visual Studio 15 Win64"                                                        #
rem    make.bat -G "Visual Studio 16 2019" -A x64                                                  #
rem    make.bat -GNinja -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe                      #
rem ---------------------------------------------------------------------------------------------- #

rem create some required variables -----------------------------------------------------------------

rem The CMake generator and other flags can be passed as parameters.
set CMAKE_FLAGS=-G "Visual Studio 15 Win64"
IF NOT "%~1"=="" (
  SET CMAKE_FLAGS=%*
)

rem Check if ComoScout VR Debug build is set with the environment variable
IF "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
  ECHO CosmoScout VR debug build is enabled!
  set BUILD_TYPE=Debug
) else (
  set BUILD_TYPE=Release
)

rem Check if unity build is disabled with "set COSMOSCOUT_USE_UNITY_BUILD=false".
IF "%COSMOSCOUT_USE_UNITY_BUILD%"=="false" (
  echo Unity build is disabled!
  set UNITY_BUILD=Off
) else (
  set UNITY_BUILD=On
)

rem Check if precompiled headers should not be used with "set COSMOSCOUT_USE_PCH=false".
IF "%COSMOSCOUT_USE_PCH%"=="false" (
  echo Precompiled headers are disabled!
  set PRECOMPILED_HEADERS=Off
) else (
  set PRECOMPILED_HEADERS=On
)

rem This directory should contain the top-level CMakeLists.txt - it is assumed to reside in the same
rem directory as this script.
set CMAKE_DIR=%~dp0

rem Get the current directory - this is the default location for the build and install directory.
set CURRENT_DIR=%cd%

rem The build directory.
set BUILD_DIR=%CURRENT_DIR%/build/windows-%BUILD_TYPE%

rem The install directory.
set INSTALL_DIR=%CURRENT_DIR%/install/windows-%BUILD_TYPE%

rem This directory should be used as the install directory for make_externals.bat.
set EXTERNALS_INSTALL_DIR=%CURRENT_DIR%/install/windows-externals-%BUILD_TYPE%

rem create build directory if necessary -----------------------------------------------------------

if exist "%BUILD_DIR%" goto BUILD_DIR_CREATED
    mkdir "%BUILD_DIR%"
:BUILD_DIR_CREATED

rem configure, compile & install -------------------------------------------------------------------

cd "%BUILD_DIR%"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD% -DCOSMOSCOUT_USE_PRECOMPILED_HEADERS=%PRECOMPILED_HEADERS%^
      -DCOSMOSCOUT_EXTERNALS_DIR="%EXTERNALS_INSTALL_DIR%" "%CMAKE_DIR%"  || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS% || exit /b

rem Delete empty files installed by cmake
robocopy "%INSTALL_DIR%\lib" "%INSTALL_DIR%\lib" /s /move || exit /b

cd "%CURRENT_DIR%"
echo Finished successfully.

@echo on
