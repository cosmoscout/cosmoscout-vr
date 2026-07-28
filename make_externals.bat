@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem ---------------------------------------------------------------------------------------------- #

rem SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
rem SPDX-License-Identifier: MIT

rem ---------------------------------------------------------------------------------------------- #
rem Make sure to run "git submodule update --init" before executing this script!                   #
rem Default build mode is release, if "set COSMOSCOUT_DEBUG_BUILD=true" is executed before, all    #
rem dependencies will be built in debug mode.                                                      #
rem Usage:                                                                                         #
rem    make_externals.bat [additional CMake flags, defaults to -G "Visual Studio 16 2019" -A x64]  #
rem Examples:                                                                                      #
rem    make_externals.bat                                                                          #
rem    make_externals.bat -G "Visual Studio 16 2019" -A x64                                        #
rem    make_externals.bat -G "Visual Studio 17 2022" -A x64                                        #
rem    make_externals.bat -GNinja -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe            #
rem ---------------------------------------------------------------------------------------------- #

rem The CMake generator and other flags can be passed as parameters.
set CMAKE_FLAGS=-G "Visual Studio 16 2019" -A x64
IF NOT "%~1"=="" (
  SET CMAKE_FLAGS=%*
)

rem We need to check if Ninja is used as a generator, since there are some minor differences in generation paths.
echo.%CMAKE_FLAGS%|findstr /C:"Ninja" >nul 2>&1
IF NOT errorlevel 1 (
   set USING_NINJA=true
) else (
   set USING_NINJA=false
)

rem Check if ComoScout VR debug build is enabled with "set COSMOSCOUT_DEBUG_BUILD=true".
IF "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
  echo CosmoScout VR debug build is enabled!
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

rem Create some required variables. ----------------------------------------------------------------

rem This directory should contain all submodules - they are assumed to reside in the subdirectory 
rem "externals" next to this script. We replace all \ with /.
set EXTERNALS_DIR=%~dp0\externals
set EXTERNALS_DIR=%EXTERNALS_DIR:\=/%

rem Get the current directory - this is the default location for the build and install directory.
rem We replace all \ with /.
set CURRENT_DIR=%cd:\=/%

rem The build directory.
set BUILD_DIR=%CURRENT_DIR%/build/windows-externals-%BUILD_TYPE%

rem The install directory.
set INSTALL_DIR=%CURRENT_DIR%/install/windows-externals-%BUILD_TYPE%

rem Create some default installation directories.
cmake -E make_directory "%INSTALL_DIR%/lib"
cmake -E make_directory "%INSTALL_DIR%/share"
cmake -E make_directory "%INSTALL_DIR%/bin"
cmake -E make_directory "%INSTALL_DIR%/include"

rem vcpkg ------------------------------------------------------------------------------------------
:setup_vcpkg

echo.
echo Setting up vcpkg ...
echo.

if NOT EXIST "%CURRENT_DIR%/vcpkg" (
  echo Cloning vcpkg...
  git clone https://github.com/microsoft/vcpkg "%CURRENT_DIR%/vcpkg" || goto :error
) else (
  echo vcpkg already exists, updating...
  cd "%CURRENT_DIR%/vcpkg"
  git pull || goto :error
  cd "%CURRENT_DIR%"
)

echo Bootstrapping vcpkg...
cd "%CURRENT_DIR%/vcpkg"
call bootstrap-vcpkg.bat || goto :error
cd "%CURRENT_DIR%"

rem Install GLEW via vcpkg (this will be replaced as more libraries are migrated)
echo Installing GLEW via vcpkg...
cd "%CURRENT_DIR%/vcpkg"
call vcpkg install || goto :error
cd "%CURRENT_DIR%"

rem Use vcpkg toolchain for GLEW and other vcpkg-managed dependencies
set VCPKG_TOOLCHAIN=%CURRENT_DIR%/vcpkg/scripts/buildsystems/vcpkg.cmake
set VCPKG_INSTALL_DIR=%CURRENT_DIR%/vcpkg_installed/x64-windows

rem opensg -----------------------------------------------------------------------------------------
:opensg

echo.
echo Building and installing opensg-1.8 ...
echo.

cmake -E make_directory "%BUILD_DIR%/opensg-1.8" && cd "%BUILD_DIR%/opensg-1.8"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD% -DOPENSG_INFINITE_REVERSE_PROJECTION=ON^
      -DOPENSG_USE_PRECOMPILED_HEADERS=%PRECOMPILED_HEADERS%^
      -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DOPENSG_BUILD_WINDOW=Off^
      -DOPENSG_BUILD_TESTS=Off "%EXTERNALS_DIR%/opensg-1.8" || goto :error

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS% || goto :error

rem OpenVR ----------------------------------------------------------------------------------------
:openvr

echo.
echo Building and installing OpenVR ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/openvr/bin/win64" "%INSTALL_DIR%/bin"            || goto :error
cmake -E copy_directory "%EXTERNALS_DIR%/openvr/lib/win64" "%INSTALL_DIR%/lib"            || goto :error
cmake -E copy_directory "%EXTERNALS_DIR%/openvr/headers"   "%INSTALL_DIR%/include/openvr" || goto :error

rem vista ------------------------------------------------------------------------------------------
:vista

echo.
echo Building and installing vista ...
echo.

cmake -E make_directory "%BUILD_DIR%/vista" && cd "%BUILD_DIR%/vista"

rem If you have the 3DConnexion SDK for the Space Navigator installed, you can add
rem -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On to the flags below.

cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" -DVISTADEMO_ENABLED=Off^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DVISTACORELIBS_USE_OPENVR=On -DVISTADRIVERS_BUILD_OPENVR=On^
      -DVISTACORELIBS_USE_INFINITE_REVERSE_PROJECTION=On -DOPENSG_ROOT_DIR=%INSTALL_DIR%^
      -DOPENVR_ROOT_DIR="%INSTALL_DIR%" -DVISTACORELIBS_USE_GLUT_WINDOWIMP=Off^
      -DGLEW_ROOT_DIR="%VCPKG_INSTALL_DIR%" -DVISTACORELIBS_USE_SDL2_WINDOWIMP=On -DSDL2_ROOT_DIR=%VCPKG_INSTALL_DIR%^
      -DSDL2_TTF_ROOT_DIR=%VCPKG_INSTALL_DIR% -DCMAKE_UNITY_BUILD=%UNITY_BUILD%^
      -DVISTA_USE_PRECOMPILED_HEADERS=%PRECOMPILED_HEADERS% "%EXTERNALS_DIR%/vista" || goto :error
cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS% || goto :error

rem cef --------------------------------------------------------------------------------------------
:cef

echo.
echo Downloading, building and installing cef (this may take some time) ...
echo.

set CEF_DIR=cef_binary_135.0.20+ge7de5c3+chromium-135.0.7049.85_windows64_minimal

cmake -E make_directory "%BUILD_DIR%/cef/extracted" && cd "%BUILD_DIR%/cef"

IF NOT EXIST cef.tar.bz2 (
  curl.exe https://cef-builds.spotifycdn.com/cef_binary_135.0.20+ge7de5c3+chromium-135.0.7049.85_windows64_minimal.tar.bz2 --output cef.tar.bz2

  cd "%BUILD_DIR%/cef/extracted"

  cmake -E tar xfj ../cef.tar.bz2

  rem We don't want the example applications.
  cmake -E remove_directory %CEF_DIR%/tests

  rem Very ugly workaround for a linking bug, where CEF is build with different flags than the
  rem rest of the project.
  IF "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content '%BUILD_DIR%/cef/extracted/%CEF_DIR%/cmake/cef_variables.cmake') -replace '_HAS_ITERATOR_DEBUGGING=0', '_HAS_ITERATOR_DEBUGGING=1' | Set-Content '%BUILD_DIR%/cef/extracted/%CEF_DIR%/cmake/cef_variables.cmake'"
  )
) else (
  echo File 'cef.tar.bz2' already exists, no download required.
)

cd "%BUILD_DIR%/cef/

cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD% -DCEF_RUNTIME_LIBRARY_FLAG=/MD -DCEF_DEBUG_INFO_FLAG=""^
      "%BUILD_DIR%/cef/extracted/%CEF_DIR%" || goto :error

cmake --build . --config %BUILD_TYPE% --parallel %NUMBER_OF_PROCESSORS% || goto :error

echo Installing cef...
cmake -E make_directory "%INSTALL_DIR%/include/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/include"                   "%INSTALL_DIR%/include/cef/include"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/Resources"                 "%INSTALL_DIR%/share/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/Release"                   "%INSTALL_DIR%/lib"

if %USING_NINJA%==true (
  cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/libcef_dll_wrapper.lib"  "%INSTALL_DIR%/lib"
) else (
  cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/%BUILD_TYPE%/libcef_dll_wrapper.lib"  "%INSTALL_DIR%/lib"
)

rem ------------------------------------------------------------------------------------------------

:finish
echo Finished successfully.
goto :end

:error
echo Errors occurred!

:end
cd "%CURRENT_DIR%"
@echo on
