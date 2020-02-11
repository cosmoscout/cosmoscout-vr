<p align="center"> 
  <img src ="img/banner-earth.jpg" />
</p>

# Generic Build Instructions

:information_source: _**Tip:** This page contains generic build instructions for CosmoScout VR. Alternatively, you can follow a [guide specific to your IDE](ide-setup.md)._

**CosmoScout VR supports 64 bits only and can be build in debug and release mode on Linux and Windows.
You will need a copy of [CMake](https://cmake.org/) (version 3.13 or greater), [Boost](https://www.boost.org/) (version 1.69 or greater) and a recent C++ compiler (gcc 7, clang 5 or msvc 19).
For the compilation of the externals [Python](https://www.python.org/) is also required.**

When compiling from source, you can either choose the `master` branch which contains the code of the last stable release or you can switch to the `develop` branch to test the latest features.

## Linux

On Linux, one can either use the provided shell script ([make.sh](../make.sh)) or build the software manually using CMake. 
**Using the provided script** is easy and definitely the recommended way.

Before you start, it may be necessary to install some additional system packages.
As there are many distributions with varying default libs and available packages, giving an exhaustive list is difficult.
Here is an exemplary list for Ubuntu 19.10 which you have to adapt to your specific distribution:

```bash
sudo apt-get install git cmake build-essential xorg-dev libboost-all-dev libglu1-mesa-dev libssl-dev
```

### Cloning the repository

```shell
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```

The default branch is `master`, which contains the code of the last stable release. If you want to test the latest features, you can switch to the `develop` branch.

```shell
git checkout develop
```

### Getting the dependencies

Per default, CosmoScout VR and all dependencies are built in release mode.
You can switch to debug mode by setting the environment variable `export COSMOSCOUT_DEBUG_BUILD=true` before executing the scripts below.
This step only has to be done once.

```shell
git submodule update --init
./make_externals.sh -G "Unix Makefiles"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr/build/linux-externals-release` and will install them to `cosmoscout-vr/install/linux-externals-release`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

### Compiling CosmoScout VR

This will configure and build CosmoScout VR in `cosmoscout-vr/build/linux-release` and will install it to `cosmoscout-vr/install/linux-release`.
Again, all parameters given to `make.sh` will be forwarded to CMake:

```shell
./make.sh -G "Unix Makefiles" -DCOSMOSCOUT_UNIT_TESTS=On
```


The application can be executed with:

```shell
./install/linux-release/bin/start.sh
```

When started for the very first time, some example datasets will be downloaded from the internet.
**This will take some time!**
The progress of this operation is shown on the loading screen.

Since you specified `-DCOSMOSCOUT_UNIT_TESTS=On` at build time, you can now execute the unit tests with (the _graphical tests_ require [Xvfb](https://en.wikipedia.org/wiki/Xvfb) and [imagemagick](https://imagemagick.org/index.php) to be installed on your system):

```shell
./install/linux-release/bin/run_tests.sh
./install/linux-release/bin/run_graphical_tests.sh
```

:information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

For **manual compilation** follow the steps outlined in [make.sh](../make.sh).

:information_source: _**Tip:** You can use [ccache](https://ccache.dev/) to considerably speed up build times. You just need to call `./make_externals.sh -G "Unix Makefiles" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache` and `./make.sh -G "Unix Makefiles" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache` respectively._

## Windows

:warning: _**Warning:** During compilation of the externals, files with pretty long names are generated. Since Windows does not support paths longer 260 letters, you have to compile CosmoScout VR quite close to your file system root (`e.g. C:\cosmoscout-vr`). If you are on Windows 10, [you can disable this limit](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/)._

### Cloning the repository

For Windows, there is a batch script ([make.bat](../make.bat)) which can be used in the same way as the script for Linux.
First you have to clone the repository:

```batch
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```

The default branch is `master`, which contains the code of the last stable release. If you want to test the latest features, you can switch to the `develop` branch.

```shell
git checkout develop
```

### Getting the dependencies

Getting a precompiled version of boost suitable for CosmoScout VR which will be found by CMake can be difficult: Older CMake versions fail to find boost versions which are too new; but on the other hand you need a rather new version if you use a very recent version of MSVC (e.g. 14.2, the one shipped with Visual Studio 2019). The "oldest" precompiled boost which you can get on SourceForge for MSVC 14.2 is version 1.70.0.

So using version 1.70.0 may work in most cases. You can get it from from https://sourceforge.net/projects/boost/files/boost-binaries/1.70.0

MSVC | Visual Studio | File | Link
--- | --- | --- | ---
14.2 | 2019 | `boost_1_70_0-unsupported-msvc-14.2-64.exe` | [download](https://sourceforge.net/projects/boost/files/boost-binaries/1.70.0/boost_1_70_0-unsupported-msvc-14.2-64.exe/download)
14.1 | 2017 | `boost_1_70_0-msvc-14.1-64.exe` | [download](https://sourceforge.net/projects/boost/files/boost-binaries/1.70.0/boost_1_70_0-msvc-14.1-64.exe/download)
14.0 | 2015 | `boost_1_70_0-msvc-14.0-64.exe` | [download](https://sourceforge.net/projects/boost/files/boost-binaries/1.70.0/boost_1_70_0-msvc-14.0-64.exe/download)



Then you have to compile the dependencies.
Per default, CosmoScout VR and all dependencies are built in release mode.
You can switch to debug mode by setting the environment variable `set COSMOSCOUT_DEBUG_BUILD=true` (or `$env:COSMOSCOUT_DEBUG_BUILD = 'true'` if you are using PowerShell) before executing the scripts below.
This step only has to be done once.

If you are using Visual Studio 2019, you have to replace `-G "Visual Studio 15 Win64"` with `-G "Visual Studio 16 2019" -A x64`.

```batch
git submodule update --init
make_externals.bat -G "Visual Studio 15 Win64"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr\build\windows-externals-release` and will install them to `cosmoscout-vr\install\windows-externals-release`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

### Compiling CosmoScout VR

On Linux, boost is usually found automatically by CMake, on Windows you have to provide the `BOOST_ROOT` path.
**Replace the path in the command below to match your setup!**

```batch
set BOOST_ROOT=C:\local\boost_1_69_0
make.bat -G "Visual Studio 15 Win64" -DCOSMOSCOUT_UNIT_TESTS=On
```

This will configure and build CosmoScout VR in `cosmoscout-vr\build\windows-release` and will install it to `cosmoscout-vr\install\windows-release`.
The application can be executed with:

```batch
cd install\windows-release\bin
start.bat
```

When started for the very first time, some example datasets will be downloaded from the internet.
**This will take some time!**
The progress of this operation is shown on the loading screen.

Since you specified `-DCOSMOSCOUT_UNIT_TESTS=On` at build time, you can now execute the unit tests with:

```batch
install\linux-release\bin\run_tests.bat
```

:information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

:information_source: _**Tip:** You can use [clcache](https://github.com/frerich/clcache) to considerably speed up build times. You just need to call `make_externals.bat -G "Visual Studio 15 Win64" -DCMAKE_VS_GLOBALS="CLToolExe=clcache.exe;TrackFileAccess=false"` and `make.bat -G "Visual Studio 15 Win64" -DCMAKE_VS_GLOBALS=CLToolExe"=clcache.exe;TrackFileAccess=false"` respectively._

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="ide-setup.md">Setup your IDE &rsaquo;</a>
</p>
