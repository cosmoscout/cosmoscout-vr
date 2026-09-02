<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

<p align="center"> 
  <img src ="img/banner-earth.jpg" />
</p>

# Generic Build Instructions

:information_source: _**Tip:** This page contains generic build instructions for CosmoScout VR. Alternatively, you can follow a [guide specific to your IDE](ide-setup.md)._

**CosmoScout VR supports 64 bits only and can be build in debug and release mode on Linux and Windows.
You will need a copy of [CMake](https://cmake.org/) (version 3.28 or greater) and a recent C++ compiler (gcc 13, clang 18 or msvc 19).**

Before you start, it may be necessary to install some additional system packages.
As there are many distributions with varying default libs and available packages, giving an exhaustive list is difficult.
Here is an exemplary list for Ubuntu 24.04 which you have to adapt to your specific distribution:

```bash
sudo apt-get install git cmake build-essential ninja-build xorg-dev libglu1-mesa-dev libssl-dev libxkbcommon0
```

For Windows the following software needs to be installed:
- Visual Studio 2019 or newer
- CMake
- Ninja Build (optional, but recommended)

### Cloning the repository

```shell
git clone https://github.com/cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```

### Getting the dependencies

CosmoScout VR is being build with vcpkg as a package manager. We bundle vcpkg as a submodule. For this we need to initialize it:

```shell
git submodule update --init
```

### Compiling CosmoScout VR

One can either use [CMake Presets](https://cmake.org/cmake/help/v3.28/manual/cmake-presets.7.html) or build the software manually using CMake. 
**Using CMake Presets** is easy and definitely the recommended way.

You can get a list of available configuration presets using the following command:

```shell
cmake --workflow --list-presets
```

The results will be structured the following way: <platform>-<build-tool>-<build-type>.
After you decided for a preset, you can run it with the following command:

```shell
cmake --workflow --preset <preset-name>
```

The application can be executed with:

```shell
# Linux
./install/linux-Release/bin/start.sh

# Windows
./install/windows-Release/bin/start.bat
```

When started for the very first time, some example datasets will be downloaded from the internet.
**This will take some time!**
The progress of this operation is shown on the loading screen.

When `-DCOSMOSCOUT_UNIT_TESTS=On` is specified, you can run unit tests with (the _graphical tests_ require [Xvfb](https://en.wikipedia.org/wiki/Xvfb) and [imagemagick](https://imagemagick.org/index.php) to be installed on your system. On Ubuntu: `sudo apt-get install xvfb imagemagick`. Windows doesn't support graphical tests.):

```shell
# Linux
./install/linux-Release/bin/run_tests.sh
./install/linux-Release/bin/run_graphical_tests.sh

# Windows
./install/windows-Release/bin/run_tests.bat
```

> [!TIP]
> If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation.

> [!TIP]
> You can create your own presets by creating `CMakeUserPresets.json` and inherit from a global preset or define your completely own preset.

> [!TIP]
> You can override preset variables by simply setting them in the command line after the preset name e.g.:
> ```bash
> cmake --preset <preset-name> -DCOSMOSCOUT_UNIT_TESTS=Off
> cmake --build --preset <preset-name> --parallel 2
> ```

> [!TIP]
> You can use [ccache](https://ccache.dev/) on Linux to considerably speed up build times. You just need to call `cmake --preset <preset-name> -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache`.

> [!TIP]
> You can use [clcache](https://github.com/frerich/clcache) on Windows to considerably speed up build times. You just need to call `cmake --preset <preset-name> -DCMAKE_VS_GLOBALS=CLToolExe"=clcache.exe;TrackFileAccess=false"`.


<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="ide-setup.md">Setup your IDE &rsaquo;</a>
</p>
