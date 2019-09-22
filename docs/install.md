<p align="center"> 
  <img src ="img/banner-earth.jpg" />
</p>

# Build Instructions

:warning: _**Warning:** The [default configuration](../config/base/scene/simple_desktop.json) only contains a few data sets with very low resolution. Please read the [Configuring Guide](configuring.md) and the documentation of the [individual plugins](../README.md#Plugins-for-CosmoScout-VR) for including new data sets._

**CosmoScout VR supports 64 bits only and can be build in debug and release mode on Linux and Windows.
You will need a copy of [CMake](https://cmake.org/) (version 3.12 or greater), [Boost](https://www.boost.org/) (version 1.69 or greater) and a recent C++ compiler (gcc 7, clang 5 or msvc 19).
For the compilation of the externals [Python](https://www.python.org/) is also required.**

When compiling from source, you can either choose the `master` branch which contains the code of the last stable release or you can switch to the `develop` branch to test the latest features.

## Linux

On Linux, one can either use the provided shell script ([make.sh](../make.sh)) or build the software manually using CMake. 
**Using the provided script** is easy and definitely the recommended way.

In any way, first you have to clone the repository.

```shell
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```

The default branch is `master`, which contains the code of the last stable release. If you want to test the latest features, you can switch to the `develop` branch.

```shell
git checkout develop
```

Then you have to compile the dependencies.
Per default, CosmoScout VR and all dependencies are built in release mode.
You can switch to debug mode by setting the environment variable `export COSMOSCOUT_DEBUG_BUILD=true` before executing the scripts below.
This step only has to be done once.

```shell
git submodule update --init
./make_externals.sh -G "Unix Makefiles"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr/build/linux-externals-release` and will install them to `cosmoscout-vr/install/linux-externals-release`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

Now you can compile CosmoScout VR:

```shell
./make.sh -G "Unix Makefiles"
```

This will configure and build CosmoScout VR in `cosmoscout-vr/build/linux-release` and will install it to `cosmoscout-vr/install/linux-release`.
Again, all parameters given to `make.sh` will be forwarded to CMake.

The application can be executed with:

```shell
cd install/linux-release/bin
./start.sh
```

:information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

For **manual compilation** follow the steps outlined in [make.sh](../make.sh).

## Windows

:warning: _**Warning:** During compilation of the externals, files with pretty long names are generated. Since Windows does not support paths longer 260 letters, you have to compile CosmoScout VR quite close to your file system root (`e.g. C:\cosmoscout-vr`). If you are on Windows 10, [you can disable this limit](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/)._

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

Then you have to compile the dependencies.
Per default, CosmoScout VR and all dependencies are built in release mode.
You can switch to debug mode by setting the environment variable `set COSMOSCOUT_DEBUG_BUILD=true` (or `$env:COSMOSCOUT_DEBUG_BUILD = 'true'` if you are using PowerShell) before executing the scripts below.
This step only has to be done once.

```batch
git submodule update --init
make_externals.bat -G "Visual Studio 15 Win64"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr\build\windows-externals-release` and will install them to `cosmoscout-vr\install\windows-externals-release`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

Now you can compile CosmoScout VR.
On Linux, boost is usually found automatically by CMake, on Windows you have to provide the `BOOST_ROOT` path.
**Replace the path in the command below to match your setup!**

```batch
set BOOST_ROOT=C:\local\boost_1_69_0
make.bat -G "Visual Studio 15 Win64"
```

This will configure and build CosmoScout VR in `cosmoscout-vr\build\windows-release` and will install it to `cosmoscout-vr\install\windows-release`.
The application can be executed with:

```batch
cd install\windows-release\bin
start.bat
```

:information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

<p align="center">
  <a href="dependencies.md">&lsaquo; Dependencies</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="using.md">Using CosmoScout VR &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>