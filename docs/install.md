<p align="center"> 
  <img src ="img/banner-earth.jpg" />
</p>

# Build Instructions

:warning: _**Warning:** The [default configuration](../config/base/scene/simple_desktop.json) only contains a few data sets with very low resolution. Please read the [Configuring Guide](configuring.md) and the documentation of the [individual plugins](../README.md#Plugins-for-CosmoScout-VR) for including new data sets._

CosmoScout VR supports 64 bits only and can be build in debug and release mode on Linux and Windows.
Most dependencies are included as [git submodules](../externals).
**You will only need a copy of [CMake](https://cmake.org/) (version 3.12 or greater), [Boost](https://www.boost.org/) (version 1.69 or greater) and a recent C++ compiler (gcc 7, clang 5 or msvc 19).
For the compilation of the externals [Python](https://www.python.org/) is also required.**

When compiling from source, you can either choose the `master` branch which contains the code of the last stable release or you can switch to the `develop` branch to test the latest features.

## Linux

On Linux, one can either use the provided shell scripts ([make_release.sh](../make_release.sh) and [make_debug.sh](../make_debug.sh)) or build the software manually using CMake. 
**Using the provided scripts** is easy and definitely the recommended way.

In any way, first you have to clone the repository.

```shell
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```

The default branch is `master`, which contains the code of the last stable release. If you want to test the latest features, you can switch to the `develop` branch.

```shell
git checkout develop
```

Then you have to compile the dependencies. This step only has to be done once.

```shell
git submodule update --init
./make_externals.sh -G "Unix Makefiles"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr/build/linux-externals` and will install them to `cosmoscout-vr/install/linux-externals`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

For a minimal setup you will need some data sets which are not included in this git repository.
Run this script to download the data to `cosmoscout-vr/data`:

```shell
./get_example_data.sh
```

Now you can compile CosmoScout VR:

```shell
./make_release.sh -G "Unix Makefiles"
```

This will configure and build CosmoScout VR in `cosmoscout-vr/build/linux-release` and will install it to `cosmoscout-vr/install/linux-release`.
Again, all parameters given to `make_externals.bat` will be forwarded to CMake.

The application can be executed with:

```shell
cd install/linux-release/bin
./start.sh
```

:information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

For **manual compilation** follow the steps outlined in [make_release.sh](../make_release.sh) or [make_debug.sh](../make_debug.sh).

## Windows

For Windows, there are batch scripts ([make_release.bat](../make_release.bat) and [make_debug.bat](../make_debug.bat)) which can be used in the same way as the scripts for Linux:

In any way, first you have to clone the repository.

```batch
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
```


The default branch is `master`, which contains the code of the last stable release. If you want to test the latest features, you can switch to the `develop` branch.

```shell
git checkout develop
```

Then you have to compile the dependencies. This step only has to be done once.


```batch
git submodule update --init
make_externals.bat -G "Visual Studio 15 Win64"
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr\build\windows-externals` and will install them to `cosmoscout-vr\install\windows-externals`.
All parameters given to `make_externals.bat` will be forwarded to CMake. For example, you can change the CMake generator this way.

For a minimal setup you will need some data sets which are not included in this git repository.
Run this script to download the data to `cosmoscout-vr\data`:

```batch
get_example_data.bat
```

Now you can compile CosmoScout VR.
On Linux, boost is usually found automatically by CMake, on Windows you have to provide the `BOOST_ROOT` path.
**Replace the path in the command below to match your setup!**

```batch
set BOOST_ROOT=C:\local\boost_1_69_0
make_release.bat -G "Visual Studio 15 Win64"
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