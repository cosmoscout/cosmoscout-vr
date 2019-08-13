<p align="center"> 
  <img src ="img/banner-earth.jpg" />
</p>

# Build Instructions

For now, no binary packages of CosmoScout VR are provided.
If you want to test or use the software, you have to compile it yourself.

CosmoScout VR can be build in debug and release mode on Linux and Windows.
Most dependencies are included as [git submodules](../externals).
**You will only need a copy of [CMake](https://cmake.org/) (version 3.12 or greater), [Boost](https://www.boost.org/) (version 1.69 or greater) and a recent C++ compiler (gcc 7, clang 5 or msvc 19).
For the compilation of the externals [Python](https://www.python.org/) is also required.**

## Linux

On Linux, one can either use the provided shell scripts ([make_release.sh](../make_release.sh) and [make_debug.sh](../make_debug.sh)) or build the software manually using CMake. 
**Using the provided scripts** is easy and definitely the recommended way.

In any way, first you have to compile the dependencies. This step only has to be done once.

```shell
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
git submodule update --init
./make_externals.sh
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr/build/linux-externals` and will install them to `cosmoscout-vr/install/linux-externals`.
For a minimal setup you will need some data sets which are not included in this git repository.
Run this script to download the data to `cosmoscout-vr/data`:

```shell
./get_example_data.sh
```

Now you can compile CosmoScout VR:

```shell
./make_release.sh
```

This will configure and build CosmoScout VR in `cosmoscout-vr/build/linux-release` and will install it to `cosmoscout-vr/install/linux-release`.
The application can be executed with:

```shell
cd install/linux-release/bin
./start.sh
```

> :information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

For **manual compilation** follow the steps outlined in [make_release.sh](../make_release.sh) or [make_debug.sh](../make_debug.sh).

## Windows

For Windows, there are batch scripts ([make_release.bat](../make_release.bat) and [make_debug.bat](../make_debug.bat)) which can be used in the same way as the scripts for Linux:

First you have to compile the dependencies.
This step only has to be done once.
Run the commands below from the Visual Studio Developer Command Line:

```batch
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
git submodule update --init
make_externals.bat
```

This will clone the repository to `cosmoscout-vr` configure and build all externals in `cosmoscout-vr\build\windows-externals` and will install them to `cosmoscout-vr\install\windows-externals`.
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
make_release.bat
```

This will configure and build CosmoScout VR in `cosmoscout-vr\build\windows-release` and will install it to `cosmoscout-vr\install\windows-release`.
The application can be executed with:

```batch
cd install\windows-release\bin
start.bat
```

> :information_source: _**Tip:** If you wish, you can delete the directories `build` and `install` at any time in order to force a complete reconfiguration or re-installation._

<p align="center">
  <a href="dependencies.md">&lsaquo; Dependencies</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="using.md">User Interface &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>