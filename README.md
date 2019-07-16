<p align="center"> 
  <img src ="resources/logo/large.svg" />
</p>

CosmoScout VR is a virtual 3D-universe which lets you explore, analyze and present huge planetary datasets and large simulation data in real-time.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![c++17](https://img.shields.io/badge/C++-17-orange.svg)
[![source loc](https://img.shields.io/badge/source_loc-9.4k-green.svg)](cloc.sh)
[![plugin loc](https://img.shields.io/badge/plugin_loc-12.7k-green.svg)](cloc.sh)
[![comments](https://img.shields.io/badge/comments-2.6k-yellow.svg)](cloc.sh)
[![gitter](https://badges.gitter.im/cosmoscout/cosmoscout.svg)](https://gitter.im/cosmoscout/community)

CosmoScout uses C++17 and OpenGL. It can be build on Linux (GCC) and Windows (MSVC). Nearly all dependencies are included as [git submodules](externals), please refer to the section [Build Instructions](#build-instructions) in order to get started.

We try to add as many comments to the source code as possible. The number of source code lines and comment lines above is computed with the script [cloc.sh](cloc.sh). This script only counts *real comments*. Any dumb comments (such as copy-right headers or stuff like `/////////`) are not included in this number.

# Getting Started

We are happy to receive contributions to CosmoScout VR in the form of **merge requests** via Github. Feel free to fork the repository, implement your changes and create a merge request to the `develop` branch.

Further information on how to contribute can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

## Build Instructions

This software can be build in debug and release mode on Linux and Windows. Below you find the generic build instructions. Most dependencies are included as [git submodules](externals). You will only need a copy of [CMake](https://cmake.org/) (version 3.12 or greater), [Boost](https://www.boost.org/) (version 1.69 or greater) and a recent C++ compiler (GCC 8 or MSVC 19). For the compilation of the externals [Python](https://www.python.org/) is also required.

### Linux

On Linux, one can either use the provided shell scripts ([make_release.sh](make_release.sh) and [make_debug.sh](make_debug.sh)) or build the software manually using CMake. **Using the provided scripts** is easy and definitely the recommended way.

In any way, first you have to compile the dependencies. This step only has to be done once.

```shell
mkdir cosmoscout
cd cosmoscout
git clone git@github.com:cosmoscout/cosmoscout-vr.git src
cd src
git submodule update --init
cd ..
src/make_externals.sh
```

This will clone the repository to `cosmoscout/src` configure and build all externals in `cosmoscout/build/linux-externals` and will install them to `cosmoscout/install/linux-externals`. You can delete the directories in `cosmoscout/build` and `cosmoscout/install` at any time in order to force a reconfiguration or re-installation. Now you can compile CosmoScout VR:

```shell
src/make_release.sh
```

This will configure and build CosmoScout VR in `cosmoscout/build/linux-release` and will install it to `cosmoscout/install/linux-release`. You can delete the directories in `cosmoscout/build` and `cosmoscout/install` at any time in order to force a reconfiguration or re-installation. The application can be executed with:

```shell
cd install/linux-release/bin
./start.sh
```

For **manual compilation** follow the steps outlined in [make_release.sh](make_release.sh) or [make_debug.sh](make_debug.sh).

### Windows

For Windows, there are batch scripts ([make_release.bat](make_release.bat) and [make_debug.bat](make_debug.bat)) which can be used in the same way as the scripts for Linux:

First you have to compile the dependencies. This step only has to be done once. Run the commands below from the Visual Studio Developer Command Line:

```batch
mkdir cosmoscout
cd cosmoscout
git clone git@github.com:cosmoscout/cosmoscout-vr.git src
cd src
git submodule update --init
cd ..
src\make_externals.bat
```

This will clone the repository to `cosmoscout\src` configure and build all externals in `cosmoscout\build\windows-externals` and will install them to `cosmoscout\install\windows-externals`. You can delete the directories in `cosmoscout\build` and `cosmoscout\install` at any time in order to force a reconfiguration or re-installation. Now you can compile CosmoScout VR:

```batch
src\make_release.bat
```

This will configure and build CosmoScout VR in `cosmoscout\build\windows-release` and will install it to `cosmoscout\install\windows-release`. You can delete the directories in `cosmoscout\build` and `cosmoscout\install` at any time in order to force a reconfiguration or re-installation. The application can be executed with:

```batch
cd install\windows-release\bin
start.bat
```

## Using the application

There is a more in-depth tutorial available in [the doc folder](doc/intro.html). Once cloned, open this file in your web-browser to get started!


# Complete List of Dependencies

The list below contains all dependencies of CosmoScout VR. Besides Boost, all of them are included either as [git submodules](externals) or directly in the source tree. Some of the dependencies are only required by some plugins.

| Library | License |
|---|---|
| Alegreya Sans Font | [Open Font License ](https://fonts.google.com/specimen/Alegreya+Sans) |
| Boost (chrono, filesystem, date_time, system, thread) | [Boost Software License](http://www.boost.org/LICENSE_1_0.txt) |
| Bootstrap Date Picker | [Apache License 2.0](https://github.com/uxsolutions/bootstrap-datepicker/blob/master/LICENSE) |
| c-ares | [MIT](https://c-ares.haxx.se/license.html) |
| Chromium Embedded Framework | [BSD](https://bitbucket.org/chromiumembedded/cef/raw/a5a5e7ff08129f4122437dfdbba93d2a746c5c59/LICENSE.txt) |
| Curl | [MIT style](https://curl.haxx.se/legal/licmix.html) |
| CurlPP | [MIT style](https://github.com/jpbarrette/curlpp/blob/master/doc/LICENSE) |
| D3.js | [BSD 3-Clause](https://github.com/d3/d3/blob/master/LICENSE) |
| FreeGlut | [MIT](https://sourceforge.net/p/freeglut/code/HEAD/tree/trunk/freeglut/freeglut/COPYING) |
| fuzzyset.js | [BSD](https://github.com/Glench/fuzzyset.js) |
| glew | [Modified BSD](http://glew.sourceforge.net/glew.txt) |
| glm | [Happy Bunny / MIT](https://glm.g-truc.net/copying.txt) |
| GTest | [BSD 3-Clause](https://github.com/google/googletest/blob/master/googletest/LICENSE) |
| jQuery | [MIT](https://jquery.org/license/) |
| JsonHPP | [MIT](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT) |
| LibTiff | [BSD-like](http://www.libtiff.org/misc.html) |
| Material Icons | [Apache License 2.0](https://github.com/google/material-design-icons/blob/master/LICENSE) |
| MaterializeCSS | [MIT](https://github.com/Dogfalo/materialize/blob/master/LICENSE) |
| noUiSlider | [WTFPL](http://www.wtfpl.net/about/) |
| OpenSG | [LGPL](https://sourceforge.net/p/opensg/code/ci/master/tree/COPYING) |
| OpenVR | [BSD 3-Clause](https://github.com/ValveSoftware/openvr/blob/master/LICENSE) |
| SPICE | [Custom License](https://naif.jpl.nasa.gov/naif/rules.html) |
| STBImage | [Public Domain](https://github.com/nothings/stb/blob/master/docs/why_public_domain.md) |
| TinyOBJ | [MIT](https://github.com/syoyo/tinyobjloader/blob/master/LICENSE) |
| Ubuntu Font | [Ubuntu font licence](https://www.ubuntu.com/legal/terms-and-policies/font-licence) |
| Vista | [LGPL](https://sourceforge.net/projects/vistavrtoolkit/) |
| VRPN | [Boost Software License](https://github.com/vrpn/vrpn/wiki/License) |
| zlib | [MIT style](https://zlib.net/zlib_license.html) |

## Additional Run Time Dependencies

| Library | License |
|---|---|
| Apache built by [Apachehaus](https://www.apachehaus.com/cgi-bin/download.plx) | [Apache License 2.0](https://www.apache.org/licenses/) |
| GDAL | [X11/MIT](https://trac.osgeo.org/gdal/wiki/FAQGeneral#WhatlicensedoesGDALOGRuse) |
| Mapserver | [MIT](http://mapserver.org/copyright.html) |

# MIT License

Copyright (c) 2019 German Aerospace Center (DLR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
