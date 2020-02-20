<p align="center"> 
  <img src ="img/banner-mars.jpg" />
</p>

# Complete List of Dependencies

The list below contains all dependencies of CosmoScout VR. Besides Boost, all of them are included either as [git submodules](../externals) or directly in the source tree.
Some of the dependencies are only required by some plugins.

A text document containing [all individual license texts](../LICENSE-3RD-PARTY.txt) is provided in the root directory of the source tree.

| Engine Dependencies | Description | License |
|:---|:---|:---|
| [Boost (system, chrono, filesystem, date_time)](http://www.boost.org) | Used for time conversions and file system operations. | [Boost Software License](http://www.boost.org/LICENSE_1_0.txt) |
| [c-ares](https://c-ares.haxx.se) | A dependency of curl, used for asynchronous DNS requests. | [MIT](https://c-ares.haxx.se/license.html) |
| [Chromium Embedded Framework](https://bitbucket.org/chromiumembedded/cef) | For the Webkit based user interface. | [BSD](https://bitbucket.org/chromiumembedded/cef/raw/a5a5e7ff08129f4122437dfdbba93d2a746c5c59/LICENSE.txt) |
| [Curl](https://curl.haxx.se) | Library for downloading stuff from the Internet. | [MIT style](https://curl.haxx.se/legal/licmix.html) |
| [CurlPP](https://github.com/jpbarrette/curlpp) | C++ wrapper for curl. | [MIT style](https://github.com/jpbarrette/curlpp/blob/master/doc/LICENSE) |
| [FreeGlut](https://sourceforge.net/p/freeglut) | Windowing toolkit dependency of OpenSG. | [MIT](https://sourceforge.net/p/freeglut/code/HEAD/tree/trunk/freeglut/freeglut/COPYING) |
| [glew](http://glew.sourceforge.net) | OpenGL extension wrangler. | [Modified BSD](http://glew.sourceforge.net/glew.txt) |
| [glm](https://glm.g-truc.net) | Math library used throughout CosmoScout. | [Happy Bunny / MIT](https://glm.g-truc.net/copying.txt) |
| [JsonHPP](https://github.com/nlohmann/json) | Parses json files in C++. | [MIT](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT) |
| [LibTiff](http://www.libtiff.org) | For .tiff image format support. | [BSD-like](http://www.libtiff.org/misc.html) |
| [OpenSG](https://sourceforge.net/p/opensg) | Scenegraph used as backend by Vista. | [LGPL](https://sourceforge.net/p/opensg/code/ci/master/tree/COPYING) |
| [OpenVR](https://github.com/ValveSoftware/openvr) | Adds support for the HTC-Vive. | [BSD 3-Clause](https://github.com/ValveSoftware/openvr/blob/master/LICENSE) |
| [spdlog](https://github.com/gabime/spdlog) | Fast C++ logging library. | [MIT](https://github.com/gabime/spdlog/blob/v1.x/LICENSE) |
| [SPICE](https://naif.jpl.nasa.gov/naif) | Library to compute positions of celestial objects. | [Custom License](https://naif.jpl.nasa.gov/naif/rules.html) |
| [STBImage](https://github.com/nothings/stb) | Library for loading .jpg and .png files. | [Public Domain](https://github.com/nothings/stb/blob/master/docs/why_public_domain.md) |
| [Vista](https://sourceforge.net/projects/vistavrtoolkit/) | VR-Framework for scenegraphs, distributed rendering and low-level VR-device access. | [LGPL](https://sourceforge.net/projects/vistavrtoolkit/) |
| [VRPN](https://github.com/vrpn/vrpn) | Used for supporting various hardware devices. | [Boost Software License](https://github.com/vrpn/vrpn/wiki/License) |
| [zlib](https://zlib.net) | Dependency of Vista. | [MIT style](https://zlib.net/zlib_license.html) |
| **UI Dependencies** | **Description** | **License** |
| [Alegreya Sans Font](https://fonts.google.com/specimen/Alegreya+Sans) | This font is used for the CosmoScout VR logo. | [Open Font License ](https://fonts.google.com/specimen/Alegreya+Sans) |
| [Bootstrap](https://github.com/twbs/bootstrap) | Main framework for the user interface. | [MIT](https://github.com/twbs/bootstrap/blob/master/LICENSE) |
| [Bootstrap Date Picker](https://github.com/uxsolutions/bootstrap-datepicker) | The calendar in the user interface. | [Apache License 2.0](https://github.com/uxsolutions/bootstrap-datepicker/blob/master/LICENSE) |
| [Bootstrap Select](https://github.com/snapappointments/bootstrap-select/) | Used for dropdowns in the UI. | [MIT](https://github.com/snapappointments/bootstrap-select/blob/v1.13.0-dev/LICENSE) |
| [Color Picker](https://tovic.github.io/color-picker/) | Used for color pickers in the UI. | [MIT](https://github.com/tovic/color-picker/blob/master/LICENSE) |
| [D3.js](https://github.com/d3/d3) | Used for some tools (like path measurement) to draw graphs in the user interface. | [BSD 3-Clause](https://github.com/d3/d3/blob/master/LICENSE) |
| [fuzzyset.js](https://github.com/Glench/fuzzyset.js) | Used for the location search on other planets and moons. | [BSD](https://github.com/Glench/fuzzyset.js) |
| [jQuery](https://jquery.org) | JavaScript library which is used extensively in the user interface. | [MIT](https://jquery.org/license/) |
| [Material Icons](https://github.com/google/material-design-icons) | Icon set uses in the user interface. | [Apache License 2.0](https://github.com/google/material-design-icons/blob/master/LICENSE) |
| [noUiSlider](https://refreshless.com/nouislider) | JavaScript library for advanced sliders. | [WTFPL](http://www.wtfpl.net/about/) |
| [Ubuntu Font](https://design.ubuntu.com/font) | This font is used in the user interface of CosmoScout VR. | [Ubuntu font licence](https://www.ubuntu.com/legal/terms-and-policies/font-licence) |
| [vis-timeline](https://github.com/visjs/vis-timeline) | This is used for the timeline. | [Apache License 2.0](https://github.com/visjs/vis-timeline/blob/master/LICENSE-APACHE-2.0) or [MIT](https://github.com/visjs/vis-timeline/blob/master/LICENSE-MIT)|
| **Runtime Dependencies** | **Description** | **License** |
| [Apache built by Apachehaus](https://www.apachehaus.com/cgi-bin/download.plx) | This is a pre-compiled apache server for windows which can be used to run your own Mapserver. Only required by the `csp-lod-bodies` plugin. | [Apache License 2.0](https://www.apache.org/licenses/) |
| [GDAL](https://trac.osgeo.org/gdal) | Dependency of the Mapserver. Only required by the `csp-lod-bodies` plugin. | [X11/MIT](https://trac.osgeo.org/gdal/wiki/FAQGeneral#WhatlicensedoesGDALOGRuse) |
| [Mapserver](http://mapserver.org) | Used to provide map data over the internet. Only required by the `csp-lod-bodies` plugin. | [MIT](http://mapserver.org/copyright.html) |
| [proj.4](https://proj.org) | The `csp-lod-bodies` requires a [special version](https://github.com/cosmoscout/proj.4) of this library to be used by the Mapserver. | [MIT](https://proj.org/about.html) |

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="architecture.md">&lsaquo; Software Architecture</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="coordinate-systems.md">Coordinate Systems &rsaquo;</a>
</p>
