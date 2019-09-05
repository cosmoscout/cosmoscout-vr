<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Configuring CosmoScout VR

When you launch CosmoScout VR via the start scripts ([`start.sh`](../config/base/scripts/start.sh) or
[`start.bat`](../config/base/scripts/start.bat)), two main configuration files are passed as command line arguments.

The first ([`simple_desktop.json`](../config/base/scene/simple_desktop.json)) configures your virtual _scene_.
This includes for example the simulation time, the observer position in space and the configuration for each and every plugin.
The second file ([`vista.ini`](../config/base/vista/vista.ini)) configures your _hardware setup_ - your input devices, the screens to render on, the setup of your render cluster and much more. 

## Scene Configuration

The most simple scene configuration file contains the JSON data below.
It will produce an empty universe with the virtual camera being centered on where usually the earth would have been.
In the following, the individual parameters are explained and the required steps for populating the black universe are outlined.

```javascript
{
  "startDate": "today",
  "observer": {
    "center": "Earth",
    "frame": "IAU_Earth",
    "distance": 10000000.0,
    "longitude": 11.281067,
    "latitude": 48.086709
  },
  "spiceKernel": "../share/config/spice/simple.txt",
  "widgetScale": 0.6,
  "enableMouseRay": false,
  "anchors": {},
  "plugins": {}
}
```

* **`"startDate"`:** This should be either `"today"` or in the format `"1950-01-02 00:00:00.000"` and determines the initial simulation time.
* **`"observer"`:** Specifies the initial position of the virtual camera. `"center"` and `"frame"` define the initial SPICE reference frame; `"distance"` (in meters), `"longitude"` and `"latitude"` (in degree) the 3D-position inside this reference frame.
For more background information on SPICE reference frames, you may read [this document](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf). 
* **`spiceKernel`:** The path to the SPICE meta kernel. If you want to start experimenting with SPICE, you can read the [SPICE-kernels-required-reading document](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html). 
However, the included [meta kernel](../config/base/spice/simple-linux.txt) contains already data for many of the solar system's bodies from 1950 to 2050.
* **`"widgetScale"`:** This factor specifies the initial scaling factor for world-space UI elements.
You can modify this if in your screen setup the 3D-UI elements seem too large or too small.
* **`"enableMouseRay"`:** In a virtual reality setup you want to set this to `true` as it will enable drawing of a ray emerging from your pointing device.
* **`"anchors"`:** This item contains an object for each and every celestial anchor you are using later in the config. Take this as an example:
  ```javascript
  "anchors": {
    "Moon": {
      "center": "Moon",
      "frame": "IAU_Moon",
      "startExistence": "1950-01-02 00:00:00.000",
      "endExistence": "2049-12-31 00:00:00.000"
    },
    ...
  }
  ```
  Now if you want to attach a simple body or a trajectory to this anchor, the configuration of the respective plugins will only refer to `"Moon"`. `"center"` and `"frame"` define the SPICE reference frame, `"startExistence"` and `"endExistence"` should match the data coverage of your SPICE kernels.
* **`"plugins"`:** This item contains an object for each plugin.
The name of the object must match the library name of the plugin.
In the example below, CosmoScout VR will attempt to load a plugin library called `../share/plugins/libcsp-simple-bodies.so` (`..\share\plugins\csp-simple-bodies.dll` on Windows). 
  ```javascript
  "plugins": {
    "csp-simple-bodies": {
      "bodies": {
        "Moon": {
          "texture": "../share/resources/textures/moon.jpg"
        },
        ...
      }
    },
    ...
  }
  ```
  The content of the objects in `"plugins"` is directly passed to the loaded plugin.
  For more information on the configuration of the plugins, please refer to the repositories of the [individual plugins](../README.md#Plugins-for-CosmoScout-VR).
  In the example above, a simple textured sphere will be attached to the celestial anchor `"Moon"` we defined earlier in the `"anchors"` object.

  :information_source: _**Tip:** Since CosmoSout VR will ignore items in `"plugins"` for which no plugin library is found, an easy way to temporarily disable a plugin is modifying its name:_ 
  ```javascript
  ...
  "#csp-simple-bodies": {
  ...
  ```
* **`"gui"`:** This is an optional top-level configuration which is required for virtual reality scenarios.
If omitted, the UI is directly drawn to (and resized with) the window.
However, in typical VR-scenarios you want to draw the UI somewhere onto a _virtual screen_.
The `"gui"` object defines the size and position of this _virtual screen_.
As an example, you can have a look at the provided [`simple_vive.json`](../config/base/scene/simple_vive.json) file:
  ```javascript
  "gui": {
    "heightMeter": 1.8,
    "heightPixel": 1200,
    "posXMeter": 0.0,
    "posYMeter": 1.2,
    "posZMeter": -1.0,
    "rotX": 0.0,
    "rotY": 0.0,
    "rotZ": 0.0,
    "widthMeter": 2.4,
    "widthPixel": 1600
  }
  ```

## ViSTA Configuration

:construction: _**Under Construction:** This guide is still far from complete. We will improve it in the future._

Thanks to the ViSTA framework, it is possible to run CosmoScout VR on a desktop, on the HTC-Vive, a tiled display or a six-sided CAVE without any modifications to the source code.
While this is very versatile, configuring ViSTA is also a very complex and involved topic.
This guide will give you some information on how to get started.

:information_source: _**Tip:** If you need help configuring CosmoScout VR for your complex setup, we may be able to provide some help. Just get in touch via cosmoscout@dlr.de_

### [`vista.ini`](../config/base/vista/vista.ini)

This is the entry file which is given as parameter to the `cosmoscout` executable.
While you could put the entire configuration in this file, it is a good practice to modularize the configuration and use separate `.ini`-files for different aspects.
The example configuration uses two further files: [`interaction.ini`](../config/base/vista/interaction.ini) and [`display.ini`](../config/base/vista/display.ini).

### [`interaction.ini`](../config/base/vista/interaction.ini)

This file specifies which ViSTA device drivers should be loaded and how the device's input data is fed to the application.
The data-flow itself is modelled in separate `.xml`-files which you can find in the [`config/base/vista/xml`](../config/base/vista/xml) directory.

### [`display.ini`](../config/base/vista/display.ini)

This file describes your window geometry, projection and stereo mode.

## Advanced Customization

While you can start by modifying the provided configuration files ([`simple_desktop.json`](../config/base/scene/simple_desktop.json) and [`simple_vive.json`](../config/base/scene/simple_vive.json)), you will most likely end up with a bunch of configuration files which are very specific for your setup.
Therefore you should not commit them to the source tree of CosmoScout VR - however, you can still use git for version control!

The trick is, to create a separate git repository for your configuration files and clone this repository to `cosmoscout-vr/config/<your-custom-config-dir>`.
All directories in `cosmoscout-vr/config` except for the `base` directory are ignored by git (see [.gitignore](../.gitignore)).

The [`CMakeLists.txt`](../config/CMakeLists.txt) in `cosmoscout-vr/config` automatically collects adds all subdirectories.
So you can install any scripts / configs / data files you like as part of CosmoScout's build process!

<p align="center">
  <a href="using.md">&lsaquo; Using CosmoScout VR</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="contributing.md">Contributing Guides &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>