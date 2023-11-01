<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Changelog of CosmoScout VR

## [v1.8.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** TBD

#### New Features

* The `csp-timings` plugin now also shows the number of generated samples and primitives in the user interface.
* The `csp-timings` plugin now shows the timings in the world-space gui-area per default. The old behavior can be restored with `"useLocalGui": true` in the plugin's settings.
* A new "Ambient Occlusion" slider in the user interface can be used to control the amount of slope shading on the terrain.
* The water surface shader of `csp-atmospheres` has been improved significantly. It now reflects the sky and features some beautiful waves. The waves can be disabled as they are quite demanding in terms of GPU power. Both, the reflections and the waves are not physically based in any way; they are mostly intended for presentation purposes.
* Bodies in `csp-simple-bodies` can now be shaded by a ring. 

#### Refactoring

* The `csp-lod-bodies` plugin has received some major refactoring. Here are the main changes:
  * The terrain tiles are not stitched together anymore, instead, skirt polygons are drawn around the tiles to hide any seams.
  * The resolution of the tile's elevation and image data are now configurable (via the new `tileResolutionDEM` and `tileResolutionIMG` settings keys).
  * The image channel now uses RGBA instead of RGB internally in order to improve graphics performance thanks to proper four-byte alignment.
  * The number of maximum tile uploads to the GPU has been reduced from 20 to 5 per channel in order to reduce performance drops during tile loading.
  * It is not required anymore to set the `format` of the terrain data sources anymore.
  * There's a new `autoLodRange` option for setting the LoD-Factor range which is used if auto-lod is enabled.
* The `csp-atmospheres` plugin received a major refactoring. Here are the main changes:
  * It is now possible to add alternative atmospheric models to CosmoScout VR. As an example, the [excellent scattering model by Eric Bruneton](https://github.com/ebruneton/precomputed_atmospheric_scattering) has been included.
  * The attributes `enableWater`, `waterLevel`, `enableClouds`, and `cloudAltitude` are now stored per planet. Enabling water on Earth will not flood Mars anymore.
  * `waterLevel` and `cloudAltitude` are now given in meters, not relative to the atmosphere height anymore.
  * `cloudAltitude` can now be configured at runtime in the user interface.
  * In non-HDR mode, the atmosphere now performs filmic tonemapping which results in much less over-exposure around the Sun during sunset or sunrise.
  * The shaders are now loaded from files and not compiled into the binary.
  * No support for light shafts anymore. While not impossible, it would be pretty difficult to implement this properly with Eric Bruneton's model given that we use cascaded shadow maps. Furthermore, this feature used to have a terrible performance anyways.
  * No graphical tests of the atmosphere anymore. With the new architecture, it's not possible anymore to render an atmosphere without CosmoScout's core classes.

#### Other Changes

* A couple of changes and fixes were added in order to support much higher resolution map data:
  * Increased maximum HEALPix depth from 20 to 30. This reduces our minimum tile size from about 13 m to 13 mm.
  * Improved the scene scaling of the default configuration to allow for a smoother navigation close to the surface.
  * Fixed an issue which led to an accumulated error in the rotation quaternion of the observer.
  * Fixed an issue which caused precision issues for very small movements of the click-and-drag navigation.
  * The world-space depth reconstruction in the atmosphere shader now operates relative to the camera, resulting in a much higher precision if the user is close to the surface. This will allow us to create a very dense media, for example for under-water scenes.
* In order to improve the rendering performance, the stars of `csp-stars` are not drawn anymore if the observer is on the day-side of a planet with an atmosphere.
* The default exposure and glare values as well as the glare-slider mapping have been tweaked for a better appearance of the atmospheres in HDR mode.
* The default star rendering mode has been changed to `eSmoothDisc`
* When saving the scene with `CosmoScout.callbacks.core.save(...)`, optional properties of the celestial objects are not written anymore. This also fixes a warning about the Barycenter having no radii.

#### Bug Fixes

* The user interface now avoids rerenders of components that did not change. This lead to the whole UI rerendering most of the time. 
* There were some scenarios where corrupt tiff files in the map-cache directory weren't removed. Appropriate error handling has been added.
* Fixed loading of the LoD-factor of `csp-lod-bodies` from the settings files.


## [v1.7.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2022-12-13

#### New Features

* To allow shared code for plugins, without requiring it to be in the core libraries we introduced a new concept called plugin libraries. They are prefixed `csl` and sit beside the traditional plugins in the `plugins` folder.
* A new powershell script to create plugin libraries has been added: `tools/New-PluginLibrary.ps1`.
* **New Plugin Library: `csl-node-editor`.** With this library, plugins can provide an interface to create complex data flow graph. The node editor interface is made available via a web server. Hence the node graph can be modified either within CosmoScout VR or on an external device.
* **New Plugin: `csp-demo-node-editor`.**  An example on how to use the `csl-node-editor` plugin library for creating data flow graphs within CosmoScout VR.

#### Other Changes

* The branching scheme has been simplified. Instead of `master` and `develop`, there is now only a single `main` branch.

#### Refactoring

* The `tools` were moved from `cs-core` to a new plugin library called `csl-tools`. `csp-measurement-tools` are now making use of this new plugin library.

#### Bug Fixes

* The depth images which can be captured with `csp-web-api` are now in meters again.

## [v1.6.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2022-10-14

#### New Features

* Celestial bodies can now cast eclipse shadows onto other objects. This is now used by `csp-lod-bodies`, `csp-simple-bodies`, `csp-rings` as well as by `csp-atmospheres`. For now, this is computed by approximating the involved bodies as spheres as well as ignoring atmospheric effects.
* The height and slope maps used by `csp-lod-bodies` now support an alpha channel. This way, planets can be made selectively translucent.
* New default bodies are now available as `csp-simple-bodies`: 
  * Mars moons Phobos and Deimos
  * Dwarf planets Ceres and Vesta
  * Jupiter moons Io, Enceladus, Ganymede and Callisto
  * Saturn moons Mimas, Tethys, Dione, Rhea, Titan and Iapetus
  * Uranus moons Miranda, Ariel, Umbriel, Titania and Oberon
  * Neptune's moon Triton
  * The dwarf planet Pluto and his moon companion Charon


#### Other Changes

* CosmoScout VR now follows the [REUSE Specification](https://reuse.software/spec).
* CosmoScout VR now uses a reverse infinite projection for 3D rendering. This removes the far clipping plane and increases the depth precision for the entire scene signifcantly.
* Removed the obsolete "Mix with RGB" option for height or slope coloring of the LOD-Bodies.
* Lighting is now enabled by default
* It is now possible to have simple body textures, that have the prime meridian at the edge.
* Rings now have a lit and an unlit side.
* `CelestialObject`s now have four new properties:
  * `bodyCullingRadius`: This can be used to determine the visiblity of an object. If the object is too far away to be seen, plugins may want to skip updating or drawing attached things.
  * `orbitCullingRadius`:This can be used to determine the visiblity of an object's trajectory. If the object is really far away, plugins may want to skip updating or drawing attached things (like trajectories or anchor labels).
  * `trackable`: If set to `false`, the observer will not automatically follow this object.
  * `collidable`:  If set to `false`, the observer can travel through the object. 
* The mouse navigation has been made a bit smoother. This makes it possible to update the scene scale only once each frame. Before, it had to be updated multiple times each frame which introduced a significant CPU overhead during rapid movements.
* All plugins now save their configuration before being unloaded. This makes it possible that plugins keep their current state when being reloaded at run-time.
* Test source files are now only compiled into the executable and into the libraries if `COSMOSCOUT_UNIT_TESTS` is set to `true`.
* It's now possible to reload the `csp-satellites` plugin at run-time.
* Anchor Labels are now shown for each configured object per default. A blacklist has been added to the plugin configuration to remove labels selectively.
* Anchor Labels now show the name of the configured object rather than the SPICE center name.
* The maximum size of Anchor Labels has been increased to prevent line breaking for longer object names.
* Anchor Labels now use the new `orbitCullingRadius` of the object for deciding whether they should be shown or not.

#### Refactoring

* The `Settings` class now has a map of immutable `CelestialObject`s which are instantiated when CosmoScout is started. Before it used to have a list of anchor configurations which could be used to initialize `CelestialObject`s.
* The observer-relative transformation of each `CelestialObject` is updated once each frame by the `SolarSystem`.
* All classes which previously derived from `CelestialAnchor`, `CelestialObject`, or `CelestialBody` now get the observer-relative transformation from the `CelestialObject`s in the `Settings`.
* Classes which previously derived from `CelestialBody` now have to derive from the new class `CelestialSurface`. A `CelestialSurface` can then be assigned to a `CelestialObject`.
* `CelestialAnchorNode`s do not exist anymore. Classes which  used to use these, now have to use a `VistaTransformNode` instead and update its transformation once each frame with the observer-relative transformation from the respective  `CelestialObject`.
* Much per-frame logic used to be executed in overrides of `CelestialObject::update()`. This method does not exist any more; plugins now need to update their objects manually.
* Some common code has been moved from the measurement tools to the base classes `cs::core::tools::Tool` and `cs::core::tools::Mark`. This simplifies some code in the measurement tools.
* The interface of the `GuiManager` has been streamlined. Before, HTML templates could be added via `addHtmlToGui()` but had to be removed with an explicit JavaScript call. The same was true for adding and removing stylesheets. Here are all the corresponding changes:
  * `addScriptToGui()` has been removed. It was never used and is actually identical to `getGui()->executeJavascript()`
  * `addScriptToGuiFromJS()` has been renamed to `executeJavascriptFile` as this describes more clearly what the method actually does.
  * `addHtmlToGui()` has been renamed to `addTemplate()`, as this is what the method actually does.
  * A corresponding `removeTemplate()` has been added.
  * `addCssToGui()` has been renamed to `addCSS()`.
  * A corresponding `removeCSS()` has been added.

#### Bug Fixes

* Back-face culling of LOD-Bodies does now work properly.
* Fixed black LOD-Bodies if no image channel is selected initially.
* The `csp-wms-overlays` plugin now correctly uses jpeg instead of png, if an opaque layer is selected and jpeg is available.
* Fixed a warning which was printed if an GLTF model did not explicitly specify texture filtering parameters.

## [v1.5.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2022-02-25

#### New Features

* **New Plugin**: `csp-vr-accessibility` has been added, which can draw a stable floor grid and / or a field-of-view limiting vignette. Both features are designed to mitigate cyber sickness.
* **Physically-Based Rendering of Celestial Bodies**: The celestial bodies rendered with the `csp-lod-bodies` plugin can now be shaded with either a Lambert, an Oren-Nayer, or a Hapke BRDF.

#### Other Enhancements

* **Configurable Tone-Mapping**: The tone-mapping operator of the HDR-rendering path can now be chosen.
* Added a `CITATION.cff` to the project to make citing the project easier.
* Most external dependencies have been updated to the latest versions.
* The CI jobs on GitHub now run Ubuntu 20.04 (previously 18.04).

#### Bug Fixes

* The current animation time speed is now saved and loaded from configuration files.
* The download links for the SPICE kernel files have been updated.

## [v1.4.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2021-03-30

#### New Features

* **New plugin**: `csp-wms-overlays` has been added, which allows overlaying time dependent map data from Web-Map-Services (WMS) over bodies rendered by other plugins.
* In HDR-mode, the glaring can now be computed in a perspective-correct manner. While this is computationally more expensive, it increases immersion with large fields of view.

#### Other Enhancements

* The `csp-timings` has been rewritten and allows now for much more fine-grained timing analysis. (See [#230](https://github.com/cosmoscout/cosmoscout-vr/pull/230) and [#240](https://github.com/cosmoscout/cosmoscout-vr/pull/240)).
* A new star rendering method has been added, which scales the rendered discs based on the star's magnitude. This makes the star figures more easy to spot when HDR rendering is disabled.
* The billboard rendering method of stars has been improved by using an updated star texture.
* Documentation on [how-to setup a map server on WSL](https://github.com/cosmoscout/cosmoscout-vr/tree/main/plugins/csp-lod-bodies) for `csp-lod-bodies` has been written.
* Performance has been slightly improved by removing redundant uniform location queries from the render loop.

#### Bug Fixes

* Fixed rotation values of six-dof bookmarks.
* In HDR-mode, the surface shading of simple bodies and lod-bodies is now energy conserving.
* In HDR-mode, the luminance of the Sun is now computed correctly.


## [v1.3.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2020-11-24

The settings format has changed slightly in version 1.3.0, see the [Migration Guide](migration.md) for details.

#### New Features

* **New plugin**: `csp-web-api` has been added, which allows remote controlling a running instance of CosmoScout VR. It can also be used to capture screenshots over an http API.
* **New plugin**: `csp-recorder` has been added, which allows basic recording of high-quality videos.
* **New plugin**: `csp-minimap` has been added. This tiny plugin can be used to show a 2D-Map of the observer's current position on a planet.
* **New plugin**: `csp-custom-web-ui` has been added. This plugin allows to add any web content to the user interface or to the 3D scene.
* An **interactive JavaScript** console has been added to the user interface. It features auto-completion, a command history and can be used control CosmoScout VR with scripts.
* An experimental feature has been added which allows **saving and restoring the current scene**. For now, this is only available in the interactive JavaScript console.
  - Use `CosmoScout.callbacks.core.save("test.json")` to save the current scene.
  - Use `CosmoScout.callbacks.core.load("test.json")` to restore the previously saved scene at any later point in time.
* A **bookmark system** has been added. You can now create bookmarks in time and space. They will be visualized both on the timeline and by `csp-fly-to-locations`.
  - The location of a bookmark is defined by a SPICE anchor, an optional cartesian position and an optional rotation.
  - The time of a bookmark has an optional end parameter which makes the bookmark describe a time span rather a time point.
  - Location and time are both optional, but omitting both results in a pretty useless bookmark.
* Optional **High Dynamic Range Rendering** (HDR) which uses true luminance values has been added. This can be toggled at runtime.
* You can now **modify the field of view** by choosing a sensor size and a focal length.
* [spdlog](https://github.com/gabime/spdlog) is now our **new logging library**. The logger will print colourful messages to the console and store it's messages in a file called `cosmoscout.log`. 
* A new javascript API has been added which can be used to perform **forward and reverse geocoding**. This is used by `csp-measurement-tools` to show address information and by a new search bar below the timeline. The geo-coding for Earth uses OpenStreetMap, all other planets use CSV files obtained from https://planetarynames.wr.usgs.gov/AdvancedSearch. The search bar beneath the timeline which supports queries like
  - "berlin" Fly to something called like "Berlin" on the current planet.
  - "venus:" Fly to Venus
  - "mars:olymps mns" Fly to something called like "olymps mns" on Mars.
* The ability to **zoom WebViews** has been added. 
  - This is used to increase the DPI of the measurement tools.
  - Also, a slider has been added to the Graphics Settings to adjust the **overall scale of the main user interface**. This is an initial step to properly support high resolution screens. On a 4k-15"-Laptop-Screen you can now simply set the Main UI Scale to 2.0.
* The possibility to specify a **fixed sun direction** has been added. This can be used to create artificial lighting conditions.

#### Other Enhancements

* Significantly improved star rendering. Several rendering modes are implemented and can be toggled at runtime.
* The plugins `csp-atmospheres`, `csp-sharad`, `csp-measurement-tools`, `csp-simple-bodies` and `csp-lod-bodies` now use **true three-axes ellipsoids** for rendering. Before they used to perform math on spheres.
* All default **plugins are now part of the source tree** and no individual submodules anymore. This simplifies the software development cycle significantly.
* **Plugins can now be reloaded at run time**. This allows faster development cycles as code modifications can be injected while CosmoScout VR is running.
* Data which is downloaded at application start-up is now stored im temporary *.part files until the download finished. **This prevents corrupt files in case the download fails**.
* The selection handling of CosmoScout VR has been refactored. It's now possible to insert text into text boxes even if the mouse is not hovering it.
* GuiItems can now ignore scroll events.
* **Draggable windows**: Some CSS classes and some JavaScript code have been added to allow the creation of draggable windows in the main user interface. This is used for the calendar. 
* JavaScript callbacks can now take **optional arguments** (specified as `std::optional`)
* **More intuitive signature for multi-handle slider callbacks**. Rather than a value + the ID of the changed handle they now simple get all slider handle values.
* Updated vis-timeline.js to the latest stable version. This increased timeline rendering performance quite significantly.
* Documentation on [how-to setup a map server](https://github.com/cosmoscout/cosmoscout-vr/tree/main/plugins/csp-lod-bodies) for `csp-lod-bodies` has been written.
* `csp-lod-bodies` can now show tile bounding boxes for debugging purposes.
* Styling of UI elements has received a make-over.
* On Linux, CosmoScout VR's **window has now an icon** and a name. These are required to properly represent the window in the taskbar, the Alt-Tab application switcher and in other places.
* The **date display** in the center of the screen now **shows UTC** in the less German-looking YYYY-MM-DD HH:MM:SS.
* It's now possible to zoom on the timeline even if time is playing.
* Vertex position and normal calculation for `csp-lod-bodies` have been improved slightly.
* Vista and OpenSG are now built with **precompiled headers and unity builds** improving the build time significantly.
* Vista's **HTC Vive Driver** has been updated to work with the latest SteamVR.
* The **Optitrack** device driver of Vista works now with the latest version of NatNet SDK.
* CosmoScout VR is now build with MSVC, Clang and GCC with **all warnings enabled** and any warnings are treated as errors.

#### Bug Fixes

* On Windows, the mouse pointer now shows a hand symbol when hovering a hyperlink (and no question mark anymore).
* We actually mixed local timezones and UTC in our code. Now the cs::utils::convert::time namespace makes sure that we always stay in UTC.
* The conversion from boost::posix_time::ptime to BDT of SPICE was wrong - we did not include leap seconds! Therefore the simulation time and visualized time were off by a few minutes.
* Allow multiples calls to CelestialObserver::moveTo per frame: Even if the animation time was set to zero, it took one frame to update the observer's position and rotation. Hence multiple calls to CelestialObserver::moveTo would "overwrite" each other.
* Multisampling now works in HDR-Mode as well.
* SPICE frames are now updated before the plugins. This resolves some issues where plugin code would use object's position from the last frame.
* Moving the timeline will properly restore the playback speed once the drag operation is finished.
* Correct matrices for shadow frustum culling are now used. This results in less shadow-popping artifacts.

## [v1.2.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2020-02-11

#### New Features

* A **timeline** has been added to the user interface. This allows for quick temporal navigation and adjustments of the simulation time speed.
* The loading screen now shows an **animated progress bar**. Also, the loading screen now has a fade-out effect.
* Asset files (including SPICE kernels) can now be **downloaded at startup**. The files to download are specified in the scene configuration file. The download progress is shown on loading screen.
* Support for **Unit Tests** has been added. This also adds support for **_graphical_ tests**. These tests require an OpenGL context and will open a window; oftentimes they capture a screenshot and compare the result to a reference image. In order to make this possible even if there is no graphics card or display attached, they require [Xvfb](https://en.wikipedia.org/wiki/Xvfb) and [imagemagick](https://imagemagick.org/index.php).
* **Github Actions** is now used instead of Travis for continuous integration of CosmoScout VR. These features are configured:
  * **Binary builds** of CosmoScout VR for Linux and Windows whenever a tag is pushed.
  * Clang-Format: For each and every push event, a job is executed which checks whether the code base obeys our clang-format rules.
  * For pull requests only, a job is run which analyses the amount of comments in the source tree. This test will pass if the percentage of comments did not decrease.
  * CosmoScout VR and all of its dependencies are compiled and tests are run. As this job takes quite some time, it is only executed for the events below:
    * For each push to `master` or `develop` if the commit message does not contain `[no-ci]`.
    * For each push to any branch if commit message contains `[run-ci]`.
    * For pull requests.

#### Other Enhancements

* The user interface code of CosmoScout VR has been **vastly restructured**. Most importantly, the main user interface now consists of **one HTML page only**. This simplifies development of plugins which have to modify the UI. Furthermore, the API between C++ and JavaScript is now much cleaner. Other UI related changes include: 
  * The middle mouse button and double click events are now supported for UI elements.
  * Chromium Embedded Framework  has been upgrade to version 79.
* Other parts of the code base have been refactored, leading to
  * a **clean shutdown** - no more crashes on windows when the CosmoScout VR window is closed and
  * several **performance improvements**.
* An icon has been assigned to the CosmoScout VR executable on Windows.
* An icon has been assigned to the CosmoScout VR window on Windows.
* The high performance GPU is now automatically selected on Windows.
* A notification is now shown when travelling to a location.
* A lot of **documentation** has been added:
  * Explanation of CosmoScout VR's coordinate systems.
  * Guides on how to setup various IDEs for CosmoScout VR development.
  * A list of required software packages for Linux
  * A guide on how-to get a matching version of Boost
* More Space Mice from 3DConnexion are now supported.
* `develop` is now the default branch for CosmoScout VR and its plugins.

#### Bug Fixes

* An issue has been fixed which leads to an instant zoom-to-surface effect when CosmoScout VR runs at low frame rates and the mouse wheel is scrolled quickly.
* The Sun's position did not change at dates before 2000-01-01.
* The axis formatting in the path tool did not show the fractional part of decimal numbers.

## [v1.1.1](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2019-09-20

#### Bug Fixes

* Fixed a regression which messed up the display of world space user interfaces 

## [v1.1.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2019-09-11

#### New Features

* If built from the git source tree, the current branch and commit hash is shown on the loading screen.
* If there are errors in the scene configuration file, detailed error information is now printed to the console.
* Selection management in the user interface has been improved significantly: 
When the left mouse button is pressed over an item, this item will receive input events until the mouse button is released regardless of the mouse position on the screen.
* The user interface now uses bootstrap instead of materializecss. This makes the code easier to maintain and more future proof.
* Scene and ViSTA configuration files can now be passed as arguments to the Windows start script.
* The make_* scripts now support passing arguments to CMake.
* The application can now be built with Visual Studio 2019.

#### Other Enhancements

* The software has been made citable by registering a DOI on zenodo.org.
* The WMS configuration of `csp-lod-bodies` is now directly in the scene settings.
* This changelog has been created.
* Release management using Github Milestones has been set up.
* A Code of Conduct has been created.
* A guide on where to submit issues has been created.
* Issue templates have been added.
* Some documentation regarding ViSTA configuration has been added.
* Line endings in the repository are now handled properly by git.
* Most shaders now require only GLSL `#version 330`, only a few related to physically based rendering require up to `#version 450`.
* No shaders require `compatibility` mode anymore.
* The external submodules use absolute URLs now. Hence it is now possible to mirror the repository to other services (such as gitlab.com).

#### Bug Fixes

* On some systems, the boost system library was not installed correctly.
* Several typos in the documentation have been fixed.

## [v1.0.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 2019-08-30

* Initial publication under the MIT license on Github.

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="code_of_conduct.md">&lsaquo; Code of Conduct</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="migration.md">Migration Guide &rsaquo;</a>
</p>
