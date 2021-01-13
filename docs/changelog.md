<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Changelog of CosmoScout VR

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
* Documentation on [how-to setup a map server](https://github.com/cosmoscout/cosmoscout-vr/tree/develop/plugins/csp-lod-bodies) for `csp-lod-bodies` has been written.
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
