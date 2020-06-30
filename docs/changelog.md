<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Changelog of CosmoScout VR

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
  <a href="citation.md">How to cite CosmoScout VR &rsaquo;</a>
</p>
