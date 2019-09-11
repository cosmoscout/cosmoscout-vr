<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Changelog of CosmoScout VR

## [v1.1.0](https://github.com/cosmoscout/cosmoscout-vr/releases)

**Release Date:** 11 / 09 / 2019

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

**Release Date:** 30 / 08 / 2019

* Initial publication under the MIT license on Github.

<p align="center">
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="citation.md">How to cite CosmoScout VR &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>
