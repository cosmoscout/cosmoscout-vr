<p align="center"> 
  <img src ="img/banner-phobos.jpg" />
</p>

# Configuring CosmoScout VR

When you launch CosmoScout VR via the start scripts ([`start.sh`](../config/base/scripts/start.sh) or
[`start.bat`](../config/base/scripts/start.bat)), two main configuration files are passed as command line arguments.

The first ([`config/base/scene/simple_desktop.json`](../config/base/scene/simple_desktop.json)) configures your virtual _scene_.
This includes for example the simulation time, the observer position in space and the configuration for each and every plugin.
The second file ([`config/base/vista/vista.ini`](../config/base/vista/vista.ini)) configures your _hardware setup_ - your input devices, the screens to render on, the setup of your render cluster and much more. 

## Scene Configuration

:warning: _**Warning:** The [default configuration](../config/base/scene/simple_desktop.jso) only contains a few data sets with very low resolution.
The configuration of additional data sets is not part of this guide.
Please read the documentation of the [individual plugins](../README.md#Plugins-for-CosmoScout-VR) for guides on how to include new data sets._

## ViSTA Configuration

<p align="center">
  <a href="using.md">&lsaquo; User Interface</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="contributing.md">Contributing Guides &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>