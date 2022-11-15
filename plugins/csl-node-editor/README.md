<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

<p align="center"> 
  <img src ="img/banner-node-editor.png" />
</p>

# Node Editor Library for CosmoScout VR

This plugin library provides a web-server class which serves a highly configurable node editor over an HTTP interface.
It can be used to create all kinds of data flow editors for CosmoScout VR.

The node editor creates a web server which serves a web frontend on a given port via HTTP. The user can access this frontend with a web browser and start creating a node graph. For each created node or connection, a C++ counterpart is instantiated on the backend. Any data flow happens on the C++ side, the HTML / JavaScript graph is "just" a visualization of the graph. Whenever a node in the graph needs to display some data, a message needs to be sent from the C++ backend to the JavaScript frontend. Similarly, whenever the user modifies the graph on the frontend, a message is sent to the C++ backend. Custom node types have to inherit from the Node class. This base class provides methods for communicating with the JavaScript counterpart of the node. Internally, the communication happens via a web socket. At any given time, there can be only one open connection to a client. If an additional client connects, an error message will be shown instead of the frontend web page.