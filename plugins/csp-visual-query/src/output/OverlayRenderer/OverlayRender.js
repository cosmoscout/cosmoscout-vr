////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class OverlayRender extends Rete.Component {
  constructor() {
    // This name must match the RenderNode::sName defined in Render.cpp.
    super("OverlayRender");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single input. The first parameter is the name of this input and must be
    // unique amongst all sockets. It is also used in the RenderNode::process() to read the
    // input of this node. The second parameter is shown as name on the node. The last parameter
    // references a socket type which has been registered with the node factory before.
    let input = new Rete.Input('Image2D', "Image 2D", CosmoScout.socketTypes['Image2D']);
    node.addInput(input);

    return node;
  }
}
